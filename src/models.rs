//! Loading `histo_kir_hmm`'s trained profile HMMs, and keeping them loaded.
//!
//! A model directory is exactly what `histo_kir_hmm.serialization.save_models` writes: one
//! `<class>.npz` per model, a `background.npy`, and a `manifest.json`. We vendor a copy (see
//! `codegen/vendor_models.py`), and `model_dir=` lets a caller point at their own.
//!
//! Four things here are load-bearing for parity with `histo_kir_hmm.KIRClassifier`:
//!
//! * **Manifest order.** `gene_models` and `domain_models` are the orders the reference's
//!   `dict`s iterate in, which is the order Python's `max()` and stable `sorted()` fall back on
//!   when two models score identically. Preserve them exactly; never re-sort.
//! * **The calibration block.** `per_residue_threshold`, `per_residue_width`, `tie_tolerance`
//!   and `domain_threshold` are measured numbers from training, not tuning knobs. They are read
//!   from the manifest, never defaulted to something plausible.
//! * **`exact_index` and `cross_gene_ectodomains`.** These carry the irreducible ties (invariant
//!   3/4): 127 IPD-KIR alleles have an ectodomain byte-identical to an allele of another gene.
//!   The index is keyed on a 16-hex-character sha256 prefix of the scored region; the cross-gene
//!   list holds the colliding *sequences* so containment can be tested in both directions,
//!   because a real query is a precursor or a structural chain and is never byte-equal to a
//!   199-residue training extent.
//! * **`per_match_state_stats`.** The mean/std that `typicality` is a percentile under.
//!
//! The cache exists because inflating 18 archives yields ~1 MB of f64 and takes milliseconds;
//! per row that would dwarf the actual work, so a `ModelSet` is loaded once per directory per
//! process and shared.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;
use serde::Deserialize;

use crate::npy;

/// The amino-acid alphabet `histo_kir_hmm` trains against (`histo_kir_hmm/alphabet.py`).
pub const ALPHABET_SIZE: usize = 20;
/// Transition classes per model position: MM, MI, MD, IM, II, DM, DD.
pub const N_TRANS: usize = 7;

// Transition indices, matching histo_kir_hmm/profile_hmm.py.
pub const MM: usize = 0;
pub const MI: usize = 1;
pub const MD: usize = 2;
pub const IM: usize = 3;
pub const II: usize = 4;
pub const DM: usize = 5;
pub const DD: usize = 6;

/// One trained profile HMM, stored **transposed** relative to numpy.
///
/// numpy stores these as `(L+1) x 20` (and `(L+1) x 7`), so a Viterbi row -- which sweeps `j`
/// with the residue fixed -- would stride by 20 f64 and touch a fresh cache line every step.
/// Transposing to `20 x (L+1)` makes each sweep a contiguous run. `read_model` transposes on
/// load; nothing downstream sees numpy's layout. The arithmetic is untouched: the max-plus
/// recursion visits the same values in the same order.
pub struct Model {
    pub name: String,
    /// Number of match states: 194 (D0D2), 199 (D1D2), 294 (D0D1D2) for genes; 92-102 for domains.
    pub l: usize,
    /// `"gene"` or `"domain"`.
    pub kind: String,
    /// Canonical architecture of the gene, `None` for a domain model.
    pub architecture: Option<String>,
    /// Mean/std of this model's own training sequences, per match state. Backs `typicality`.
    pub per_match_state: Option<(f64, f64)>,
    /// Mean/std on raw totals -- the fallback `_typicality` uses for older manifests.
    pub training_stats: Option<(f64, f64)>,
    /// `20 x (L+1)`. Column 0 is unused and is all `-inf`; match states are 1..=L.
    match_emit_t: Vec<f64>,
    /// `20 x (L+1)`.
    insert_emit_t: Vec<f64>,
    /// `7 x (L+1)`, indexed by the `MM`..`DD` constants.
    trans_t: Vec<f64>,
}

impl Model {
    /// Match emissions for one residue across every model position: `me[j]`, contiguous.
    #[inline(always)]
    pub fn match_emit_row(&self, aa: usize) -> &[f64] {
        let n = self.l + 1;
        &self.match_emit_t[aa * n..(aa + 1) * n]
    }

    /// Insert emissions for one residue across every model position: `ie[j]`, contiguous.
    #[inline(always)]
    pub fn insert_emit_row(&self, aa: usize) -> &[f64] {
        let n = self.l + 1;
        &self.insert_emit_t[aa * n..(aa + 1) * n]
    }

    /// One transition class across every model position: `t[j]`, contiguous.
    #[inline(always)]
    pub fn trans_row(&self, t: usize) -> &[f64] {
        let n = self.l + 1;
        &self.trans_t[t * n..(t + 1) * n]
    }
}

/// Transpose a row-major `rows x cols` matrix into `cols x rows`.
fn transpose(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// The calibration block, as measured by `scripts/train_models.py` and stored in the manifest.
///
/// Every field is read, never defaulted to a plausible number: a threshold invented here would
/// be a guess presented as a calibration. The two `Option`s mirror `KIRClassifier.__init__`,
/// which does default them (`is_kir_threshold` and `domain_threshold` fall back to 0.0).
pub struct Calibration {
    /// `is_kir` tests log-odds per match state against this (invariant 2).
    pub per_residue_threshold: f64,
    /// Quarter of the KIR/decoy gap; the logistic scale for `confidence`.
    pub per_residue_width: f64,
    /// Log-odds separation below which a gene call is reported as a score tie.
    pub tie_tolerance: f64,
    /// Floor a domain placement must clear to enter the segmentation DP.
    pub domain_threshold: f64,
}

/// Every model in a directory, plus the shared background distribution and the manifest values
/// the classifier depends on.
pub struct ModelSet {
    /// All models, in `manifest["classes"]` order.
    pub models: Vec<Model>,
    /// Indices into `models` of the 15 gene models, in `manifest["gene_models"]` order.
    pub gene_idx: Vec<usize>,
    /// Indices into `models` of the 3 domain models, in `manifest["domain_models"]` order.
    pub domain_idx: Vec<usize>,
    /// Background log-frequencies, length 20. The null model.
    pub background: Vec<f64>,
    pub calibration: Calibration,
    /// sha256-prefix -> genes whose training set holds that exact Ig extent.
    pub exact_index: HashMap<String, Vec<String>>,
    /// The colliding ectodomains as `(sequence, genes)`, for containment testing.
    pub cross_gene: Vec<(String, Vec<String>)>,
}

impl ModelSet {
    /// Index of a model by name, or `None`.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.models.iter().position(|m| m.name == name)
    }

    /// The gene models, in manifest order.
    pub fn genes(&self) -> impl Iterator<Item = &Model> {
        self.gene_idx.iter().map(|&i| &self.models[i])
    }
}

// --- manifest deserialisation ------------------------------------------------

#[derive(Deserialize)]
struct RawStats {
    mean: Option<f64>,
    std: Option<f64>,
}

#[derive(Deserialize)]
struct RawModelEntry {
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    architecture: Option<String>,
    #[serde(default)]
    per_match_state_stats: Option<RawStats>,
    #[serde(default)]
    training_stats: Option<RawStats>,
}

#[derive(Deserialize)]
struct RawCalibration {
    #[serde(default)]
    per_residue_threshold: Option<f64>,
    #[serde(default)]
    per_residue_width: Option<f64>,
    #[serde(default)]
    tie_tolerance: Option<f64>,
    #[serde(default)]
    domain_threshold: Option<f64>,
}

#[derive(Deserialize)]
struct RawCrossGene {
    sequence: String,
    genes: Vec<String>,
}

#[derive(Deserialize)]
struct Manifest {
    classes: Vec<String>,
    #[serde(default)]
    gene_models: Vec<String>,
    #[serde(default)]
    domain_models: Vec<String>,
    #[serde(default)]
    models: HashMap<String, RawModelEntry>,
    #[serde(default)]
    calibration: Option<RawCalibration>,
    #[serde(default)]
    exact_index: HashMap<String, Vec<String>>,
    #[serde(default)]
    cross_gene_ectodomains: Vec<RawCrossGene>,
}

fn read_manifest(dir: &Path) -> Result<Manifest, String> {
    let path = dir.join("manifest.json");
    let text =
        std::fs::read_to_string(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("parsing {}: {e}", path.display()))
}

fn read_background(dir: &Path) -> Result<Vec<f64>, String> {
    let path = dir.join("background.npy");
    let buf = std::fs::read(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let (shape, data) = npy::parse(&buf)
        .and_then(|n| n.into_f64())
        .map_err(|e| format!("{}: {e}", path.display()))?;

    if shape != [ALPHABET_SIZE] {
        return Err(format!(
            "{}: expected shape [{ALPHABET_SIZE}], found {shape:?}",
            path.display()
        ));
    }
    Ok(data)
}

fn read_model(dir: &Path, name: &str, meta: Option<&RawModelEntry>) -> Result<Model, String> {
    let path = dir.join(format!("{name}.npz"));
    let file =
        std::fs::File::open(&path).map_err(|e| format!("opening {}: {e}", path.display()))?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| format!("{} is not a .npz: {e}", path.display()))?;

    let ctx = |e: String| format!("{}: {e}", path.display());

    let (len_shape, len_data) = npy::from_npz(&mut archive, "length")
        .and_then(|n| n.into_i64())
        .map_err(ctx)?;
    if len_data.len() != 1 {
        return Err(format!(
            "{}: 'length' should hold one value, found {len_shape:?}",
            path.display()
        ));
    }
    if len_data[0] < 1 {
        return Err(format!(
            "{}: model length {} is not positive",
            path.display(),
            len_data[0]
        ));
    }
    let l = len_data[0] as usize;

    let mut get = |member: &str, cols: usize| -> Result<Vec<f64>, String> {
        let (shape, data) = npy::from_npz(&mut archive, member)
            .and_then(|n| n.into_f64())
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if shape != [l + 1, cols] {
            return Err(format!(
                "{}: '{member}' has shape {shape:?}, expected [{}, {cols}] for a length-{l} model",
                path.display(),
                l + 1
            ));
        }
        Ok(data)
    };

    let match_emit = get("match_emit", ALPHABET_SIZE)?;
    let insert_emit = get("insert_emit", ALPHABET_SIZE)?;
    let trans = get("trans", N_TRANS)?;

    // A NaN anywhere would propagate through the max-plus recursion and poison every comparison
    // downstream, silently. -inf is expected (row 0 of match_emit is unused) and handled.
    for (member, arr) in [
        ("match_emit", &match_emit),
        ("insert_emit", &insert_emit),
        ("trans", &trans),
    ] {
        if arr.iter().any(|v| v.is_nan()) {
            return Err(format!("{}: '{member}' contains NaN", path.display()));
        }
    }

    let stats = |s: Option<&RawStats>| -> Option<(f64, f64)> {
        // The reference tests `if stats and stats.get("std")`, so a zero or absent std falls
        // through to the next branch rather than dividing by zero.
        s.and_then(|s| match (s.mean, s.std) {
            (Some(m), Some(sd)) if sd != 0.0 => Some((m, sd)),
            _ => None,
        })
    };

    Ok(Model {
        name: name.to_string(),
        l,
        kind: meta
            .and_then(|m| m.kind.clone())
            .unwrap_or_else(|| "gene".to_string()),
        architecture: meta.and_then(|m| m.architecture.clone()),
        per_match_state: stats(meta.and_then(|m| m.per_match_state_stats.as_ref())),
        training_stats: stats(meta.and_then(|m| m.training_stats.as_ref())),
        match_emit_t: transpose(&match_emit, l + 1, ALPHABET_SIZE),
        insert_emit_t: transpose(&insert_emit, l + 1, ALPHABET_SIZE),
        trans_t: transpose(&trans, l + 1, N_TRANS),
    })
}

fn load_uncached(dir: &Path) -> Result<ModelSet, String> {
    let manifest = read_manifest(dir)?;
    if manifest.classes.is_empty() {
        return Err(format!("{}: manifest lists no classes", dir.display()));
    }
    let background = read_background(dir)?;

    // rayon's `collect` into a Vec preserves input order, so manifest order survives.
    let models = manifest
        .classes
        .par_iter()
        .map(|name| read_model(dir, name, manifest.models.get(name)))
        .collect::<Result<Vec<_>, _>>()?;

    let position = |name: &str| -> Result<usize, String> {
        models.iter().position(|m| m.name == name).ok_or_else(|| {
            format!(
                "{}: manifest names model '{name}' but it is not in 'classes'",
                dir.display()
            )
        })
    };

    // A manifest without the explicit split is a `histo_hmm`-style one; fall back to `kind`,
    // which `read_model` defaults to "gene" exactly as `save_models` does.
    let gene_names: Vec<String> = if manifest.gene_models.is_empty() {
        models
            .iter()
            .filter(|m| m.kind == "gene")
            .map(|m| m.name.clone())
            .collect()
    } else {
        manifest.gene_models.clone()
    };
    let domain_names: Vec<String> = if manifest.domain_models.is_empty() {
        models
            .iter()
            .filter(|m| m.kind == "domain")
            .map(|m| m.name.clone())
            .collect()
    } else {
        manifest.domain_models.clone()
    };

    if gene_names.is_empty() {
        return Err(format!("{}: manifest lists no gene models", dir.display()));
    }

    let gene_idx = gene_names
        .iter()
        .map(|n| position(n))
        .collect::<Result<Vec<_>, _>>()?;
    let domain_idx = domain_names
        .iter()
        .map(|n| position(n))
        .collect::<Result<Vec<_>, _>>()?;

    let raw_cal = manifest.calibration;
    let cal = Calibration {
        // `KIRClassifier.__init__` indexes `cal["per_residue_threshold"]` directly -- a manifest
        // without it is a training failure, not a case to paper over with a default.
        per_residue_threshold: raw_cal
            .as_ref()
            .and_then(|c| c.per_residue_threshold)
            .ok_or_else(|| {
                format!(
                    "{}: manifest has no calibration.per_residue_threshold; the models were not \
                     calibrated and is_kir would be a guess",
                    dir.display()
                )
            })?,
        // The reference floors this at 1e-9 to keep the logistic well defined.
        per_residue_width: raw_cal
            .as_ref()
            .and_then(|c| c.per_residue_width)
            .unwrap_or(0.1)
            .max(1e-9),
        tie_tolerance: raw_cal
            .as_ref()
            .and_then(|c| c.tie_tolerance)
            .unwrap_or(1.0),
        domain_threshold: raw_cal
            .as_ref()
            .and_then(|c| c.domain_threshold)
            .unwrap_or(0.0),
    };

    Ok(ModelSet {
        models,
        gene_idx,
        domain_idx,
        background,
        calibration: cal,
        exact_index: manifest.exact_index,
        cross_gene: manifest
            .cross_gene_ectodomains
            .into_iter()
            .map(|e| (e.sequence, e.genes))
            .collect(),
    })
}

type Cache = Mutex<HashMap<PathBuf, Arc<ModelSet>>>;

fn cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Load a model directory, or return the already-loaded copy.
///
/// Keyed on the canonical path, so `models/` and `models/../models/` share one entry.
pub fn load(dir: &Path) -> Result<Arc<ModelSet>, String> {
    let key = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());

    if let Some(hit) = cache()
        .lock()
        .expect("model cache mutex poisoned")
        .get(&key)
    {
        return Ok(Arc::clone(hit));
    }

    // Loading outside the lock: two threads racing the same cold directory duplicate the work
    // once, which is cheaper than making every reader wait behind whoever got there first.
    let set = Arc::new(load_uncached(dir)?);

    let mut guard = cache().lock().expect("model cache mutex poisoned");
    Ok(Arc::clone(guard.entry(key).or_insert(set)))
}
