//! A port of `histo_kir_hmm`'s `classify.py` -- `KIRClassifier`, `ChainType` and `DomainSegment`.
//!
//! Every constant, comparison and tie-break here is chosen to match the reference rather than to
//! be defensible on its own terms: the contract is "same outputs", so where `histo_kir_hmm` is
//! quirky, we are quirky. Three of its invariants are reproduced deliberately and are the things
//! a casual port gets wrong:
//!
//! * **`is_kir` tests log-odds per MATCH STATE** with denominator `min(n_scored, L_gene)`
//!   (invariant 2). Not the total -- that is a length threshold in disguise, and it rejects
//!   `8tui` chain A, a real 98-residue KIR2DL3 D1 domain. Not per residue either -- that dilutes
//!   a precursor's score with the ~150 residues outside the ectodomain and rejected 248 of 433
//!   KIR2DL1 alleles.
//! * **Ties are reported, never broken** (invariant 3). `gene` is `None` with `genes` holding the
//!   set. 127 IPD-KIR alleles have an ectodomain byte-identical to another gene's, including the
//!   reference allele `KIR2DS2*0010101`, which appears in four deposited structures. There is no
//!   rule that could pick correctly -- the information is not in the sequence.
//! * **Exact-tie detection tests containment, not just hash equality** (invariant 4). A real
//!   query is a precursor or a structural chain and is never byte-equal to a 199-residue training
//!   extent, so a hash-only test never fires.
//!
//! The emissions invariant (fixed effective count, `n_effective=20`) is baked into the trained
//! `.npz` arrays, which we read byte-identical; there is nothing to re-derive here, and
//! `codegen/vendor_models.py` asserts the models came from a manifest that records it.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use rayon::prelude::*;
use sha2::{Digest, Sha256};

use crate::hmm::{encode, encodes, log_odds_pairwise, log_odds_sequential, viterbi, Scratch};
use crate::models::ModelSet;

// --- Scan geometry (classify.py) ---------------------------------------------
const MIN_WINDOW: usize = 90;
const MAX_WINDOW: usize = 300;
const WINDOW_STEP: usize = 15;
const SCAN_STRIDE: usize = 15;
/// `_MAX_WINDOW + 60`. Only queries longer than this are windowed; every deposited structural
/// chain, including the 98-residue single-domain one, is scored whole.
const SCAN_TRIGGER: usize = MAX_WINDOW + 60;

// --- Domain segmentation geometry -------------------------------------------
const DOMAIN_LENGTHS: [usize; 6] = [86, 90, 94, 98, 102, 106];
const DOMAIN_STRIDE: usize = 2;

/// `_SOFTMAX_TEMPERATURE`. 8.0 here, against `histo_hmm`'s 30.0 -- KIR log-odds separations are
/// much smaller than MHC locus separations.
const SOFTMAX_TEMPERATURE: f64 = 8.0;

/// Minimum scorable residues for a call at all (`type_chain`'s floor).
const MIN_SCORABLE: usize = 40;
/// Minimum scorable residues for domain segmentation (`call_architecture`'s floor).
const MIN_FOR_ARCHITECTURE: usize = 60;

/// Genes with zero deposited structures anywhere in the PDB (`_NO_STRUCTURE_GENES`). Their loop
/// windows are alignment predictions, so a positive call on one carries a note saying so.
const NO_STRUCTURE_GENES: [&str; 8] = [
    "KIR2DL5A", "KIR2DL5B", "KIR2DS1", "KIR2DS3", "KIR2DS5", "KIR3DL2", "KIR3DL3", "KIR3DS1",
];

/// One Ig domain located in a query sequence. Mirrors `DomainSegment`.
pub struct DomainSegment {
    pub domain: String,
    pub start: usize,
    pub end: usize,
    pub score: f64,
}

/// The Rust mirror of `ChainType`. Field names and semantics are the contract; see the
/// `histo_kir_hmm.classify` module docstring.
pub struct ChainType {
    pub chain_type: &'static str,
    pub is_kir: bool,
    pub gene: Option<String>,
    pub genes: Vec<String>,
    pub is_tie: bool,
    pub tie_kind: &'static str,
    pub architecture: Option<String>,
    pub domains: Vec<DomainSegment>,
    pub confidence: f64,
    pub gene_confidence: f64,
    pub typicality: f64,
    pub score: f64,
    pub scored_region: (usize, usize),
    pub notes: Vec<String>,
    /// Log-odds against every gene model, in manifest order (paired with `set.genes()`).
    pub raw_scores: Vec<f64>,
}

/// `_clean_sequence`: uppercase, then keep only the 20 amino acids **and `X`**.
///
/// Everything else -- gaps, `*`, and the ambiguity codes `B`/`Z`/`J` -- is *dropped*, which
/// shortens the sequence. That is why `scored_region` and the domain indices index the cleaned
/// sequence, not the caller's input string. `X` survives to encode as `-1` and score with a
/// uniform emission.
pub fn clean_sequence(seq: &str) -> Vec<u8> {
    if seq.is_ascii() {
        return seq
            .bytes()
            .map(|b| b.to_ascii_uppercase())
            .filter(|b| is_kept(*b))
            .collect();
    }
    // The reference uppercases with Python's `str.upper()`, which is Unicode-aware and can turn
    // one character into several ('ß' -> "SS"). An ASCII-only uppercase would drop those instead
    // and silently score a shorter sequence, so non-ASCII input takes the Unicode path.
    seq.to_uppercase().bytes().filter(|b| is_kept(*b)).collect()
}

/// `c in AA_TO_IDX or c == "X"`.
#[inline]
fn is_kept(b: u8) -> bool {
    b == b'X' || encodes(b)
}

/// `_sequence_hash`: the first 16 hex characters of the sha256 of the sequence.
fn sequence_hash(seq: &[u8]) -> String {
    let digest = Sha256::digest(seq);
    let mut out = String::with_capacity(16);
    for byte in &digest[..8] {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// `max(values)` with numpy's NaN-propagating comparison semantics, and the index of the **first**
/// maximum -- which is what `np.argmax` and `max(dict, key=dict.get)` both return.
fn arg_max(values: &[f64]) -> (usize, f64) {
    let mut best_i = 0usize;
    let mut best = f64::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if i == 0 || v > best {
            best_i = i;
            best = v;
        }
    }
    (best_i, best)
}

/// numpy's pairwise sum, for the 15-element softmax denominator and `np.mean`. Small-n only,
/// which is the `n < 8` / `n <= 128` branch of `pairwise_sum_DOUBLE`.
fn pairwise_sum_small(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in a {
            res += v;
        }
        return res;
    }
    let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
    let end = n - (n % 8);
    let mut i = 8;
    while i < end {
        for k in 0..8 {
            r[k] += a[i + k];
        }
        i += 8;
    }
    let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    while i < n {
        res += a[i];
        i += 1;
    }
    res
}

/// `scipy.stats.norm.cdf`, to within ~1e-15 relative.
///
/// `Phi(z) = erfc(-z / sqrt(2)) / 2`, with `erfc` computed as the upper incomplete gamma
/// `Q(1/2, x^2)` -- series below the crossover, modified Lentz continued fraction above, both
/// iterated to relative 1e-16. This is the one value in the port that is not bit-exact against
/// the reference: scipy's `ndtr` is Cephes' rational-polynomial approximation, and matching it
/// bit-for-bit would mean vendoring those coefficients. It backs `typicality` only, which no
/// decision reads -- `is_kir`, the gene call, the tie set and the architecture are all decided
/// before it is computed. The measured agreement is reported in the parity report.
fn norm_cdf(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    // erfc saturates well inside these bounds; short-circuit before the series is asked to
    // converge on a number that is already 0 or 1 in double precision.
    if z > 40.0 {
        return 1.0;
    }
    if z < -40.0 {
        return 0.0;
    }
    let x = -z / std::f64::consts::SQRT_2;
    0.5 * erfc(x)
}

/// `erfc(x)` for any finite `x`, via the incomplete gamma function.
fn erfc(x: f64) -> f64 {
    let ax = x.abs();
    // Q(1/2, ax^2) == erfc(ax).
    let a = 0.5;
    let xx = ax * ax;
    let q = if xx < a + 1.0 {
        // Series for the *lower* incomplete gamma, then complement.
        1.0 - gamma_p_series(a, xx)
    } else {
        gamma_q_cf(a, xx)
    };
    if x >= 0.0 {
        q
    } else {
        2.0 - q
    }
}

const GAMMA_EPS: f64 = 3e-16;
const GAMMA_ITMAX: usize = 400;
/// `ln Gamma(1/2) = ln sqrt(pi)`. The only value of `ln Gamma` the series needs.
const LN_GAMMA_HALF: f64 = 0.5723649429247001;

/// Regularised lower incomplete gamma `P(a, x)` by its series expansion.
fn gamma_p_series(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..GAMMA_ITMAX {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * GAMMA_EPS {
            break;
        }
    }
    sum * (-x + a * x.ln() - LN_GAMMA_HALF).exp()
}

/// Regularised upper incomplete gamma `Q(a, x)` by the modified Lentz continued fraction.
fn gamma_q_cf(a: f64, x: f64) -> f64 {
    const FPMIN: f64 = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=GAMMA_ITMAX {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < GAMMA_EPS {
            break;
        }
    }
    (-x + a * x.ln() - LN_GAMMA_HALF).exp() * h
}

/// Log-odds of one cleaned+encoded sequence against every gene model, in manifest order.
///
/// `_score_gene_models` calls `log_odds_score`, so the null is the pairwise one.
fn score_gene_models(encoded: &[i8], set: &ModelSet) -> Vec<f64> {
    set.gene_idx
        .par_iter()
        .map_init(Scratch::new, |scratch, &gi| {
            log_odds_pairwise(encoded, &set.models[gi], &set.background, scratch)
        })
        .collect()
}

/// The best window of a construct scan: `(scores, start, end)`.
///
/// `_scan_for_region`, structurally intact. The reference iterates window sizes outer and, within
/// a size, takes `np.argmax` of the per-window column maxima -- the **first** maximum -- then
/// keeps the size's result only if it beats the running best with a strict `>`. So the earliest
/// size and, within it, the earliest start wins a tie. The parallel reduction below breaks ties
/// on the lower window index, which makes it associative and independent of rayon's scheduling.
///
/// Window scores use `log_odds_score_batch`, whose null is accumulated sequentially -- not the
/// pairwise null of the whole-sequence path. Mixing the two would shift a score in its last bits
/// and could flip which window wins.
fn scan_for_region(encoded: &[i8], set: &ModelSet) -> Option<(Vec<f64>, usize, usize)> {
    let n = encoded.len();
    let mut best_score = f64::NEG_INFINITY;
    let mut best: Option<(Vec<f64>, usize, usize)> = None;

    let upper = MAX_WINDOW.min(n);
    let mut size = MIN_WINDOW;
    while size <= upper {
        let starts: Vec<usize> = (0..=n - size).step_by(SCAN_STRIDE).collect();
        if starts.is_empty() {
            size += WINDOW_STEP;
            continue;
        }

        // Per window: all 15 gene scores, and the column maximum.
        let per_window: Vec<(Vec<f64>, f64)> = starts
            .par_iter()
            .map(|&s| {
                let window = &encoded[s..s + size];
                let scores: Vec<f64> = set
                    .gene_idx
                    .iter()
                    .map(|&gi| {
                        // A fresh Scratch per window would allocate; one per rayon task is enough
                        // because the 15 models are scored sequentially inside the task.
                        let mut scratch = Scratch::new();
                        log_odds_sequential(window, &set.models[gi], &set.background, &mut scratch)
                    })
                    .collect();
                let (_, col_max) = arg_max(&scores);
                (scores, col_max)
            })
            .collect();

        let (idx, col_max) = arg_max(&per_window.iter().map(|(_, m)| *m).collect::<Vec<_>>());
        if col_max > best_score {
            best_score = col_max;
            best = Some((per_window[idx].0.clone(), starts[idx], starts[idx] + size));
        }
        size += WINDOW_STEP;
    }
    best
}

/// `_gene_scores`: whole-sequence scores, or a scan for a long query.
fn gene_scores(encoded: &[i8], set: &ModelSet) -> (Vec<f64>, usize, usize) {
    let n = encoded.len();
    if n <= SCAN_TRIGGER {
        return (score_gene_models(encoded, set), 0, n);
    }
    match scan_for_region(encoded, set) {
        Some(v) => v,
        // `if not best` in the reference: unreachable here, because the branch guarantees
        // n > 360 >= MIN_WINDOW, so at least one window exists.
        None => (score_gene_models(encoded, set), 0, n),
    }
}

/// `_probabilities`: temperature-scaled softmax over the gene log-odds.
fn probabilities(scores: &[f64]) -> Vec<f64> {
    let scaled: Vec<f64> = scores.iter().map(|s| s / SOFTMAX_TEMPERATURE).collect();
    let (_, max) = arg_max(&scaled);
    let exps: Vec<f64> = scaled.iter().map(|s| (s - max).exp()).collect();
    let total = pairwise_sum_small(&exps);
    exps.iter().map(|e| e / total).collect()
}

/// `_per_match_state`: `score / max(min(n_scored, L_gene), 1)` -- invariant 2's denominator.
fn per_match_state(score: f64, n_scored: usize, model_length: usize) -> f64 {
    score / n_scored.min(model_length).max(1) as f64
}

/// `_kir_confidence`: logistic on the per-match-state scale, 0.5 at the calibrated threshold.
fn kir_confidence(per_state: f64, threshold: f64, width: f64) -> f64 {
    let z = ((per_state - threshold) / width).clamp(-50.0, 50.0);
    1.0 / (1.0 + (-z).exp())
}

/// `_typicality`: percentile of the score under the winning model's own training distribution,
/// measured per match state. Saturates on structural input by design (invariant 12) -- do not
/// threshold on it.
fn typicality(set: &ModelSet, model_idx: usize, score: f64, n_scored: usize) -> f64 {
    let model = &set.models[model_idx];
    if let Some((mean, std)) = model.per_match_state {
        let observed = per_match_state(score, n_scored, model.l);
        let z = (observed - mean) / std;
        return norm_cdf(z).powf(0.7).clamp(0.0, 1.0);
    }
    // Older manifests without per-match-state stats fall back to raw totals.
    if let Some((mean, std)) = model.training_stats {
        let z = (score - mean) / std;
        return norm_cdf(z).powf(0.7).clamp(0.0, 1.0);
    }
    0.5
}

/// The outcome of `assign_gene`.
struct Assignment {
    gene: Option<String>,
    genes: Vec<String>,
    is_tie: bool,
    tie_kind: &'static str,
    score: f64,
    gene_confidence: f64,
    typicality: f64,
    scored_region: (usize, usize),
    raw_scores: Vec<f64>,
    notes: Vec<String>,
    /// Index into `set.models` of the top-scoring gene model.
    top_idx: usize,
}

/// `assign_gene`: gene call with both tie mechanisms, reporting ties honestly.
fn assign_gene(cleaned: &[u8], set: &ModelSet) -> Assignment {
    let encoded = encode(cleaned);
    let (scores, start, end) = gene_scores(&encoded, set);
    let region = &cleaned[start..end];

    // `sorted(..., key=value, reverse=True)` is stable in Python, and `reverse=True` does not
    // reverse equal elements: ties keep manifest order. Rust's `sort_by` is stable too.
    let mut ordered: Vec<(usize, f64)> = scores.iter().copied().enumerate().collect();
    ordered.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let (top_pos, top_score) = ordered[0];
    let top_gene = set.models[set.gene_idx[top_pos]].name.clone();
    let margin = if ordered.len() > 1 {
        top_score - ordered[1].1
    } else {
        f64::INFINITY
    };

    let mut notes: Vec<String> = Vec::new();

    // Exact tie: the scored region is byte-identical to training extents of more than one gene.
    let mut exact_hit: Vec<String> = set
        .exact_index
        .get(&sequence_hash(region))
        .cloned()
        .unwrap_or_default();
    if exact_hit.len() <= 1 {
        // Invariant 4: a hash test alone never fires on a real query, so also test containment in
        // both directions. Either direction is an irreducible tie -- containment means the query
        // carries the whole ambiguous ectodomain, being contained means it is a fragment of it.
        let region_str = String::from_utf8_lossy(region);
        for (tied_seq, tied_genes) in &set.cross_gene {
            if region_str.contains(tied_seq.as_str())
                || (region.len() >= 60 && tied_seq.contains(region_str.as_ref()))
            {
                exact_hit = tied_genes.clone();
                break;
            }
        }
    }

    let genes: Vec<String>;
    let tie_kind: &'static str;
    if exact_hit.len() > 1 {
        genes = sorted_unique(exact_hit);
        tie_kind = "exact";
        notes.push(format!(
            "Scored region is byte-identical to training ectodomains of {}; these genes cannot \
             be separated on ectodomain sequence alone.",
            genes.join(" and ")
        ));
    } else {
        let tied: Vec<usize> = ordered
            .iter()
            .filter(|(_, s)| top_score - s <= set.calibration.tie_tolerance)
            .map(|(i, _)| *i)
            .collect();
        if tied.len() > 1 {
            genes = sorted_unique(
                tied.iter()
                    .map(|&i| set.models[set.gene_idx[i]].name.clone())
                    .collect(),
            );
            tie_kind = "score";
            notes.push(format!(
                "Log-odds separation {:.2} is within the calibrated tie tolerance {:.2}; \
                 reported as a tie.",
                margin, set.calibration.tie_tolerance
            ));
        } else {
            genes = vec![top_gene.clone()];
            tie_kind = "none";
        }
    }

    // `sum(probs[g] for g in genes)` -- Python's builtin sum, left to right in `genes` order.
    let probs = probabilities(&scores);
    let mut gene_confidence = 0.0;
    for g in &genes {
        if let Some(pos) = set
            .gene_idx
            .iter()
            .position(|&gi| set.models[gi].name == *g)
        {
            gene_confidence += probs[pos];
        }
    }

    // The typicality model is the single called gene, or the top scorer when tied.
    let typ_name = if genes.len() == 1 {
        &genes[0]
    } else {
        &top_gene
    };
    let typ_idx = set.index_of(typ_name).unwrap_or(set.gene_idx[top_pos]);
    let n_scored = (end - start).max(1);
    let typ = typicality(set, typ_idx, top_score, n_scored);

    let no_struct: Vec<&str> = NO_STRUCTURE_GENES
        .iter()
        .copied()
        .filter(|g| genes.iter().any(|x| x == g))
        .collect();
    if !no_struct.is_empty() {
        // The reference iterates `genes` (sorted) here; NO_STRUCTURE_GENES is itself sorted, so
        // filtering either way gives the same order.
        notes.push(format!(
            "No deposited structure exists for {}; loop windows for these genes are alignment \
             predictions.",
            no_struct.join(", ")
        ));
    }

    Assignment {
        gene: if genes.len() == 1 {
            Some(genes[0].clone())
        } else {
            None
        },
        is_tie: genes.len() > 1,
        genes,
        tie_kind,
        score: top_score,
        gene_confidence,
        typicality: typ,
        scored_region: (start, end),
        raw_scores: scores,
        notes,
        top_idx: set.gene_idx[top_pos],
    }
}

/// `sorted(set_of_genes)` -- deduplicated and lexicographically ordered, as Python's `sorted`
/// over a set produces.
fn sorted_unique(v: Vec<String>) -> Vec<String> {
    v.into_iter().collect::<BTreeSet<_>>().into_iter().collect()
}

/// The outcome of `call_architecture`.
struct Architecture {
    architecture: Option<String>,
    domains: Vec<DomainSegment>,
}

/// `call_architecture`: segment into Ig domains and name the architecture.
///
/// Uses only the D0/D1/D2 models, so the result is independent of the gene call and the two can be
/// cross-checked (invariant 10). It reports what it observes, not what the gene implies
/// (invariant 11): a 167-residue KIR2DL3 chain segments as `"D1"`, and that is correct.
///
/// Placement scores come from `log_odds_score_batch` (sequential null); the rival scores in the
/// confidence calculation come from `log_odds_score` (pairwise null). That asymmetry is the
/// reference's and is reproduced rather than tidied.
fn call_architecture(cleaned: &[u8], set: &ModelSet) -> Architecture {
    let n = cleaned.len();
    if n < MIN_FOR_ARCHITECTURE {
        return Architecture {
            architecture: None,
            domains: Vec::new(),
        };
    }
    let encoded = encode(cleaned);

    // Placements keyed by end position, in the reference's insertion order: domain-major, then by
    // (window length, start). The DP's strict `>` makes that order the tie-break.
    let mut placements: Vec<Vec<(usize, usize, f64)>> = vec![Vec::new(); n + 1];

    for (d_pos, &di) in set.domain_idx.iter().enumerate() {
        let model = &set.models[di];
        let mut windows: Vec<(usize, usize)> = Vec::new();
        for &length in DOMAIN_LENGTHS.iter() {
            if length > n {
                continue;
            }
            windows.extend(
                (0..=n - length)
                    .step_by(DOMAIN_STRIDE)
                    .map(|s| (s, s + length)),
            );
        }
        if windows.is_empty() {
            continue;
        }

        let scores: Vec<f64> = windows
            .par_iter()
            .map_init(Scratch::new, |scratch, &(s, e)| {
                log_odds_sequential(&encoded[s..e], model, &set.background, scratch)
            })
            .collect();

        for (&(start, end), &score) in windows.iter().zip(scores.iter()) {
            if score >= set.calibration.domain_threshold {
                placements[end].push((start, d_pos, score));
            }
        }
    }

    // DP over end position: best total score using non-overlapping domains.
    let mut best = vec![0.0f64; n + 1];
    let mut back: Vec<Option<(usize, usize, f64)>> = vec![None; n + 1];
    for i in 1..=n {
        best[i] = best[i - 1];
        for &(start, d_pos, score) in &placements[i] {
            let candidate = best[start] + score;
            if candidate > best[i] {
                best[i] = candidate;
                back[i] = Some((start, d_pos, score));
            }
        }
    }

    let mut segments: Vec<DomainSegment> = Vec::new();
    let mut i = n;
    while i > 0 {
        match back[i] {
            None => i -= 1,
            Some((start, d_pos, score)) => {
                segments.push(DomainSegment {
                    domain: set.models[set.domain_idx[d_pos]].name.clone(),
                    start,
                    end: i,
                    score,
                });
                i = start;
            }
        }
    }
    segments.reverse();

    if segments.is_empty() {
        return Architecture {
            architecture: None,
            domains: Vec::new(),
        };
    }

    let architecture: String = segments.iter().map(|s| s.domain.as_str()).collect();
    Architecture {
        architecture: Some(architecture),
        domains: segments,
    }
}

/// `call_architecture`'s confidence: for each placed domain, how decisively its own model beat the
/// other domain models on the same span, averaged. Computed separately because `type_chain` never
/// reads it -- it keeps `kir_confidence` -- so a caller asking only for `classify` does not pay
/// for `2 * n_domains` extra Viterbi passes.
fn architecture_confidence(cleaned: &[u8], set: &ModelSet, arch: &Architecture) -> f64 {
    if arch.domains.is_empty() {
        return 0.0;
    }
    let encoded = encode(cleaned);
    let per_domain: Vec<f64> = arch
        .domains
        .iter()
        .map(|seg| {
            let mut scratch = Scratch::new();
            let rivals: Vec<f64> = set
                .domain_idx
                .iter()
                .filter(|&&di| set.models[di].name != seg.domain)
                .map(|&di| {
                    log_odds_pairwise(
                        &encoded[seg.start..seg.end],
                        &set.models[di],
                        &set.background,
                        &mut scratch,
                    )
                })
                .collect();
            let gap = if rivals.is_empty() {
                seg.score
            } else {
                seg.score - arg_max(&rivals).1
            };
            1.0 / (1.0 + (-(gap / 5.0).clamp(-50.0, 50.0)).exp())
        })
        .collect();
    // np.mean: pairwise sum divided by n.
    pairwise_sum_small(&per_domain) / per_domain.len() as f64
}

/// `type_chain`. See the `histo_kir_hmm.classify` module docstring for the contract.
pub fn type_chain(sequence: &str, set: &ModelSet, with_architecture: bool) -> ChainType {
    let cleaned = clean_sequence(sequence);

    if cleaned.len() < MIN_SCORABLE {
        return ChainType {
            chain_type: "non_kir",
            is_kir: false,
            gene: None,
            genes: Vec::new(),
            is_tie: false,
            tie_kind: "none",
            architecture: None,
            domains: Vec::new(),
            confidence: 1.0,
            gene_confidence: 0.0,
            typicality: 0.0,
            score: f64::NEG_INFINITY,
            scored_region: (0, cleaned.len()),
            notes: vec![format!(
                "Sequence has {} scorable residues; below the {MIN_SCORABLE}-residue floor for a \
                 meaningful profile-HMM call. Rejected without scoring.",
                cleaned.len()
            )],
            raw_scores: Vec::new(),
        };
    }

    let assignment = assign_gene(&cleaned, set);
    let (start, end) = assignment.scored_region;
    let n_scored = (end - start).max(1);
    let top_model = &set.models[assignment.top_idx];
    let per_state = per_match_state(assignment.score, n_scored, top_model.l);
    let positive = per_state >= set.calibration.per_residue_threshold;
    let conf = kir_confidence(
        per_state,
        set.calibration.per_residue_threshold,
        set.calibration.per_residue_width,
    );

    if !positive {
        return ChainType {
            chain_type: "non_kir",
            is_kir: false,
            gene: None,
            genes: Vec::new(),
            is_tie: false,
            tie_kind: "none",
            architecture: None,
            domains: Vec::new(),
            confidence: 1.0 - conf,
            gene_confidence: 0.0,
            typicality: 0.0,
            score: assignment.score,
            scored_region: (start, end),
            notes: vec![format!(
                "Best gene-model log-odds {:.1} is {:.3} per match state, below the calibrated \
                 is_kir threshold {:.3} (nearest model {}, {} residues scored).",
                assignment.score,
                per_state,
                set.calibration.per_residue_threshold,
                top_model.name,
                n_scored
            )],
            raw_scores: assignment.raw_scores,
        };
    }

    let mut notes = assignment.notes;
    let (architecture, domains) = if with_architecture {
        let arch = call_architecture(&cleaned[start..end], set);

        // The gene call implies an architecture; a disagreement is a note, not an error, and
        // usually means an incomplete chain (invariants 10 and 11).
        let expected: BTreeSet<String> = assignment
            .genes
            .iter()
            .filter_map(|g| {
                set.index_of(g)
                    .and_then(|i| set.models[i].architecture.clone())
            })
            .collect();
        if let Some(observed) = arch.architecture.as_deref() {
            if !expected.is_empty() && !expected.contains(observed) {
                notes.push(format!(
                    "Domain segmentation gives {} but the gene call ({}) implies {}; the chain is \
                     probably incomplete or a single-domain construct.",
                    observed,
                    assignment.genes.join("/"),
                    expected.iter().cloned().collect::<Vec<_>>().join("/")
                ));
            }
        }

        // Shift domain indices back into the caller's (cleaned) coordinate frame.
        let domains = arch
            .domains
            .iter()
            .map(|d| DomainSegment {
                domain: d.domain.clone(),
                start: d.start + start,
                end: d.end + start,
                score: d.score,
            })
            .collect();
        (arch.architecture, domains)
    } else {
        (None, Vec::new())
    };

    ChainType {
        chain_type: "kir",
        is_kir: true,
        gene: assignment.gene,
        genes: assignment.genes,
        is_tie: assignment.is_tie,
        tie_kind: assignment.tie_kind,
        architecture,
        domains,
        confidence: conf,
        gene_confidence: assignment.gene_confidence,
        typicality: assignment.typicality,
        score: assignment.score,
        scored_region: (start, end),
        notes,
        raw_scores: assignment.raw_scores,
    }
}

/// The `domains` expression: `call_architecture` on its own, including its confidence.
pub struct DomainCall {
    pub architecture: Option<String>,
    pub domains: Vec<DomainSegment>,
    pub confidence: f64,
}

/// `call_architecture` as a standalone call, on the whole cleaned sequence -- the reference's
/// public method, which does not scan for a region first.
pub fn domains_of(sequence: &str, set: &ModelSet) -> DomainCall {
    let cleaned = clean_sequence(sequence);
    let arch = call_architecture(&cleaned, set);
    let confidence = architecture_confidence(&cleaned, set, &arch);
    DomainCall {
        architecture: arch.architecture,
        domains: arch.domains,
        confidence,
    }
}

/// `ProfileHMM.log_odds_score` against one named model -- the `score` expression.
///
/// The sequence is cleaned exactly as `type_chain` cleans it, so the two agree on what a residue
/// is. There is no scan and no windowing: the whole cleaned sequence is scored against the one
/// model, which is what the reference's `log_odds_score` does.
pub fn score_one(sequence: &str, set: &ModelSet, model_idx: usize, scratch: &mut Scratch) -> f64 {
    let cleaned = clean_sequence(sequence);
    let encoded = encode(&cleaned);
    log_odds_pairwise(&encoded, &set.models[model_idx], &set.background, scratch)
}

/// `ProfileHMM.viterbi` against one named model, without the null subtraction.
pub fn viterbi_one(sequence: &str, set: &ModelSet, model_idx: usize, scratch: &mut Scratch) -> f64 {
    let cleaned = clean_sequence(sequence);
    let encoded = encode(&cleaned);
    viterbi(&encoded, &set.models[model_idx], scratch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_sequence_drops_gaps_and_keeps_x() {
        assert_eq!(clean_sequence("ac-d*xB?"), b"ACDX".to_vec());
    }

    #[test]
    fn arg_max_returns_first_maximum() {
        assert_eq!(arg_max(&[1.0, 3.0, 3.0, 2.0]), (1, 3.0));
    }

    #[test]
    fn sequence_hash_is_sixteen_hex_chars() {
        let h = sequence_hash(b"ACDEFGHIKL");
        assert_eq!(h.len(), 16);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn norm_cdf_matches_known_values() {
        // Reference values from scipy.stats.norm.cdf.
        for (z, expect) in [
            (0.0, 0.5),
            (1.0, 0.8413447460685429),
            (-1.0, 0.15865525393145707),
            (1.959963984540054, 0.975),
            (-3.0, 0.0013498980316300933),
            (6.0, 0.9999999990134123),
        ] {
            let got = norm_cdf(z);
            assert!(
                (got - expect).abs() <= 1e-14 * expect.abs().max(1e-300),
                "norm_cdf({z}) = {got}, expected {expect}"
            );
        }
    }

    #[test]
    fn norm_cdf_far_tail_underflows_like_scipy() {
        assert_eq!(norm_cdf(-100.0), 0.0);
        assert_eq!(norm_cdf(100.0), 1.0);
    }
}
