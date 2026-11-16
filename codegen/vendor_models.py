"""Vendor histo_kir_hmm's trained profile HMMs into `python/polars_kir_hmm/models/`.

The `.npz` files are copied **byte-identical** from a histo_kir_hmm checkout: this plugin reads
exactly the format histo_kir_hmm writes (its invariant 13 -- arrays in npz, scalars in JSON, no
pickle anywhere), so re-syncing retrained models is a copy rather than a conversion, and there is
no chance of a transcoding bug silently changing a model.

`manifest.json` is copied whole rather than rewritten, because the plugin needs almost all of it:

  calibration                per_residue_threshold, per_residue_width, tie_tolerance,
                             domain_threshold -- measured numbers, not tuning knobs
  gene_models/domain_models  the iteration orders that decide Python's stable-sort tie-breaks
  exact_index                sha256-prefix -> genes, the irreducible ectodomain collisions
  cross_gene_ectodomains     the colliding sequences, for containment testing (invariant 4)
  models[*].per_match_state_stats  the mean/std that typicality is a percentile under
  models[*].architecture     what the gene call implies, for the segmentation cross-check

Only a `vendored_from` block is added, recording where the copy came from. Re-running against the
same checkout is byte-for-byte reproducible apart from that block's `checksums`, which are content
hashes rather than timestamps.

What this script *checks* is that every model is shaped the way the Rust kernel assumes, and that
the manifest still carries the four calibration values and the training provenance. A missing
calibration value is a hard failure: the plugin would then have to invent a threshold, which is a
guess presented as a calibration.

Run:  python codegen/vendor_models.py --histo-kir-hmm ../histo_kir_hmm
      python codegen/vendor_models.py            # resolves the installed histo_kir_hmm package
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "python" / "polars_kir_hmm" / "models"

# The alphabet histo_kir_hmm trains against (histo_kir_hmm/alphabet.py). Vendored models must
# match, otherwise the plugin's hardcoded 20-letter encoding would silently misread them.
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
ALPHABET_SIZE = len(ALPHABET)
N_TRANS = 7

# Calibration values the plugin reads. `per_residue_threshold` is the one it cannot do without.
REQUIRED_CALIBRATION = (
    "per_residue_threshold",
    "per_residue_width",
    "tie_tolerance",
    "domain_threshold",
)


def _validate_model(npz_path: Path) -> int:
    """Check one model archive is shaped the way the Rust kernel assumes. Returns L."""
    data = np.load(npz_path)

    missing = {"match_emit", "insert_emit", "trans", "length"} - set(data.files)
    if missing:
        raise SystemExit(f"{npz_path.name}: missing array(s) {sorted(missing)}")

    length = int(data["length"][0])
    expected = {
        "match_emit": (length + 1, ALPHABET_SIZE),
        "insert_emit": (length + 1, ALPHABET_SIZE),
        "trans": (length + 1, N_TRANS),
    }
    for name, shape in expected.items():
        arr = data[name]
        if arr.shape != shape:
            raise SystemExit(f"{npz_path.name}: {name} has shape {arr.shape}, expected {shape}")
        if arr.dtype != np.float64:
            raise SystemExit(f"{npz_path.name}: {name} has dtype {arr.dtype}, expected float64")
        if np.isnan(arr).any():
            raise SystemExit(f"{npz_path.name}: {name} contains NaN")
        # -inf is expected (row 0 of match_emit is unused), +inf is not.
        if np.isposinf(arr).any():
            raise SystemExit(f"{npz_path.name}: {name} contains +inf")

    return length


def _find_source(explicit: Path | None) -> Path:
    """Locate histo_kir_hmm's models directory."""
    if explicit is not None:
        for candidate in (
            explicit / "src" / "histo_kir_hmm" / "models",
            explicit / "models",
            explicit,
        ):
            if (candidate / "manifest.json").exists():
                return candidate
        raise SystemExit(f"no models/manifest.json under {explicit} -- is that a histo_kir_hmm checkout?")

    try:
        import histo_kir_hmm
    except ImportError:
        raise SystemExit(
            "histo_kir_hmm is not importable and --histo-kir-hmm was not given; "
            "pass the path to a checkout"
        ) from None
    src = Path(histo_kir_hmm.__file__).parent / "models"
    if not (src / "manifest.json").exists():
        raise SystemExit(f"the installed histo_kir_hmm has no models at {src}")
    return src


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--histo-kir-hmm",
        type=Path,
        default=None,
        help="path to a histo_kir_hmm checkout (default: the installed package)",
    )
    args = ap.parse_args()

    src = _find_source(args.histo_kir_hmm)
    manifest = json.loads((src / "manifest.json").read_text())

    classes = manifest["classes"]
    gene_models = manifest.get("gene_models", [])
    domain_models = manifest.get("domain_models", [])

    if not gene_models:
        raise SystemExit("upstream manifest lists no gene_models")

    # Iteration order is load-bearing: it is Python's stable-sort tie-break order. Assert it is
    # what we expect rather than silently re-deriving it, and never re-sort here.
    for name, block in (("classes", classes), ("gene_models", gene_models), ("domain_models", domain_models)):
        if block != sorted(block):
            raise SystemExit(f"upstream manifest '{name}' is not sorted -- tie-break order would drift")

    calibration = manifest.get("calibration", {})
    for key in REQUIRED_CALIBRATION:
        if key not in calibration:
            raise SystemExit(
                f"upstream manifest has no calibration.{key}; the plugin would have to invent a "
                "threshold, which is a guess presented as a calibration. Retrain with "
                "scripts/train_models.py."
            )

    if manifest.get("alphabet") not in (None, ALPHABET):
        raise SystemExit(
            f"upstream manifest alphabet {manifest['alphabet']!r} != {ALPHABET!r}; the Rust "
            "encoding is hardcoded to this alphabet and would misread the models"
        )

    if not (src / "background.npy").exists():
        raise SystemExit(f"no background.npy in {src}")
    background = np.load(src / "background.npy")
    if background.shape != (ALPHABET_SIZE,):
        raise SystemExit(f"background.npy has shape {background.shape}, expected ({ALPHABET_SIZE},)")

    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    shutil.copy2(src / "background.npy", DEST / "background.npy")

    lengths: dict[str, int] = {}
    checksums: dict[str, str] = {}
    for name in classes:
        npz = src / f"{name}.npz"
        if not npz.exists():
            raise SystemExit(f"manifest lists {name} but {npz} is missing")
        lengths[name] = _validate_model(npz)
        declared = manifest.get("models", {}).get(name, {}).get("length")
        if declared is not None and int(declared) != lengths[name]:
            raise SystemExit(
                f"{name}: manifest says length {declared} but the array says {lengths[name]}"
            )
        shutil.copy2(npz, DEST / f"{name}.npz")
        checksums[f"{name}.npz"] = hashlib.sha256(npz.read_bytes()).hexdigest()[:16]

    checksums["background.npy"] = hashlib.sha256((src / "background.npy").read_bytes()).hexdigest()[:16]

    # The manifest is copied whole, with provenance appended.
    manifest["vendored_from"] = {
        "package": "histo-kir-hmm",
        "models_dir": str(src),
        "note": "models copied byte-identical; see codegen/vendor_models.py",
        "checksums": checksums,
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    size_mb = sum(f.stat().st_size for f in DEST.iterdir()) / 1e6
    print(f"vendored {len(classes)} models ({size_mb:.3f} MB) from {src} -> {DEST}")
    print(f"  {len(gene_models)} gene, {len(domain_models)} domain; lengths {min(lengths.values())}-{max(lengths.values())}")
    print(f"  calibration: per_residue_threshold={calibration['per_residue_threshold']:.6f} "
          f"tie_tolerance={calibration['tie_tolerance']:.6f} "
          f"domain_threshold={calibration['domain_threshold']:.6f}")
    print(f"  {len(manifest.get('exact_index', {}))} exact-index entries, "
          f"{len(manifest.get('cross_gene_ectodomains', []))} cross-gene ectodomain collisions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
