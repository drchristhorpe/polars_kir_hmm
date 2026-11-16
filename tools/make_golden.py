"""Record reference outputs from histo_kir_hmm into `tests/data/golden.json`.

The parity suite needs the reference importable, which needs Python >= 3.12 for numba, while this
package's abi3 floor is 3.10. The golden file closes that gap: reference answers recorded once, so
`tests/test_golden.py` can check the plugin on any supported interpreter and in CI's abi3-floor job.

It is a *supplement* to parity, never a substitute. A golden file only proves the plugin still
agrees with whatever the reference said on the day it was recorded, which is why it stores the
histo_kir_hmm version and the model checksums it was recorded against.

Run:  python tools/make_golden.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n-alleles", type=int, default=40, help="alleles to record (spread by gene)")
    args = ap.parse_args()

    try:
        import histo_kir_hmm
        from histo_kir_hmm.classify import KIRClassifier
    except ImportError:
        raise SystemExit("histo_kir_hmm is not importable; cannot record golden outputs")

    classifier = KIRClassifier()
    structural = json.loads((DATA / "structural_chains.json").read_text())
    alleles = json.loads((DATA / "ipdkir_alleles.json").read_text())["alleles"]

    # Spread the allele sample across genes rather than taking the first N, which would be all
    # KIR2DL1 -- and include the null alleles, whose rejection is part of the contract.
    by_gene: dict[str, list[dict]] = {}
    for a in alleles:
        by_gene.setdefault(a["gene"], []).append(a)
    per_gene = max(1, args.n_alleles // len(by_gene))
    sampled = [a for g in sorted(by_gene) for a in by_gene[g][:per_gene]]
    sampled += [a for a in alleles if a["is_null"]][:5]

    cases: list[dict] = []

    def record(label: str, kind: str, sequence: str) -> None:
        c = classifier.type_chain(sequence)
        cases.append(
            {
                "label": label,
                "kind": kind,
                "sequence": sequence,
                "expected": {
                    "chain_type": c.chain_type,
                    "is_kir": c.is_kir,
                    "gene": c.gene,
                    "genes": list(c.genes),
                    "is_tie": c.is_tie,
                    "tie_kind": c.tie_kind,
                    "architecture": c.architecture,
                    "domains": [
                        {"domain": d.domain, "start": d.start, "end": d.end, "score": d.score}
                        for d in c.domains
                    ],
                    "confidence": c.confidence,
                    "gene_confidence": c.gene_confidence,
                    "typicality": c.typicality,
                    "score": c.score,
                    "region_start": c.scored_region[0],
                    "region_end": c.scored_region[1],
                    "n_notes": len(c.notes),
                    "raw_scores": dict(c.raw_scores),
                },
            }
        )

    for chain in structural["kir_chains"]:
        record(f"{chain['pdb_id']}_{chain['chain']}", "structural_kir", chain["sequence"])
    # A sample of the 122, plus every curated hard negative.
    for chain in structural["non_kir_chains"][:20]:
        record(f"{chain['file']}_{chain['chain']}", "structural_non_kir", chain["sequence"])
    for chain in structural["non_kir_curated"]:
        record(chain["identity"], "curated_negative", chain["sequence"])
    for allele, sequence in structural["tied_ectodomains"].items():
        record(allele, "tied_ectodomain", sequence)
    for a in sampled:
        record(a["allele"], "allele", a["sequence"])

    manifest = json.loads((ROOT / "python" / "polars_kir_hmm" / "models" / "manifest.json").read_text())
    (DATA / "golden.json").write_text(
        json.dumps(
            {
                "_provenance": {
                    "recorded_from": "histo_kir_hmm",
                    "histo_kir_hmm_version": getattr(histo_kir_hmm, "__version__", "unknown"),
                    "model_dir": str(classifier.model_dir),
                    "model_checksums": manifest.get("vendored_from", {}).get("checksums", {}),
                    "calibration": manifest.get("calibration", {}),
                    "note": (
                        "Reference answers for machines without numba (the reference needs Python "
                        ">= 3.12; this package supports >= 3.10). A supplement to the parity suite, "
                        "not a substitute: it only proves agreement with what the reference said "
                        "when recorded, hence the version and checksums above."
                    ),
                },
                "cases": cases,
            },
            indent=1,
        )
        + "\n"
    )
    from collections import Counter

    print(f"recorded {len(cases)} golden cases -> {DATA / 'golden.json'}")
    for kind, n in sorted(Counter(c["kind"] for c in cases).items()):
        print(f"  {kind}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
