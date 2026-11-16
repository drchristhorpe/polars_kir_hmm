"""Check the plugin against recorded reference outputs.

This is what runs when `histo_kir_hmm` is not importable -- it needs Python >= 3.12 for numba,
while this package's abi3 floor is 3.10, so CI's floor job has no reference to compare against.
The golden file was recorded by `tools/make_golden.py` from the real reference; regenerate it if the
models are retrained.

A supplement to `test_parity.py`, not a substitute: it proves the plugin still says what the
reference said when recorded, which catches a regression in the plugin but not a divergence
introduced by an upstream change. The provenance block records the version and model checksums it
was taken against so a stale golden file is identifiable.
"""

from __future__ import annotations

import polars as pl
import pytest

import polars_kir_hmm


def test_golden_provenance_matches_the_bundled_models(golden):
    """A golden file recorded against different models is stale, and silently so.

    The comparison is on the vendoring checksums, which change if any model array changes.
    """
    import json

    manifest = json.loads((polars_kir_hmm.model_dir() / "manifest.json").read_text())
    recorded = golden["_provenance"].get("model_checksums", {})
    current = manifest.get("vendored_from", {}).get("checksums", {})
    if not recorded or not current:
        pytest.skip("no checksums recorded to compare")
    assert recorded == current, (
        "tests/data/golden.json was recorded against different model files; regenerate it with "
        "tools/make_golden.py"
    )


def test_golden_calls(golden):
    """Every recorded case: the call fields, exactly."""
    cases = golden["cases"]
    assert cases, "the golden file holds no cases"

    rows = (
        pl.DataFrame({"sequence": [c["sequence"] for c in cases]})
        .with_columns(call=pl.col("sequence").kir.classify())
        .unnest("call")
        .to_dicts()
    )

    failures: list[str] = []
    for case, row in zip(cases, rows):
        want = case["expected"]
        for field in (
            "chain_type",
            "is_kir",
            "gene",
            "genes",
            "is_tie",
            "tie_kind",
            "architecture",
            "region_start",
            "region_end",
        ):
            if row[field] != want[field]:
                failures.append(
                    f"{case['label']} ({case['kind']}): {field} = {row[field]!r}, "
                    f"recorded {want[field]!r}"
                )
    assert not failures, "\n".join(failures[:20])


def test_golden_scores_are_bit_identical(golden):
    """`score`, the per-gene log-odds and the domain scores, with no tolerance.

    Viterbi is max-plus and both null-summation orders are reproduced, so equality is the right
    assertion. A tolerance here would let a real drift through.
    """
    cases = golden["cases"]
    rows = (
        pl.DataFrame({"sequence": [c["sequence"] for c in cases]})
        .with_columns(call=pl.col("sequence").kir.classify())
        .unnest("call")
        .to_dicts()
    )

    failures: list[str] = []
    for case, row in zip(cases, rows):
        want = case["expected"]
        if row["score"] != want["score"]:
            failures.append(
                f"{case['label']}: score {row['score']!r} != recorded {want['score']!r}"
            )
        if row["confidence"] != want["confidence"]:
            failures.append(f"{case['label']}: confidence differs")
        if row["gene_confidence"] != want["gene_confidence"]:
            failures.append(f"{case['label']}: gene_confidence differs")

        got_raw = {e["gene"]: e["score"] for e in (row["raw_scores"] or [])}
        if got_raw != want["raw_scores"]:
            differing = [
                g for g, v in want["raw_scores"].items() if got_raw.get(g) != v
            ]
            failures.append(f"{case['label']}: raw scores differ for {differing[:5]}")

        got_dom = [(d["domain"], d["start"], d["end"], d["score"]) for d in (row["domains"] or [])]
        want_dom = [(d["domain"], d["start"], d["end"], d["score"]) for d in want["domains"]]
        if got_dom != want_dom:
            failures.append(f"{case['label']}: domains differ")

    assert not failures, "\n".join(failures[:20])


def test_golden_typicality_within_tolerance(golden):
    """The one non-bit-exact field: the plugin's normal CDF is not scipy's Cephes `ndtr`."""
    cases = golden["cases"]
    rows = (
        pl.DataFrame({"sequence": [c["sequence"] for c in cases]})
        .with_columns(call=pl.col("sequence").kir.classify(with_architecture=False))
        .unnest("call")
        .to_dicts()
    )
    worst = 0.0
    worst_label = ""
    for case, row in zip(cases, rows):
        want = case["expected"]["typicality"]
        got = row["typicality"]
        delta = abs(got - want)
        if delta > worst:
            worst, worst_label = delta, case["label"]
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12), (
            f"{case['label']}: typicality {got!r} vs recorded {want!r}"
        )
    # Recorded in the failure message of a future regression, and a useful bound to see pass.
    assert worst < 1e-9, f"worst typicality delta {worst:.3e} on {worst_label}"


def test_golden_covers_every_case_kind(golden):
    """A golden file that lost its ties or its negatives would pass vacuously."""
    kinds = {c["kind"] for c in golden["cases"]}
    assert {
        "structural_kir",
        "structural_non_kir",
        "curated_negative",
        "tied_ectodomain",
        "allele",
    } <= kinds
    tied = [c for c in golden["cases"] if c["kind"] == "tied_ectodomain"]
    assert any(c["expected"]["is_tie"] for c in tied), "no tie recorded in the golden file"
    assert any(
        not c["expected"]["is_kir"] for c in golden["cases"] if c["kind"] == "curated_negative"
    ), "no rejection recorded in the golden file"
