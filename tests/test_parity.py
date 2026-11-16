"""Parity against `histo_kir_hmm`: identical calls, identical scores.

This is the suite that justifies the package existing. A plugin that is merely *close* to the
reference is a second implementation that will drift, and then two answers exist for the same
question with nothing to say which is right. So the assertions here are equality, not tolerance,
everywhere the arithmetic permits it:

* `score`, the per-gene log-odds and the domain scores are asserted **exactly equal**. Viterbi is
  max-plus, so there is no summation order to differ; the null score has one, and `src/hmm.rs`
  reproduces both of the reference's orders (numpy blocked-pairwise for `log_odds_score`,
  sequential for `log_odds_score_batch`).
* `typicality` is asserted to a tolerance, because it needs a normal CDF and the plugin computes
  `erfc` via the incomplete gamma function rather than vendoring scipy's Cephes coefficients.

Skipped wholesale when `histo_kir_hmm` is not importable (it needs Python >= 3.12 for numba, while
this package's abi3 floor is 3.10); `test_golden.py` is what runs then.
"""

from __future__ import annotations

import polars as pl
import pytest

# Fields of ChainType that must agree exactly, and are what a caller dispatches on.
CALL_FIELDS = ("chain_type", "is_kir", "gene", "genes", "is_tie", "tie_kind", "architecture")


def plugin_rows(sequences: list[str]) -> list[dict]:
    """One Polars pass, as a list of row dicts."""
    return (
        pl.DataFrame({"sequence": sequences})
        .with_columns(call=pl.col("sequence").kir.classify())
        .unnest("call")
        .to_dicts()
    )


def assert_call_matches(row: dict, ref, label: str) -> None:
    """Every dispatched-on field, plus the scored region and the domain boundaries."""
    assert row["chain_type"] == ref.chain_type, f"{label}: chain_type"
    assert row["is_kir"] == ref.is_kir, f"{label}: is_kir"
    assert row["gene"] == ref.gene, f"{label}: gene"
    assert row["genes"] == list(ref.genes), f"{label}: genes"
    assert row["is_tie"] == ref.is_tie, f"{label}: is_tie"
    assert row["tie_kind"] == ref.tie_kind, f"{label}: tie_kind"
    assert row["architecture"] == ref.architecture, f"{label}: architecture"
    assert (row["region_start"], row["region_end"]) == ref.scored_region, f"{label}: scored_region"
    assert [(d["domain"], d["start"], d["end"]) for d in (row["domains"] or [])] == [
        (d.domain, d.start, d.end) for d in ref.domains
    ], f"{label}: domain boundaries"


def assert_scores_match(row: dict, ref, label: str) -> None:
    """Scores exactly; `typicality` to a tolerance, for the reason in the module docstring."""
    assert row["score"] == ref.score, f"{label}: score {row['score']!r} != {ref.score!r}"
    assert row["confidence"] == ref.confidence, f"{label}: confidence"
    assert row["gene_confidence"] == ref.gene_confidence, f"{label}: gene_confidence"

    plugin_raw = {e["gene"]: e["score"] for e in (row["raw_scores"] or [])}
    assert plugin_raw == pytest.approx(ref.raw_scores, abs=0.0), f"{label}: raw per-gene scores"

    for d_row, d_ref in zip(row["domains"] or [], ref.domains):
        assert d_row["score"] == d_ref.score, f"{label}: {d_ref.domain} score"

    # typicality: the one non-bit-exact value. It saturates to 0 on structural input, so compare
    # absolutely there and relatively in the interior of the range.
    assert row["typicality"] == pytest.approx(ref.typicality, rel=1e-9, abs=1e-12), (
        f"{label}: typicality"
    )


# --- the three corpora -------------------------------------------------------


def test_parity_on_structural_kir_chains(kir_chains, classifier):
    """The 54 deposited KIR chains -- the check that matters most.

    Real coordinates: disordered loops, internal numbering breaks, truncated constructs, and one
    lone D1 domain (`8tui` chain A, 98 residues) that a total-score threshold would reject.
    """
    sequences = [c["sequence"] for c in kir_chains]
    rows = plugin_rows(sequences)
    assert len(rows) == 54

    for chain, row in zip(kir_chains, rows):
        label = f"{chain['pdb_id']}_{chain['chain']}"
        ref = classifier.type_chain(chain["sequence"])
        assert_call_matches(row, ref, label)
        assert_scores_match(row, ref, label)


def test_parity_on_non_kir_chains(non_kir_chains, classifier):
    """All 122 non-KIR chains across the 54 framed files -- MHC I heavy, b2m, peptide, Fv, RIFIN.

    122 is the scope `histo_kir_hmm/VERIFICATION_REPORT.md` reports as 122/122 rejected, so this
    matches it rather than the 22-chain curated fixture.

    A rejection has to match too -- including its `confidence`, which is confidence in the
    *rejection*.
    """
    assert len(non_kir_chains) == 122
    sequences = [c["sequence"] for c in non_kir_chains]
    rows = plugin_rows(sequences)

    for chain, row in zip(non_kir_chains, rows):
        label = f"{chain['file']}_{chain['chain']}"
        ref = classifier.type_chain(chain["sequence"])
        assert_call_matches(row, ref, label)
        assert row["confidence"] == ref.confidence, f"{label}: rejection confidence"
        # Sanity: these really are negatives, so a passing test cannot be vacuous.
        assert ref.chain_type == "non_kir", f"{label} should reject"


def test_parity_on_curated_hard_negatives(non_kir_curated, classifier):
    """The LILR/LAIR and Ig-superfamily receptors, which appear in no structure file.

    LILRB1 is the load-bearing negative: a four-Ig-domain MHC class I-binding NK receptor in the
    same superfamily as KIR. If the plugin's threshold arithmetic drifted, this is where it shows.
    """
    sequences = [c["sequence"] for c in non_kir_curated]
    rows = plugin_rows(sequences)
    for chain, row in zip(non_kir_curated, rows):
        ref = classifier.type_chain(chain["sequence"])
        assert_call_matches(row, ref, chain["identity"])
        assert row["confidence"] == ref.confidence, f"{chain['identity']}: rejection confidence"


def test_parity_on_tied_ectodomains(tied_ectodomains, classifier):
    """The ectodomain collisions, where the reference's answer is `None` with a set."""
    labels = list(tied_ectodomains)
    rows = plugin_rows([tied_ectodomains[k] for k in labels])
    for label, row in zip(labels, rows):
        ref = classifier.type_chain(tied_ectodomains[label])
        assert_call_matches(row, ref, label)
        assert_scores_match(row, ref, label)


@pytest.mark.slow
def test_parity_on_all_ipdkir_alleles(alleles, classifier):
    """All 1983 IPD-KIR alleles, full-length precursors.

    Slow (tens of minutes): every 3D-gene precursor is longer than the 360-residue scan trigger, so
    this is the corpus that exercises the sliding-window path -- hundreds of windows x 15 models per
    sequence -- where the two null-summation orders could diverge and flip an argmax. It also covers
    the null alleles, whose frameshifted fragments must be rejected identically.
    """
    sequences = [a["sequence"] for a in alleles]
    rows = plugin_rows(sequences)
    assert len(rows) == 1983

    mismatches: list[str] = []
    for allele, row in zip(alleles, rows):
        ref = classifier.type_chain(allele["sequence"])
        try:
            assert_call_matches(row, ref, allele["allele"])
            assert_scores_match(row, ref, allele["allele"])
        except AssertionError as e:
            mismatches.append(str(e).splitlines()[0])

    assert not mismatches, f"{len(mismatches)} allele(s) disagree:\n" + "\n".join(mismatches[:20])


@pytest.mark.slow
def test_scan_path_parity_on_long_constructs(alleles, classifier):
    """Isolate the windowed path: the longest precursors, well past the scan trigger.

    Called out separately from the full-corpus test so a scan-specific regression is legible in the
    test name rather than buried in 1983 rows.
    """
    long_ones = sorted(alleles, key=lambda a: -len(a["sequence"]))[:25]
    assert len(long_ones[0]["sequence"]) > 360, "expected at least one construct past the trigger"

    rows = plugin_rows([a["sequence"] for a in long_ones])
    for allele, row in zip(long_ones, rows):
        ref = classifier.type_chain(allele["sequence"])
        # The window the scan picked is the thing at risk here.
        assert (row["region_start"], row["region_end"]) == ref.scored_region, (
            f"{allele['allele']}: scan chose a different window"
        )
        assert_call_matches(row, ref, allele["allele"])
        assert_scores_match(row, ref, allele["allele"])


# --- the expressions other than classify -------------------------------------


def test_score_expression_parity_across_every_model(kir_chains, classifier):
    """`score` against all 18 models must equal `ProfileHMM.log_odds_score` exactly."""
    sequences = [c["sequence"] for c in kir_chains[:12]]
    df = pl.DataFrame({"sequence": sequences})

    for name, model in classifier.models.items():
        got = df.select(pl.col("sequence").kir.score(name)).to_series().to_list()
        from histo_kir_hmm.classify import _clean_sequence

        want = [model.log_odds_score(_clean_sequence(s)) for s in sequences]
        assert got == want, f"{name}: log-odds differ"


def test_viterbi_expression_parity(kir_chains, classifier):
    sequences = [c["sequence"] for c in kir_chains[:8]]
    df = pl.DataFrame({"sequence": sequences})
    from histo_kir_hmm.classify import _clean_sequence

    for name in ("KIR2DL1", "KIR3DL1", "D1"):
        got = df.select(pl.col("sequence").kir.score(name, log_odds=False)).to_series().to_list()
        want = [classifier.models[name].viterbi(_clean_sequence(s)) for s in sequences]
        assert got == want, f"{name}: raw Viterbi scores differ"


def test_domains_expression_parity(kir_chains, classifier):
    """`domains` must equal `call_architecture` -- boundaries, scores and confidence."""
    sequences = [c["sequence"] for c in kir_chains[:15]]
    rows = (
        pl.DataFrame({"sequence": sequences})
        .with_columns(d=pl.col("sequence").kir.domains())
        .unnest("d")
        .to_dicts()
    )
    for seq, row in zip(sequences, rows):
        ref = classifier.call_architecture(seq)
        assert row["architecture"] == ref["architecture"]
        assert [(d["domain"], d["start"], d["end"]) for d in (row["domains"] or [])] == [
            (d.domain, d.start, d.end) for d in ref["domains"]
        ]
        for d_row, d_ref in zip(row["domains"] or [], ref["domains"]):
            assert d_row["score"] == d_ref.score
        assert row["confidence"] == pytest.approx(ref["confidence"], rel=1e-12, abs=1e-12)


def test_cleaning_parity_on_dirty_input(kir_chains, classifier):
    """Cleaning must agree, or the two are scoring different sequences.

    Gaps and punctuation are *dropped* rather than replaced, which shortens the sequence and shifts
    every index -- so `region_end` is the tell.
    """
    from histo_kir_hmm.classify import _clean_sequence

    base = kir_chains[0]["sequence"]
    dirty = [
        base.lower(),
        "-".join(base[i : i + 10] for i in range(0, len(base), 10)),
        base.replace("A", "*A", 1),
        f"  {base}  ",
        base[:50] + "X" * 5 + base[50:],
        base[:40] + "BZJ?" + base[40:],
    ]
    rows = plugin_rows(dirty)
    for seq, row in zip(dirty, rows):
        ref = classifier.type_chain(seq)
        assert row["region_end"] == ref.scored_region[1] == len(_clean_sequence(seq))
        assert_call_matches(row, ref, repr(seq[:20]))
        assert_scores_match(row, ref, repr(seq[:20]))
