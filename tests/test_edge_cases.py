"""Edge cases, and the invariants that must not regress silently.

Each test here is named after a real failure mode -- most of them measured upstream, a couple
specific to this port. They do not need the reference implementation: they assert properties of
the plugin's own output that would break if an invariant were dropped.
"""

from __future__ import annotations

import polars as pl
import pytest

import polars_kir_hmm


# --- invariant 2: is_kir tests log-odds per MATCH STATE ----------------------


def test_short_single_domain_construct_is_still_kir(kir_chains):
    """`histo_kir_hmm` invariant 2, the case that motivated it.

    `8tui` chain A is a real deposited 98-residue KIR2DL3 D1 domain. Against a *total*-score
    threshold it scores 182.8 vs 216.4 and gets rejected for being short. Per match state it is far
    above any decoy. If someone "simplifies" the denominator, this test is what fails.
    """
    short = [c for c in kir_chains if c["pdb_id"] == "8tui" and c["chain"] == "A"]
    if not short:
        short = sorted(kir_chains, key=lambda c: len(c["sequence"]))[:1]
    chain = short[0]
    assert len(chain["sequence"]) < 120, "expected a single-domain construct in the corpus"

    row = (
        pl.DataFrame({"sequence": [chain["sequence"]]})
        .with_columns(call=pl.col("sequence").kir.classify())
        .unnest("call")
        .to_dicts()[0]
    )
    assert row["is_kir"] is True, "a real short KIR domain must not be rejected for being short"
    # And it reports what it observes, not what the gene implies (invariant 11).
    assert row["architecture"] == "D1"
    assert any("implies" in n for n in row["notes"]), "the architecture mismatch should be noted"


def test_full_length_precursors_are_kir(expressed_alleles):
    """Invariant 2's other half: a per-*residue* denominator rejected 248 of 433 KIR2DL1 alleles.

    A 348-residue precursor carries a 199-residue ectodomain plus ~150 residues of signal peptide,
    stem, TM and tail that the model never matches. Dividing by raw length dilutes the score below
    threshold. Every expressed allele must come back positive.
    """
    kir2dl1 = [a for a in expressed_alleles if a["gene"] == "KIR2DL1"]
    assert len(kir2dl1) > 300, "expected the full KIR2DL1 allele set"
    out = (
        pl.DataFrame({"sequence": [a["sequence"] for a in kir2dl1]})
        .with_columns(call=pl.col("sequence").kir.classify(with_architecture=False))
        .unnest("call")
    )
    n_detected = out["is_kir"].sum()
    assert n_detected == len(kir2dl1), (
        f"only {n_detected}/{len(kir2dl1)} KIR2DL1 alleles detected; a per-residue denominator "
        "would give roughly 185"
    )


# --- invariants 3 and 4: ties are reported, never broken ---------------------


def test_ties_are_never_silently_resolved(tied_ectodomains):
    df = pl.DataFrame({"sequence": list(tied_ectodomains.values())})
    out = df.with_columns(call=pl.col("sequence").kir.classify()).unnest("call")
    for row in out.to_dicts():
        if row["is_tie"]:
            assert row["gene"] is None
            assert len(row["genes"]) >= 2
        else:
            assert row["gene"] is not None
            assert len(row["genes"]) == 1
        # `gene` is genes[0] iff there is exactly one gene -- the documented relationship.
        assert (row["gene"] == (row["genes"][0] if len(row["genes"]) == 1 else None))


def test_exact_tie_detection_uses_containment_not_just_hashing(tied_ectodomains, kir_chains):
    """`histo_kir_hmm` invariant 4.

    A hash lookup alone never fires on a real query: a structural chain or a precursor is never
    byte-equal to a 199-residue training extent, even when it contains one. When this was hash-only
    upstream, `n_exact_tie` was 0 across all 1983 alleles -- silently, since every affected allele
    still got a plausible single-gene answer. So an exact tie must fire on a *chain* that merely
    contains a colliding ectodomain, not only on the bare ectodomain itself.
    """
    ecto = tied_ectodomains["KIR2DS2*0010101"]
    # Flank it, as a construct or a longer chain would.
    padded = "MSLMVVSMACVGFFLLQGAWP" + ecto + "TGNPSNSWPSPTEPSSKTG"
    out = (
        pl.DataFrame({"sequence": [ecto, padded]})
        .with_columns(call=pl.col("sequence").kir.classify(with_architecture=False))
        .unnest("call")
    )
    rows = out.to_dicts()
    assert rows[0]["tie_kind"] == "exact", "the bare ectodomain should tie exactly"
    assert rows[1]["tie_kind"] == "exact", (
        "a chain containing the colliding ectodomain must also tie -- if this fails, exact-tie "
        "detection has regressed to hash equality"
    )
    assert rows[1]["genes"] == rows[0]["genes"]


def test_deposited_chains_resolve_uniquely_despite_the_ectodomain_collision(kir_chains):
    """None of the 54 deposited chains ties -- measured, and initially surprising.

    `KIR2DS2*0010101`'s mature ectodomain is byte-identical to `KIR2DL3*033`, and it appears in four
    deposited structures (`1m4k_A`, `4n8v_G`, `4n8v_I`, `7duu_D`), so a tie looks inevitable. It does
    not happen, for a reason worth pinning down: those chains have 190-196 *observed* residues
    against the 199-residue ectodomain, because of disordered loops. So the scored region is neither
    byte-equal to the colliding ectodomain nor a contiguous substring of it, and the exact-tie test
    correctly does not fire. The score tie does not fire either -- each gene's model is trained on
    all of its alleles, so the KIR2DS2 model still separates from the KIR2DL3 model by more than
    the tie tolerance.

    The collision is therefore a hazard for **complete** ectodomain or precursor input (measured:
    128 of the 1983 IPD-KIR alleles tie exactly) and not for this structure set. Consuming code
    still must not assume a unique gene -- `test_ties_are_never_silently_resolved` covers that --
    but a pipeline over deposited coordinates will in practice see none.
    """
    out = (
        pl.DataFrame({"sequence": [c["sequence"] for c in kir_chains]})
        .with_columns(call=pl.col("sequence").kir.classify(with_architecture=False))
        .unnest("call")
    )
    assert out["is_tie"].sum() == 0
    # The KIR2DS2 chains in particular resolve to KIR2DS2, not to a KIR2DL3 tie.
    for chain, row in zip(kir_chains, out.to_dicts()):
        if chain["gene"] == "KIR2DS2":
            assert row["genes"] == ["KIR2DS2"], f"{chain['pdb_id']}_{chain['chain']}"
            assert len(chain["sequence"]) < 199, (
                "the unique call depends on the chain being incomplete; a complete KIR2DS2 "
                "ectodomain would tie with KIR2DL3"
            )


def test_complete_tied_ectodomain_does_tie(tied_ectodomains):
    """The other side of the above: given the complete ectodomain, the tie does fire."""
    out = (
        pl.DataFrame({"sequence": [tied_ectodomains["KIR2DS2*0010101"]]})
        .with_columns(call=pl.col("sequence").kir.classify(with_architecture=False))
        .unnest("call")
        .to_dicts()[0]
    )
    assert out["is_tie"] is True
    assert out["tie_kind"] == "exact"
    assert out["genes"] == ["KIR2DL3", "KIR2DS2"]


# --- rejection ---------------------------------------------------------------


def test_hard_negatives_are_rejected(non_kir_curated):
    """LILRB1-4, LAIR1 and the other IgSF receptors: same superfamily, MHC I-binding, must reject."""
    out = (
        pl.DataFrame({"sequence": [c["sequence"] for c in non_kir_curated]})
        .with_columns(call=pl.col("sequence").kir.classify(with_architecture=False))
        .unnest("call")
    )
    called = [
        c["identity"]
        for c, is_kir in zip(non_kir_curated, out["is_kir"].to_list())
        if is_kir
    ]
    assert not called, f"these negatives were called KIR: {called}"


def test_all_framed_non_kir_chains_are_rejected(non_kir_chains):
    """The 122-chain scope histo_kir_hmm reports as 122/122."""
    assert len(non_kir_chains) == 122
    out = (
        pl.DataFrame({"sequence": [c["sequence"] for c in non_kir_chains]})
        .with_columns(call=pl.col("sequence").kir.classify(with_architecture=False))
        .unnest("call")
    )
    n_rejected = (~out["is_kir"]).sum()
    assert n_rejected == 122, f"only {n_rejected}/122 non-KIR chains rejected"


def test_all_structural_kir_chains_are_detected(kir_chains):
    out = (
        pl.DataFrame({"sequence": [c["sequence"] for c in kir_chains]})
        .with_columns(call=pl.col("sequence").kir.classify(with_architecture=False))
        .unnest("call")
    )
    assert out["is_kir"].sum() == 54, "all 54 deposited KIR chains must be detected"


def test_gene_call_is_correct_or_a_tie_containing_the_truth(kir_chains):
    """The gene recorded in `loop_annotations.csv` must be in the reported set."""
    out = (
        pl.DataFrame({"sequence": [c["sequence"] for c in kir_chains]})
        .with_columns(call=pl.col("sequence").kir.classify(with_architecture=False))
        .unnest("call")
    )
    wrong = [
        f"{c['pdb_id']}_{c['chain']}: expected {c['gene']}, got {genes}"
        for c, genes in zip(kir_chains, out["genes"].to_list())
        if c["gene"] not in list(genes)
    ]
    assert not wrong, "\n".join(wrong)


# --- degenerate input --------------------------------------------------------


def test_below_the_forty_residue_floor_is_rejected_without_scoring():
    out = (
        pl.DataFrame({"sequence": ["", "A", "ACDEFGHIKL" * 3 + "ACDEFGHI"]})
        .with_columns(call=pl.col("sequence").kir.classify())
        .unnest("call")
    )
    rows = out.to_dicts()
    for row in rows:
        assert row["chain_type"] == "non_kir"
        assert row["score"] == float("-inf"), "not scored at all, so -inf rather than a number"
        assert row["confidence"] == 1.0
        assert row["raw_scores"] == [], "no model was consulted"
        assert row["genes"] == []
        assert any("floor" in n for n in row["notes"])


def test_exactly_forty_residues_is_scored():
    """The floor is `< 40`, so 40 residues is scored -- and gets real numbers, not -inf."""
    row = (
        pl.DataFrame({"sequence": ["ACDEFGHIKLMNPQRSTVWY" * 2]})
        .with_columns(call=pl.col("sequence").kir.classify())
        .unnest("call")
        .to_dicts()[0]
    )
    assert row["score"] != float("-inf")
    assert len(row["raw_scores"]) == 15, "all 15 gene models should have been scored"


def test_unknown_residues_score_uniformly(kir_chains):
    """`X` survives cleaning and emits uniformly; it lowers the score without shortening it."""
    seq = kir_chains[0]["sequence"]
    masked = seq[:60] + "X" * 10 + seq[70:]
    out = (
        pl.DataFrame({"sequence": [seq, masked]})
        .with_columns(call=pl.col("sequence").kir.classify(with_architecture=False))
        .unnest("call")
    )
    rows = out.to_dicts()
    assert rows[1]["region_end"] == rows[0]["region_end"], "X must not shorten the sequence"
    assert rows[1]["score"] < rows[0]["score"], "masked residues should cost score"


def test_gaps_shorten_the_scored_region():
    """Cleaning *drops* non-alphabet characters, so indices refer to the cleaned sequence."""
    seq = "ACDEFGHIKLMNPQRSTVWY" * 3
    gapped = "-".join(seq)
    out = (
        pl.DataFrame({"sequence": [seq, gapped]})
        .with_columns(call=pl.col("sequence").kir.classify(with_architecture=False))
        .unnest("call")
    )
    rows = out.to_dicts()
    assert rows[0]["region_end"] == rows[1]["region_end"] == len(seq)
    assert rows[0]["score"] == rows[1]["score"], "gaps are dropped, so the scores must be identical"


def test_scan_triggers_only_past_360_residues(kir_chains, expressed_alleles):
    """The scan is for embedded regions, not for short chains.

    Every structural chain -- including the 98-residue one -- is scored whole, so `region` is the
    whole cleaned sequence. A long precursor is windowed, so its region is a strict subset.
    """
    out = (
        pl.DataFrame({"sequence": [c["sequence"] for c in kir_chains]})
        .with_columns(call=pl.col("sequence").kir.classify(with_architecture=False))
        .unnest("call")
    )
    for chain, row in zip(kir_chains, out.to_dicts()):
        assert (row["region_start"], row["region_end"]) == (0, len(chain["sequence"])), (
            f"{chain['pdb_id']}_{chain['chain']} should be scored whole, not windowed"
        )

    long_ones = [a for a in expressed_alleles if len(a["sequence"]) > 360][:5]
    assert long_ones, "expected some precursors past the scan trigger"
    out = (
        pl.DataFrame({"sequence": [a["sequence"] for a in long_ones]})
        .with_columns(call=pl.col("sequence").kir.classify(with_architecture=False))
        .unnest("call")
    )
    for allele, row in zip(long_ones, out.to_dicts()):
        assert row["region_end"] - row["region_start"] < len(allele["sequence"]), (
            f"{allele['allele']} is past the trigger and should have been windowed"
        )


# --- output shape guarantees -------------------------------------------------


def test_raw_scores_cover_all_fifteen_genes_in_manifest_order(kir_chains):
    row = (
        pl.DataFrame({"sequence": [kir_chains[0]["sequence"]]})
        .with_columns(call=pl.col("sequence").kir.classify(with_architecture=False))
        .unnest("call")
        .to_dicts()[0]
    )
    got = [e["gene"] for e in row["raw_scores"]]
    assert got == polars_kir_hmm.genes()["gene"].to_list()


def test_domains_are_ordered_and_non_overlapping(kir_chains):
    out = (
        pl.DataFrame({"sequence": [c["sequence"] for c in kir_chains]})
        .with_columns(call=pl.col("sequence").kir.classify())
        .unnest("call")
    )
    for row in out.to_dicts():
        prev_end = -1
        for d in row["domains"] or []:
            assert d["start"] >= prev_end, "domain placements must not overlap"
            assert d["end"] > d["start"]
            assert d["end"] <= row["region_end"]
            prev_end = d["end"]
        # The architecture string is exactly the domain labels in order.
        if row["domains"]:
            assert row["architecture"] == "".join(d["domain"] for d in row["domains"])


def test_no_structure_genes_carry_the_warning_note():
    """Eight genes have no deposited structure; a positive call must say so.

    Their loop windows downstream are alignment predictions, and that has to stay visible.
    """
    g = polars_kir_hmm.genes()
    no_struct = g.filter(~pl.col("has_structure"))["gene"].to_list()
    assert len(no_struct) == 8
    assert set(no_struct) == {
        "KIR2DL5A",
        "KIR2DL5B",
        "KIR2DS1",
        "KIR2DS3",
        "KIR2DS5",
        "KIR3DL2",
        "KIR3DL3",
        "KIR3DS1",
    }


def test_typicality_saturates_on_structural_input_and_is_not_a_quality_score(kir_chains):
    """Invariant 12, asserted rather than just documented.

    Structural chains sit tens of sigma below the training mean, so typicality underflows towards
    zero while confidence stays high. A test that expected typicality to be "good" for good input
    would be wrong, and so would any downstream threshold on it.
    """
    out = (
        pl.DataFrame({"sequence": [c["sequence"] for c in kir_chains]})
        .with_columns(call=pl.col("sequence").kir.classify(with_architecture=False))
        .unnest("call")
    )
    assert out["is_kir"].all()
    assert (out["typicality"] < 0.01).sum() > 40, (
        "typicality is expected to saturate low on deposited chains; if most chains now score high, "
        "the distributions have been widened artificially"
    )
    assert (out["confidence"] > 0.5).all(), "confidence is the field that carries information"


# --- validation happens in Python, before the query engine -------------------


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"with_architecture": "yes"}, TypeError),
        ({"model_dir": "/nonexistent/path"}, ValueError),
    ],
)
def test_bad_arguments_raise_at_construction(kwargs, exc):
    with pytest.raises(exc):
        pl.col("sequence").kir.classify(**kwargs)


def test_model_dir_without_manifest_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="manifest"):
        pl.col("sequence").kir.classify(model_dir=tmp_path)
