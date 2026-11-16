"""The expressions as Polars citizens: schema, nulls, laziness, and the tie contract."""

from __future__ import annotations

import polars as pl
import pytest

import polars_kir_hmm
from polars_kir_hmm import classify, domains, score


def test_namespace_is_registered(kir_chains):
    df = pl.DataFrame({"sequence": [kir_chains[0]["sequence"]]})
    out = df.select(pl.col("sequence").kir.classify())
    assert out.height == 1


def test_classify_schema_is_the_promised_struct(kir_chains):
    df = pl.DataFrame({"sequence": [kir_chains[0]["sequence"]]})
    dtype = df.select(pl.col("sequence").kir.classify()).schema["sequence"]
    assert isinstance(dtype, pl.Struct)
    fields = {f.name: f.dtype for f in dtype.fields}
    assert fields["chain_type"] == pl.String
    assert fields["is_kir"] == pl.Boolean
    assert fields["gene"] == pl.String
    assert fields["genes"] == pl.List(pl.String)
    assert fields["is_tie"] == pl.Boolean
    assert fields["tie_kind"] == pl.String
    assert fields["architecture"] == pl.String
    assert fields["confidence"] == pl.Float64
    assert fields["gene_confidence"] == pl.Float64
    assert fields["typicality"] == pl.Float64
    assert fields["score"] == pl.Float64
    assert fields["region_start"] == pl.UInt32
    assert fields["region_end"] == pl.UInt32
    assert fields["notes"] == pl.List(pl.String)
    assert fields["domains"] == pl.List(
        pl.Struct(
            {"domain": pl.String, "start": pl.UInt32, "end": pl.UInt32, "score": pl.Float64}
        )
    )
    assert fields["raw_scores"] == pl.List(pl.Struct({"gene": pl.String, "score": pl.Float64}))


def test_lazy_schema_matches_eager(kir_chains):
    """The declared output type must match what the expression actually produces.

    A `collect_schema` that disagrees with the realised frame is the classic plugin bug: the query
    plans fine and then fails or silently mis-types at execution.
    """
    df = pl.DataFrame({"sequence": [c["sequence"] for c in kir_chains[:3]]})
    lazy = df.lazy().with_columns(call=pl.col("sequence").kir.classify())
    assert lazy.collect_schema() == lazy.collect().schema


def test_null_sequence_gives_null_struct(kir_chains):
    """A null input is an absent sequence, not a classification that says nothing."""
    df = pl.DataFrame({"sequence": [kir_chains[0]["sequence"], None]})
    out = df.select(call=pl.col("sequence").kir.classify())
    assert out["call"][1] is None
    assert out["call"][0] is not None
    assert df.select(pl.col("sequence").kir.score("KIR3DL1")).to_series()[1] is None
    assert df.select(call=pl.col("sequence").kir.domains())["call"][1] is None


def test_all_null_column_keeps_its_schema():
    """An all-null column has no inner dtype to infer, so the cast has to pin it."""
    df = pl.DataFrame({"sequence": [None, None]}, schema={"sequence": pl.String})
    out = df.select(call=pl.col("sequence").kir.classify())
    fields = {f.name: f.dtype for f in out.schema["call"].fields}
    assert fields["genes"] == pl.List(pl.String)
    assert fields["domains"] == pl.List(
        pl.Struct({"domain": pl.String, "start": pl.UInt32, "end": pl.UInt32, "score": pl.Float64})
    )


def test_empty_frame_is_not_an_error():
    df = pl.DataFrame({"sequence": []}, schema={"sequence": pl.String})
    assert df.select(call=pl.col("sequence").kir.classify()).height == 0
    assert df.select(pl.col("sequence").kir.score("D1")).height == 0


def test_row_order_is_preserved(kir_chains):
    """`score` parallelises across rows; the result must still line up with the input."""
    seqs = [c["sequence"] for c in kir_chains[:20]]
    df = pl.DataFrame({"sequence": seqs})
    one_pass = df.select(pl.col("sequence").kir.score("KIR3DL1")).to_series().to_list()
    row_by_row = [
        pl.DataFrame({"sequence": [s]})
        .select(pl.col("sequence").kir.score("KIR3DL1"))
        .to_series()[0]
        for s in seqs
    ]
    assert one_pass == row_by_row


def test_classify_is_elementwise_under_filter(kir_chains):
    """Marked `is_elementwise`, so the optimiser may reorder it past a filter."""
    df = pl.DataFrame({"sequence": [c["sequence"] for c in kir_chains[:5]]})
    a = (
        df.lazy()
        .with_columns(call=pl.col("sequence").kir.classify())
        .filter(pl.col("call").struct.field("is_kir"))
        .collect()
    )
    b = (
        df.lazy()
        .with_columns(call=pl.col("sequence").kir.classify())
        .collect()
        .filter(pl.col("call").struct.field("is_kir"))
    )
    assert a.equals(b)


def test_function_forms_match_method_forms(kir_chains):
    df = pl.DataFrame({"sequence": [c["sequence"] for c in kir_chains[:3]]})
    assert df.select(classify("sequence")).equals(df.select(pl.col("sequence").kir.classify()))
    assert df.select(score("sequence", model="D1")).equals(
        df.select(pl.col("sequence").kir.score("D1"))
    )
    assert df.select(domains("sequence")).equals(df.select(pl.col("sequence").kir.domains()))
    # And they accept an expression, not just a column name.
    assert df.select(classify(pl.col("sequence"))).equals(df.select(classify("sequence")))


# --- the tie contract --------------------------------------------------------


def test_tied_call_has_null_gene_and_populated_genes(tied_ectodomains):
    """`histo_kir_hmm` invariant 3, at the dataframe level.

    A tie is a real, correctly-detected KIR chain whose gene cannot be determined from the
    ectodomain. `gene` must be null and `genes` must hold the set -- never a picked winner, and
    never a rejection.
    """
    df = pl.DataFrame(
        {"allele": list(tied_ectodomains), "sequence": list(tied_ectodomains.values())}
    )
    out = df.with_columns(call=pl.col("sequence").kir.classify()).unnest("call")

    tied = out.filter(pl.col("is_tie"))
    assert tied.height > 0, "the tied-ectodomain fixture should produce ties"
    # Every tied row: detected as KIR, gene null, >= 2 genes, and a note explaining why.
    assert tied["is_kir"].all()
    assert tied["gene"].null_count() == tied.height
    assert (tied["genes"].list.len() >= 2).all()
    assert (tied["notes"].list.len() >= 1).all()
    assert set(tied["tie_kind"].unique()) <= {"exact", "score"}


def test_kir2ds2_reference_allele_ties_with_kir2dl3(tied_ectodomains):
    """The concrete case that makes this not hypothetical.

    `KIR2DS2*0010101` is a reference allele appearing in four deposited structures, and its mature
    ectodomain is byte-identical to `KIR2DL3*033`.
    """
    seq = tied_ectodomains["KIR2DS2*0010101"]
    row = (
        pl.DataFrame({"sequence": [seq]})
        .with_columns(call=pl.col("sequence").kir.classify())
        .unnest("call")
        .to_dicts()[0]
    )
    assert row["is_kir"] is True
    assert row["gene"] is None
    assert row["genes"] == ["KIR2DL3", "KIR2DS2"]
    assert row["tie_kind"] == "exact"


def test_coalescing_a_label_keeps_the_ambiguity(tied_ectodomains):
    """The documented way to get one label per row without inventing a gene."""
    df = pl.DataFrame({"sequence": list(tied_ectodomains.values())})
    out = (
        df.with_columns(call=pl.col("sequence").kir.classify())
        .unnest("call")
        .with_columns(label=pl.coalesce(pl.col("gene"), pl.col("genes").list.join("/")))
    )
    for row in out.filter(pl.col("is_tie")).to_dicts():
        assert "/" in row["label"], "a tied label should name every tied gene"


# --- score and domains -------------------------------------------------------


def test_score_rejects_unknown_model_at_construction_time():
    with pytest.raises(ValueError, match="unknown model"):
        pl.col("sequence").kir.score("KIR9DL9")
    # A near-miss gets a suggestion rather than 18 names dumped at the user.
    with pytest.raises(ValueError, match="did you mean"):
        pl.col("sequence").kir.score("KIR3DL")


def test_score_accepts_gene_and_domain_models(kir_chains):
    df = pl.DataFrame({"sequence": [kir_chains[0]["sequence"]]})
    for model in ("KIR2DL1", "KIR3DL1", "D0", "D1", "D2"):
        v = df.select(pl.col("sequence").kir.score(model)).to_series()[0]
        assert v is not None and v == v  # not NaN


def test_viterbi_differs_from_log_odds(kir_chains):
    """`log_odds=False` is the raw model score, without the null subtraction."""
    df = pl.DataFrame({"sequence": [kir_chains[0]["sequence"]]})
    lod = df.select(pl.col("sequence").kir.score("KIR2DL3")).to_series()[0]
    vit = df.select(pl.col("sequence").kir.score("KIR2DL3", log_odds=False)).to_series()[0]
    # The null score of a real protein is strongly negative, so subtracting it raises the score.
    assert lod > vit


def test_domains_agrees_with_classify_on_complete_chains(kir_chains):
    """Two paths to the same segmentation: `classify` runs it on the scored region, `domains` on
    the whole cleaned sequence. For a chain short enough not to trigger a scan they coincide."""
    seqs = [c["sequence"] for c in kir_chains[:10]]
    df = pl.DataFrame({"sequence": seqs})
    via_classify = (
        df.with_columns(call=pl.col("sequence").kir.classify()).unnest("call")["architecture"].to_list()
    )
    via_domains = df.with_columns(d=pl.col("sequence").kir.domains()).unnest("d")["architecture"].to_list()
    assert via_classify == via_domains


def test_with_architecture_false_skips_segmentation(kir_chains):
    df = pl.DataFrame({"sequence": [kir_chains[0]["sequence"]]})
    full = df.with_columns(c=pl.col("sequence").kir.classify()).unnest("c").to_dicts()[0]
    fast = (
        df.with_columns(c=pl.col("sequence").kir.classify(with_architecture=False))
        .unnest("c")
        .to_dicts()[0]
    )
    # The gene call is unaffected; only the segmentation fields differ.
    assert fast["gene"] == full["gene"]
    assert fast["genes"] == full["genes"]
    assert fast["score"] == full["score"]
    assert fast["architecture"] is None
    assert fast["domains"] == []
    assert full["architecture"] is not None


# --- introspection helpers ---------------------------------------------------


def test_genes_lists_fifteen_with_structure_flags():
    g = polars_kir_hmm.genes()
    assert g.height == 15
    assert g["gene"].to_list() == sorted(g["gene"].to_list())
    # Eight genes have no deposited structure anywhere; their windows are alignment predictions.
    assert g.filter(~pl.col("has_structure")).height == 8
    assert set(g["architecture"].unique()) == {"D1D2", "D0D2", "D0D1D2"}
    # Model length follows architecture, and is what `is_kir`'s denominator is capped at.
    by_arch = dict(zip(g["architecture"], g["length"]))
    assert by_arch == {"D1D2": 199, "D0D2": 194, "D0D1D2": 294}


def test_models_lists_all_eighteen():
    m = polars_kir_hmm.models()
    assert m.height == 18
    assert m.filter(pl.col("kind") == "gene").height == 15
    assert m.filter(pl.col("kind") == "domain").height == 3
    assert set(m.filter(pl.col("kind") == "domain")["model"]) == {"D0", "D1", "D2"}


def test_calibration_carries_the_measured_thresholds():
    cal = polars_kir_hmm.calibration()
    for key in ("per_residue_threshold", "per_residue_width", "tie_tolerance", "domain_threshold"):
        assert key in cal, f"{key} is a measured number the classifier decides on"
    assert cal["per_residue_threshold"] > 0


def test_model_dir_holds_the_bundle():
    d = polars_kir_hmm.model_dir()
    assert (d / "manifest.json").exists()
    assert (d / "background.npy").exists()
    assert len(list(d.glob("*.npz"))) == 18
