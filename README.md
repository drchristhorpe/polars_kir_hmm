# polars-kir-hmm

Type KIR chains inside Polars. A Rust expression plugin wrapping
[`histo_kir_hmm`](../histo_kir_hmm)'s trained profile HMMs, so a whole column of sequences is
scored in one pass without the GIL.

It is a **performance path, not a second implementation**. The scoring kernel is a transcription of
`histo_kir_hmm`'s numba Viterbi core, the classifier is a transcription of its `KIRClassifier`, and
the models are the same `.npz` files copied byte-identical. The parity suite asserts identical calls
and scores across the full 1983-allele IPD-KIR set and all 54 deposited structural KIR chains; see
[Parity](#parity) for the measured numbers.

Sibling of [`polars_mhci_hmm`](https://github.com/drchristhorpe/polars_mhci_hmm), which does the
same for MHC class I loci. Same layout, same ergonomics.

## Install

```bash
uv sync
uv run maturin develop --release
```

The models are vendored into the package at build time. From a fresh checkout, or after retraining
upstream:

```bash
uv run python codegen/vendor_models.py --histo-kir-hmm ../histo_kir_hmm
```

## Use

```python
import polars as pl
import polars_kir_hmm  # noqa: F401  -- registers the .kir namespace

df = pl.DataFrame({"chain": ["A", "B"], "sequence": [kir_seq, hla_seq]})

out = df.with_columns(call=pl.col("sequence").kir.classify()).unnest("call")
out.select("chain", "chain_type", "gene", "genes", "is_tie", "architecture", "confidence")
```

```text
┌───────┬────────────┬─────────┬──────────────────────────┬────────┬──────────────┬────────────┐
│ chain ┆ chain_type ┆ gene    ┆ genes                    ┆ is_tie ┆ architecture ┆ confidence │
╞═══════╪════════════╪═════════╪══════════════════════════╪════════╪══════════════╪════════════╡
│ A     ┆ kir        ┆ KIR3DL1 ┆ ["KIR3DL1"]              ┆ false  ┆ D0D1D2       ┆ 0.972      │
│ B     ┆ non_kir    ┆ null    ┆ []                       ┆ false  ┆ null         ┆ 1.000      │
└───────┴────────────┴─────────┴──────────────────────────┴────────┴──────────────┴────────────┘
```

Three expressions, each with a function form:

| expression | returns | mirrors |
| --- | --- | --- |
| `.kir.classify()` | struct: `chain_type`, `is_kir`, `gene`, `genes`, `is_tie`, `tie_kind`, `architecture`, `domains`, `confidence`, `gene_confidence`, `typicality`, `score`, `region_start`, `region_end`, `notes`, `raw_scores` | `type_chain` / `ChainType` |
| `.kir.score(model)` | `Float64` log-odds against one of the 18 models | `ProfileHMM.log_odds_score` |
| `.kir.domains()` | struct: `architecture`, `domains`, `confidence` | `call_architecture` |

```python
from polars_kir_hmm import classify, score, domains, genes, models, calibration

df.with_columns(classify("sequence"), lod=score("sequence", model="KIR3DL1"))
genes()        # the 15 genes, with architecture, model length, training count, has_structure
models()       # all 18 models (15 gene + D0/D1/D2), with kind and length
calibration()  # the measured thresholds the classifier decides on
```

`classify(with_architecture=False)` skips domain segmentation, which is most of the work when only
the gene call is wanted.

## Three things about the output that are contract

**A tied call has `gene` null.** Read `genes`. 127 IPD-KIR alleles have a mature ectodomain
byte-identical to an allele of a *different* gene, in three pairs — KIR2DL5A/KIR2DL5B (71 alleles),
KIR2DL3/KIR2DS2 (54), KIR2DL2/KIR2DL3 (2). Full-length precursors separate them; the ectodomain,
which is all a structural chain contains, does not. This is not a corner case: `KIR2DS2*0010101` is
a reference allele, appears in four deposited structures, and its ectodomain is byte-identical to
`KIR2DL3*033`. So:

```python
# WRONG -- fabricates a gene the sequence cannot distinguish
out.with_columns(gene=pl.col("genes").list.first())

# right -- keep the ambiguity, and note that tied genes' loop windows are usually identical anyway
out.with_columns(label=pl.coalesce(pl.col("gene"), pl.col("genes").list.join("/")))
```

A tie is not a detection failure. `is_kir` is `true`, `confidence` is high, and `tie_kind` says
which mechanism fired: `exact` (byte-identical ectodomains — irreducible) or `score` (log-odds
separation inside the calibrated tolerance).

**`typicality` saturates on structural input and must not be thresholded.** Within a gene, IPD-KIR
alleles differ at a handful of positions, so the training score distribution has a standard
deviation of 1-6 nats on totals of 500-780. Any chain with unmodelled residues sits tens of sigma
low and the normal CDF underflows. Values of 1e-160 are normal for a perfectly good deposited
chain. `confidence` and `gene_confidence` are the fields that carry information there.

**`architecture` reports what is observed, not what the gene implies.** A 167-residue KIR2DL3 chain
segments as `"D1"`, with a note saying the gene call implies `"D1D2"`. That is a correct observation
of an incomplete chain, and the domain models never consult the gene call, so the two are an
independent cross-check.

Also: `region_start`/`region_end` and the domain `start`/`end` index the **cleaned** sequence —
uppercased and stripped of everything that is not one of the 20 amino acids or `X`. If your input
carries gaps or punctuation, they will not line up with the input string. And `score` is not
comparable across architectures: model lengths are 199 (D1D2), 194 (D0D2), 294 (D0D1D2), and
log-odds accumulates per match state.

## Parity

The reference and the plugin are asserted to agree, not assumed to. `tests/test_parity.py` runs both
over every sequence in four committed corpora and fails on any difference in the call or any
difference in `score` beyond zero. Measured by `tools/parity_report.py` — the numbers below come
from an executed run, recorded in full in [`PARITY_REPORT.md`](PARITY_REPORT.md):

| corpus | n | identical calls | max abs `score` delta | max abs `typicality` delta |
| --- | --- | --- | --- | --- |
| IPD-KIR 2.15.0 alleles (full-length precursors) | 1983 | 1983/1983 | 0 | 8.5e-15 |
| deposited structural KIR chains | 54 | 54/54 | 0 | 5.6e-17 |
| non-KIR chains in those same framed files | 122 | 122/122 | 0 | 0 |
| curated hard negatives (LILR/LAIR, other IgSF) | 22 | 22/22 | 0 | 0 |

**2181/2181** sequences give an identical call on every field of `ChainType` — `chain_type`, `gene`,
`genes`, `is_tie`, `tie_kind`, `architecture`, the domain boundaries and the scored region — with
zero disagreements. `score`, the per-gene log-odds, the domain scores, `confidence` and
`gene_confidence` are bit-identical; `typicality` is the sole exception, at ~1e-14, for the reason
below.

Regenerate with `python tools/parity_report.py`; the report records the platform, the polars version
and the model calibration it was measured against, so a stale copy is identifiable.


Scores are **bit-identical**, not merely close, and that took some care. Viterbi is max-plus, so
the model score has no summation to reorder — but the null score does, and the reference computes
it two different ways:

* `ProfileHMM.log_odds_score` sums the background terms with `np.sum`, which is numpy's *blocked
  pairwise* summation, not a left-to-right accumulation.
* `_viterbi_core_batch` (used by the construct scan and the domain sweep) accumulates the null
  sequentially, one residue at a time.

Floating-point addition is not associative, so these differ in the last bits on the same window,
and that difference can decide an argmax when two windows are close. `src/hmm.rs` reproduces both
separately (`log_odds_pairwise` and `log_odds_sequential`), including a transcription of numpy's
`pairwise_sum_DOUBLE`, and each caller uses the one its reference path uses.

The one value that is *not* bit-exact is `typicality`, which needs a normal CDF: scipy's `ndtr` is
a Cephes rational approximation, and `src/classify.rs` computes `erfc` via the incomplete gamma
function instead. Measured worst case 8.5e-15 over 2181 sequences. No decision reads it — `is_kir`,
the gene call, the tie set and the architecture are all settled before it is computed.

## Throughput

**7.4x** on a column of 200 deposited chains (median 197 residues): 0.88 s against 6.43 s for a
Python loop over `type_chain`, best of three with the reference's numba JIT warmed first. At scale
the full 1983-allele corpus — which triggers the sliding-window scan on every 3D-gene precursor —
takes 87 s against 496 s. (These are one machine's numbers, restated from `PARITY_REPORT.md`;
regenerate rather than trust them on different hardware.)

The win is not "compiled beats interpreted": the reference's Viterbi core is already numba-JIT'd
machine code. It is that the plugin runs without the GIL and fans out across cores — 15 gene models
per sequence, and hundreds of window placements per domain segmentation — so one Polars pass keeps
every core busy where the reference walks rows one at a time.

## A measured surprise: deposited chains do not tie

The KIR2DS2/KIR2DL3 ectodomain collision makes a tie look inevitable for the four deposited KIR2DS2
structures, and it does not happen — none of the 54 chains ties. Those chains have 190-196 *observed*
residues against the 199-residue colliding ectodomain, because of disordered loops, so the scored
region is neither byte-equal to it nor a contiguous substring of it and the exact-tie test correctly
does not fire; the per-gene models then separate by more than the tie tolerance.

So the collision is a hazard for **complete** ectodomain or precursor input — 128 of the 1983 alleles
tie exactly, and 221 tie by either mechanism — and not for this structure set. Consuming code still
must not assume a unique gene, but a pipeline over deposited coordinates will in practice see none.
`tests/test_edge_cases.py` pins both halves of this.

## Invariants inherited from histo_kir_hmm

Two of the upstream invariants are load-bearing here and were reproduced deliberately rather than
approximated. Both were measured upstream; see `histo_kir_hmm/CLAUDE.md` invariants 1 and 2.

1. **Emissions use a fixed effective count (`n_effective=20`), not a flat pseudocount.** Per-gene
   allele counts run 8 to 103; under a flat pseudocount a conserved column's probability is
   `(n+1)/(n+20)`, so models trained on more alleles are sharper and win on foreign sequences —
   139 of 540 training sequences went to the wrong gene, every error landing on KIR2DL1 or KIR3DL1.
   This is baked into the trained arrays, which are copied byte-identical; `codegen/vendor_models.py`
   asserts the manifest that records it is present and consistent.
2. **`is_kir` tests log-odds per MATCH STATE**, with denominator `min(len(sequence), L_gene)`. Not
   the total, which is a length threshold in disguise and rejects `8tui` chain A — a real 98-residue
   KIR2DL3 D1 domain. Not per residue either, which dilutes a precursor's score with the ~150
   residues outside the ectodomain and rejected 248 of 433 KIR2DL1 alleles. A Viterbi path visits at
   most `L` match states, so `L` is the only scale-free denominator.

The trained models come from IPD-KIR release 2.15.0. Eight of the 15 genes have **no deposited
structure anywhere** (KIR2DL5A, KIR2DL5B, KIR2DS1, KIR2DS3, KIR2DS5, KIR3DL2, KIR3DL3, KIR3DS1);
a positive call on one of them carries a note saying its loop windows are alignment predictions,
and `genes()` reports it as `has_structure=False`. Keep that distinction visible downstream.

## Layout

```
Cargo.toml                       cdylib named _internal, built by maturin
src/npy.rs                       minimal .npy/.npz reader (no pickle in the model format)
src/models.rs                    model loading, transposed layout, process-wide cache
src/hmm.rs                       Viterbi core + both null-score summations
src/classify.rs                  port of KIRClassifier: assign_gene, call_architecture, type_chain
src/expressions.rs               the three polars expressions
python/polars_kir_hmm/           namespace registration, validation, models/
codegen/vendor_models.py         copies + validates the models from histo_kir_hmm
tools/make_test_data.py          builds the committed corpora from IPD-KIR + framed structures
tools/make_golden.py             records reference answers into tests/data/golden.json
tools/parity_report.py           regenerates PARITY_REPORT.md
tests/test_parity.py             equality against the live reference, all four corpora
tests/test_golden.py             recorded answers, for interpreters without the reference
tests/test_polars.py             schema, nulls, laziness, the tie contract
tests/test_edge_cases.py         the invariants, each test named after a real failure mode
CLAUDE.md                        working notes and the invariants that must not regress
PARITY_REPORT.md                 the measured parity result
```

`cargo test` also runs 8 Rust unit tests covering the npy header parser, the pairwise summation,
sequence cleaning and the normal CDF against known scipy values.

## Licence

MIT.
