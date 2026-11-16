# polars-kir-hmm — working notes

## What this is

A Polars expression plugin wrapping `histo_kir_hmm`'s 18 trained profile HMMs (15 gene + D0/D1/D2),
so a whole column of sequences is chain-typed in one pass without the GIL. Three expressions in a
`kir` namespace — `classify`, `score`, `domains` — plus function forms and three introspection
helpers (`genes()`, `models()`, `calibration()`).

Region slicing (`chunk`) and loop slicing (`loops`) are NOT here: they live in the standalone
`polars_kir_regions`, which takes the gene from the caller's dataframe and therefore needs no
model. Keep it that way — the split exists so a caller who only wants boundaries does not have to
install an HMM.

The HLA-contacting **loops** are NOT here. They live in the standalone `polars_kir_regions`
package, whose expressions take the gene from the caller's dataframe and therefore need no HMM at
all. Nothing in this package depends on it, or the reverse.

It exists to be a **performance path, not a second implementation**. Everything in `src/` is a
transcription of the reference, and `PARITY_REPORT.md` is the evidence: 2181/2181 identical calls
with bit-identical scores. If a change makes the plugin *better* than the reference, it is still a
bug — fix it upstream in `histo_kir_hmm` and re-vendor.

Sibling of `polars_mhci_hmm` (same author, MHC class I loci); same layout and ergonomics so the two
feel alike. House conventions: `histo_tools/CONVENTIONS.md`.

## Environment

```bash
uv sync
uv run maturin develop --release
uv run pytest                      # 58 tests; -m "not slow" for the 56 fast ones
uv run python tools/parity_report.py
```

The Rust toolchain is required. Two build notes that cost time to rediscover:

* **Build through maturin, never bare `cargo build`.** A pyo3 `cdylib` on macOS needs
  `-undefined dynamic_lookup`, which maturin passes and cargo does not; a bare build fails at link
  time with a wall of undefined `_Py*` symbols that looks like a missing Python but is not.
* **`polars-arrow` must have `default-features = false`.** We use only `bitmap::Bitmap`; the default
  feature set pulls the IPC compression codecs (lz4, zstd). Beyond being dead weight, those two
  crates' published archives contain a `.gitmodules`, and a sandbox that refuses writes to `.git*`
  paths cannot unpack them — the failure is an opaque `Operation not permitted (os error 1)` during
  `cargo fetch`, nothing to do with the network.

## Key invariants — don't break these

**1. Parity is the contract, and it is asserted at equality.** `tests/test_parity.py` compares
against the live reference over all four corpora and asserts `score`, the per-gene log-odds and the
domain scores are *exactly* equal — no tolerance. If you find yourself adding `rel=` to make a test
pass, stop: that is a real divergence.

**2. There are TWO null-score summations and they are not interchangeable.** This is the subtlety
the whole bit-exactness result rests on:

* `ProfileHMM.log_odds_score` does `np.sum(background[known])` — numpy's *blocked pairwise*
  summation, an eight-accumulator tree below 128 elements and a recursive split above — and only
  then adds the unknown-residue term.
* `_viterbi_core_batch` accumulates the null **sequentially**, one residue at a time, taking the
  uniform term inline.

Floating-point addition is not associative, so these differ in the last bits on the same window, and
that difference can flip which window a scan picks. `src/hmm.rs` has `log_odds_pairwise` and
`log_odds_sequential`; each caller must use the one its reference path uses — whole-sequence gene
scoring and the `call_architecture` rival scores use pairwise, the construct scan and the
domain-placement sweep use sequential. `pairwise_sum` there is a transcription of numpy's
`pairwise_sum_DOUBLE`, verified bit-identical to `np.sum` over 3440 realistic vectors before any of
the Rust was written. Do not "simplify" it to a fold.

**3. Manifest order is load-bearing.** `gene_models` and `domain_models` are the orders the
reference's dicts iterate in, which is what Python's `max()` and stable `sorted()` fall back on when
two models score identically. `codegen/vendor_models.py` asserts they are sorted; nothing re-sorts.

**4. Ties reach the output as ties.** `gene` null, `genes` populated, `is_tie` true. Never pick a
winner, and never treat a tie as a rejection. 128 of the 1983 IPD-KIR alleles tie exactly.

**5. Calibration is read, never defaulted.** `per_residue_threshold` absent from the manifest is a
hard load error, not a case to paper over with 0.0 — a threshold invented here would be a guess
presented as a calibration. The other three fall back only because `KIRClassifier.__init__` does.

**6. `is_kir` divides by `min(n_scored, L_gene)`.** Upstream invariant 2. Neither the total (a length
threshold in disguise — rejects `8tui_A`, a real 98-residue D1 domain) nor the raw length (dilutes a
precursor — rejected 248 of 433 KIR2DL1 alleles). `test_edge_cases.py` guards both directions.

**7. `typicality` is the one non-bit-exact field, and no decision may read it.** It needs a normal
CDF; scipy's `ndtr` is Cephes, and `src/classify.rs` uses the incomplete gamma function instead
(measured worst case 8.5e-15). It saturates towards zero on structural input by design, so never
threshold it and never "fix" the saturation.

**8. Fixtures are real committed data.** Never mocks, never synthetic sequences — project-wide rule.
The sequences that break a KIR classifier are real deposited chains with disordered loops and real
alleles whose ectodomains collide.

## The non-KIR corpus is 122 chains, not 22

Worth stating because getting it wrong is easy and was got wrong once here. Two different sets:

* **122** — every non-KIR chain across the 54 framed files (32 MHC heavy, 32 b2m, 32 peptide, 26
  other partners; only 60 distinct sequences, duplicates kept so the count matches). This is the
  scope `histo_kir_hmm/VERIFICATION_REPORT.md` reports as 122/122 rejected, and what
  `tests/data/structural_chains.json` → `non_kir_chains` holds.
* **22** — `histo_kir_hmm/tests/fixtures/non_kir_chains.json`: one representative per structural
  partner class **plus** the UniProt LILR/LAIR and IgSF receptors, which appear in no structure file.
  Kept as `non_kir_curated`, as well as rather than instead of the 122, because LILRB1 is the
  load-bearing negative and is only reachable there.

## A measured surprise worth not re-deriving

None of the 54 deposited chains ties, despite four of them being KIR2DS2 whose reference-allele
ectodomain is byte-identical to `KIR2DL3*033`. Those chains have 190-196 observed residues against
the 199-residue ectodomain, so the scored region is neither byte-equal to it nor a contiguous
substring, and the containment test correctly does not fire. The collision is a hazard for complete
ectodomain/precursor input, not for this structure set. Both halves are pinned in
`tests/test_edge_cases.py`; don't "fix" the zero-tie result.

## Re-vendoring after retraining

```bash
uv run python codegen/vendor_models.py --histo-kir-hmm ../histo_kir_hmm
uv run python tools/make_test_data.py --kirs-root ..
uv run python tools/make_golden.py          # golden.json is keyed to the model checksums
uv run python tools/parity_report.py
```

`golden.json` records the model checksums it was taken against and `test_golden.py` compares them, so
a stale golden file fails loudly rather than passing against the wrong models.

## Scope

Three expressions, and that is deliberate. `classify` covers `type_chain`; `score` covers
`log_odds_score` (and raw Viterbi via `log_odds=False`); `domains` covers `call_architecture`.
The reference's `is_kir` is not exposed — it returns exactly what `classify` already gives as
`is_kir`/`confidence`, so a further expression would be a second way to ask one question. Don't
add one without checking with the user.

**Region slicing and loop slicing are not here.** `chunk` and `loops` live in the standalone
`polars_kir_regions`, which takes the gene from the caller's dataframe and so needs no model at
all. Both packages were briefly developed inside this one and then moved out; if you find
`src/chunk.rs`, `src/regions.rs`, `src/loops*.rs` or `src/align.rs` here again, something has
been put back in the wrong package. The split is the point: a caller who wants exon boundaries
should not have to install 18 profile HMMs to get them, and this package's parity surface stays
exactly the three reference methods above.

The two packages share no code at runtime. `polars_kir_regions` carries its own aligner and its
own `sequence.rs` copy of `clean_sequence` precisely so that neither imports the other; they share
only a build-time data source, `histo_kir`'s Phase 0 outputs.
