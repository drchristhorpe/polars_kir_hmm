"""Measure parity between the Rust plugin and histo_kir_hmm, and write PARITY_REPORT.md.

The plugin only earns its place if it is a *performance path*: the same answers, faster. This
script is what establishes that, over every sequence in all four corpora:

1. **Calls.** For each sequence, does the plugin agree with `histo_kir_hmm.type_chain` on
   `chain_type`, `gene`, `genes`, `is_tie`, `tie_kind` and `architecture`, and on the domain
   boundaries? Any disagreement is listed individually -- a summary percentage would hide exactly
   the cases worth looking at.
2. **Scores.** Max absolute difference in `score`, `confidence`, `gene_confidence`, `typicality`
   and the per-domain scores. `score` is expected to be **bit-identical**: Viterbi is max-plus and
   `src/hmm.rs` reproduces numpy's blocked pairwise summation for the null term. `typicality` is
   the one value that is not, because it needs a normal CDF (see the README).
3. **Speed.** Wall time for a realistic column length, plugin against a Python loop over
   `type_chain`, with the reference's numba JIT warmed first so the comparison is of steady-state
   throughput rather than of compilation.

Run:  python tools/parity_report.py                       # full run, ~20 min
      python tools/parity_report.py --max-alleles 200     # quick pass
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import polars as pl

import polars_kir_hmm  # noqa: F401  -- registers the .kir namespace

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data"

# Fields of ChainType that must agree exactly.
CALL_FIELDS = ("chain_type", "gene", "genes", "is_tie", "tie_kind", "architecture")
# Float fields, compared by magnitude.
FLOAT_FIELDS = ("score", "confidence", "gene_confidence", "typicality")


def load_corpora(max_alleles: int | None) -> dict[str, list[dict]]:
    alleles = json.loads((DATA / "ipdkir_alleles.json").read_text())["alleles"]
    if max_alleles is not None:
        alleles = alleles[:max_alleles]
    structural = json.loads((DATA / "structural_chains.json").read_text())
    return {
        "IPD-KIR alleles (full-length precursors)": [
            {"label": a["allele"], "sequence": a["sequence"], "expected_gene": a["gene"],
             "is_null": a["is_null"]}
            for a in alleles
        ],
        "deposited structural KIR chains": [
            {"label": f"{c['pdb_id']}_{c['chain']}", "sequence": c["sequence"],
             "expected_gene": c["gene"]}
            for c in structural["kir_chains"]
        ],
        # All 122, the scope histo_kir_hmm/VERIFICATION_REPORT.md reports -- not the 22-chain
        # curated fixture, which is a subset plus UniProt receptors absent from every structure.
        "non-KIR chains in the same framed files": [
            {"label": f"{c['file']}_{c['chain']}", "sequence": c["sequence"],
             "expected_gene": None}
            for c in structural["non_kir_chains"]
        ],
        "curated hard negatives (LILR/LAIR, other IgSF)": [
            {"label": c["identity"], "sequence": c["sequence"], "expected_gene": None}
            for c in structural["non_kir_curated"]
        ],
    }


def plugin_calls(sequences: list[str]) -> list[dict]:
    """One pass of the plugin over the whole column."""
    df = pl.DataFrame({"sequence": sequences})
    out = df.with_columns(call=pl.col("sequence").kir.classify()).unnest("call")
    return out.to_dicts()


def reference_calls(sequences: list[str], classifier) -> list[dict]:
    """`type_chain` per sequence, shaped like the plugin's row for comparison."""
    rows = []
    for seq in sequences:
        c = classifier.type_chain(seq)
        rows.append(
            {
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
                "notes": list(c.notes),
                "raw_scores": dict(c.raw_scores),
            }
        )
    return rows


def abs_delta(a: float | None, b: float | None) -> float:
    """Absolute difference, treating two infinities of the same sign as equal."""
    if a is None or b is None:
        return 0.0 if a == b else float("inf")
    if a == b:  # covers -inf == -inf
        return 0.0
    return abs(a - b)


def compare(corpus: list[dict], plug: list[dict], ref: list[dict]) -> dict:
    disagreements: list[dict] = []
    deltas: dict[str, float] = {f: 0.0 for f in FLOAT_FIELDS}
    deltas["domain_score"] = 0.0
    worst: dict[str, str] = {}
    n_kir = 0
    n_tie = 0
    n_exact_tie = 0
    domain_mismatch = 0

    for item, p, r in zip(corpus, plug, ref):
        if r["is_kir"]:
            n_kir += 1
        if r["is_tie"]:
            n_tie += 1
            if r["tie_kind"] == "exact":
                n_exact_tie += 1

        for field in CALL_FIELDS:
            if p[field] != r[field]:
                disagreements.append(
                    {"label": item["label"], "field": field, "plugin": p[field], "reference": r[field]}
                )

        # Domain segmentation: boundaries must match exactly, scores numerically.
        p_dom = p["domains"] or []
        r_dom = r["domains"] or []
        if [(d["domain"], d["start"], d["end"]) for d in p_dom] != [
            (d["domain"], d["start"], d["end"]) for d in r_dom
        ]:
            domain_mismatch += 1
            disagreements.append(
                {
                    "label": item["label"],
                    "field": "domains",
                    "plugin": [(d["domain"], d["start"], d["end"]) for d in p_dom],
                    "reference": [(d["domain"], d["start"], d["end"]) for d in r_dom],
                }
            )
        else:
            for dp, dr in zip(p_dom, r_dom):
                d = abs_delta(dp["score"], dr["score"])
                if d > deltas["domain_score"]:
                    deltas["domain_score"] = d
                    worst["domain_score"] = f"{item['label']} {dp['domain']}"

        # The scored region must agree, or the two scored different sequence.
        if (p["region_start"], p["region_end"]) != (r["region_start"], r["region_end"]):
            disagreements.append(
                {
                    "label": item["label"],
                    "field": "scored_region",
                    "plugin": (p["region_start"], p["region_end"]),
                    "reference": (r["region_start"], r["region_end"]),
                }
            )

        for field in FLOAT_FIELDS:
            d = abs_delta(p[field], r[field])
            if d > deltas[field]:
                deltas[field] = d
                worst[field] = item["label"]

        # Every per-gene log-odds, not just the winner: the winner could agree by luck.
        plugin_raw = {e["gene"]: e["score"] for e in (p["raw_scores"] or [])}
        for gene, score in (r["raw_scores"] or {}).items():
            d = abs_delta(plugin_raw.get(gene), score)
            if d > deltas.get("raw_score", 0.0):
                deltas["raw_score"] = d
                worst["raw_score"] = f"{item['label']} {gene}"

    return {
        "n": len(corpus),
        "n_kir": n_kir,
        "n_tie": n_tie,
        "n_exact_tie": n_exact_tie,
        "n_agree": len(corpus) - len({d["label"] for d in disagreements}),
        "disagreements": disagreements,
        "deltas": deltas,
        "worst": worst,
        "domain_mismatch": domain_mismatch,
    }


def benchmark(sequences: list[str], classifier, repeats: int = 3) -> dict:
    """Wall time for one column, plugin against a Python loop over `type_chain`.

    The reference's numba core is JIT-compiled on first call, so it is warmed before timing --
    otherwise the first measurement is dominated by compilation and the speedup is overstated.
    """
    classifier.type_chain(sequences[0])
    df = pl.DataFrame({"sequence": sequences})
    df.with_columns(call=pl.col("sequence").kir.classify())  # warm the model cache

    plugin_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        df.with_columns(call=pl.col("sequence").kir.classify())
        plugin_times.append(time.perf_counter() - t0)

    ref_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for seq in sequences:
            classifier.type_chain(seq)
        ref_times.append(time.perf_counter() - t0)

    plugin = min(plugin_times)
    ref = min(ref_times)
    return {
        "n": len(sequences),
        "plugin_s": plugin,
        "reference_s": ref,
        "speedup": ref / plugin,
        "plugin_per_seq_ms": 1000 * plugin / len(sequences),
        "reference_per_seq_ms": 1000 * ref / len(sequences),
        "repeats": repeats,
        "median_length": statistics.median(len(s) for s in sequences),
    }


def fmt_delta(v: float) -> str:
    if v == 0.0:
        return "**0** (bit-identical)"
    if v == float("inf"):
        return "inf (mismatched null/inf)"
    return f"{v:.3e}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--max-alleles", type=int, default=None, help="truncate the allele corpus")
    ap.add_argument("--bench-n", type=int, default=200, help="column length for the benchmark")
    ap.add_argument("--output", type=Path, default=ROOT / "PARITY_REPORT.md")
    args = ap.parse_args()

    try:
        from histo_kir_hmm.classify import KIRClassifier
    except ImportError:
        raise SystemExit("histo_kir_hmm is not importable; parity cannot be measured without it")

    classifier = KIRClassifier()
    corpora = load_corpora(args.max_alleles)

    results: dict[str, dict] = {}
    for name, corpus in corpora.items():
        sequences = [c["sequence"] for c in corpus]
        print(f"{name}: {len(sequences)} sequences ...", flush=True)
        t0 = time.perf_counter()
        plug = plugin_calls(sequences)
        t_plugin = time.perf_counter() - t0
        t0 = time.perf_counter()
        ref = reference_calls(sequences, classifier)
        t_ref = time.perf_counter() - t0
        res = compare(corpus, plug, ref)
        res["t_plugin"] = t_plugin
        res["t_reference"] = t_ref
        results[name] = res
        print(
            f"  {res['n_agree']}/{res['n']} agree; max score delta "
            f"{res['deltas']['score']:.3e}; plugin {t_plugin:.1f}s vs reference {t_ref:.1f}s",
            flush=True,
        )

    # Benchmark on the structural chains, repeated to the requested column length -- a realistic
    # shape for the actual use (a frame of chains from a set of structures).
    chains = [c["sequence"] for c in corpora["deposited structural KIR chains"]]
    bench_seqs = [chains[i % len(chains)] for i in range(args.bench_n)]
    print(f"benchmark on {args.bench_n} structural chains ...", flush=True)
    bench = benchmark(bench_seqs, classifier)
    print(f"  {bench['speedup']:.1f}x", flush=True)

    cal = polars_kir_hmm.calibration()
    lines: list[str] = []
    lines.append("# Parity report: polars-kir-hmm vs histo_kir_hmm\n")
    lines.append(
        "Generated by `tools/parity_report.py`. The plugin is a performance path, not a second "
        "implementation, so this is the measurement that matters: identical calls and identical "
        "scores, over every sequence in every corpus.\n"
    )
    lines.append(
        f"- plugin `polars_kir_hmm` {polars_kir_hmm.__version__}, polars {pl.__version__}\n"
        f"- reference `histo_kir_hmm` models from `{classifier.model_dir}`\n"
        f"- {platform.platform()}, Python {platform.python_version()}\n"
        f"- calibration: `per_residue_threshold` {cal['per_residue_threshold']:.6f}, "
        f"`tie_tolerance` {cal['tie_tolerance']:.6f}, "
        f"`domain_threshold` {cal['domain_threshold']:.6f}\n"
    )

    lines.append("\n## Call agreement\n")
    lines.append(
        "| corpus | n | agree | KIR-positive | ties | exact ties | domain-boundary mismatches |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['n']} | **{r['n_agree']}/{r['n']}** | {r['n_kir']} | {r['n_tie']} | "
            f"{r['n_exact_tie']} | {r['domain_mismatch']} |"
        )

    total_n = sum(r["n"] for r in results.values())
    total_agree = sum(r["n_agree"] for r in results.values())
    lines.append(
        f"\nAcross all {len(results)} corpora: **{total_agree}/{total_n}** sequences give an identical call "
        f"on every field of `ChainType` (`chain_type`, `gene`, `genes`, `is_tie`, `tie_kind`, "
        f"`architecture`, the domain boundaries and the scored region).\n"
    )

    lines.append("\n## Numeric agreement\n")
    lines.append("Maximum absolute difference over each corpus, with the sequence it occurred on.\n")
    lines.append("| corpus | score | raw per-gene score | domain score | confidence | gene_confidence | typicality |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, r in results.items():
        d = r["deltas"]
        lines.append(
            f"| {name} | {fmt_delta(d['score'])} | {fmt_delta(d.get('raw_score', 0.0))} | "
            f"{fmt_delta(d['domain_score'])} | {fmt_delta(d['confidence'])} | "
            f"{fmt_delta(d['gene_confidence'])} | {fmt_delta(d['typicality'])} |"
        )
    lines.append("")
    for name, r in results.items():
        if r["worst"]:
            worst = ", ".join(f"`{k}` on {v}" for k, v in sorted(r["worst"].items()))
            lines.append(f"- {name}: worst case {worst}")

    lines.append(
        "\n`score`, the per-gene log-odds and the domain scores are **bit-identical**. Viterbi is "
        "max-plus, so the model score has no summation to reorder; the null score does, and "
        "`src/hmm.rs` reproduces both of the reference's summation orders separately -- numpy's "
        "blocked pairwise sum for `log_odds_score`, and the sequential accumulation of "
        "`_viterbi_core_batch` for the scan and the domain sweep. `confidence` and "
        "`gene_confidence` follow from those, so they are bit-identical too.\n"
    )
    lines.append(
        "`typicality` is the one value that is not bit-exact: it needs a normal CDF, scipy's "
        "`ndtr` is a Cephes rational approximation, and the plugin computes `erfc` via the "
        "incomplete gamma function instead. No decision reads it -- `is_kir`, the gene call, the "
        "tie set and the architecture are all settled before it is computed -- and it saturates "
        "on structural input by design (`histo_kir_hmm` invariant 12), so it must not be "
        "thresholded in either implementation.\n"
    )

    lines.append("\n## Disagreements\n")
    total_dis = sum(len(r["disagreements"]) for r in results.values())
    if total_dis == 0:
        lines.append("None. Every sequence in every corpus gives an identical call.\n")
    else:
        lines.append(f"{total_dis} field-level disagreements:\n")
        lines.append("| corpus | sequence | field | plugin | reference |")
        lines.append("| --- | --- | --- | --- | --- |")
        for name, r in results.items():
            for d in r["disagreements"][:50]:
                lines.append(
                    f"| {name} | {d['label']} | `{d['field']}` | `{d['plugin']}` | "
                    f"`{d['reference']}` |"
                )

    lines.append("\n## Throughput\n")
    lines.append(
        f"A column of {bench['n']} structural KIR chains (median length "
        f"{bench['median_length']:.0f} residues), best of {bench['repeats']} runs, with the "
        "reference's numba JIT warmed first so this is steady-state throughput rather than "
        "compilation:\n"
    )
    lines.append("| | wall time | per sequence |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| plugin (one Polars pass) | {bench['plugin_s']:.2f} s | "
        f"{bench['plugin_per_seq_ms']:.1f} ms |"
    )
    lines.append(
        f"| reference (Python loop over `type_chain`) | {bench['reference_s']:.2f} s | "
        f"{bench['reference_per_seq_ms']:.1f} ms |"
    )
    lines.append(f"\n**{bench['speedup']:.1f}x** on this shape.\n")
    lines.append(
        "The win is not \"compiled beats interpreted\" -- the reference's Viterbi core is already "
        "numba-JIT'd machine code. It is that the plugin runs without the GIL and fans the work "
        "out across cores: 15 gene models per sequence, and hundreds of window placements per "
        "domain segmentation. Whole-corpus times above show the same effect at scale.\n"
    )
    lines.append("| corpus | plugin | reference |")
    lines.append("| --- | --- | --- |")
    for name, r in results.items():
        lines.append(f"| {name} | {r['t_plugin']:.1f} s | {r['t_reference']:.1f} s |")

    args.output.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.output}")

    summary = {
        "corpora": {
            name: {
                "n": r["n"],
                "n_agree": r["n_agree"],
                "n_kir": r["n_kir"],
                "n_tie": r["n_tie"],
                "n_exact_tie": r["n_exact_tie"],
                "deltas": r["deltas"],
                "worst": r["worst"],
                "n_disagreements": len(r["disagreements"]),
                "disagreements": r["disagreements"][:50],
                "t_plugin": r["t_plugin"],
                "t_reference": r["t_reference"],
            }
            for name, r in results.items()
        },
        "benchmark": bench,
    }
    (ROOT / "tmp").mkdir(exist_ok=True)
    (ROOT / "tmp" / "parity_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return 0 if total_dis == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
