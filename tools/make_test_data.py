"""Build the committed test corpora from the real sources.

Three files under `tests/data/`, all real sequences -- never mocks or synthetic strings, which is
a project-wide rule:

`ipdkir_alleles.json`
    All 1983 IPD-KIR 2.15.0 protein alleles as full-length precursor translations, with the
    expected gene from the allele name and a `is_null` flag for the trailing-`N` non-expressed
    alleles. This is the corpus `histo_kir_hmm/scripts/verify_classifier.py` check 1 uses, and it
    exercises the construct scan: a 348-residue precursor is longer than the 360-residue scan
    trigger only for the 3D genes, so the set covers both the whole-sequence and the windowed path.

`structural_chains.json`
    The 54 observed KIR chain sequences from `validation/structures/framed`, copied from
    `histo_kir_hmm/tests/fixtures/` rather than re-derived so both packages test literally the same
    bytes -- disordered loops, internal numbering breaks and truncated constructs included, which is
    the point.

    Two non-KIR sets, because they have different scopes and conflating them was a real error:

    `non_kir_chains`
        **All 122** non-KIR chains in those same 54 framed files, read from the coordinates here.
        That is the number `histo_kir_hmm/VERIFICATION_REPORT.md` reports (122/122 rejected), and it
        is what the parity suite should cover -- 32 MHC heavy chains, 32 b2m, 32 peptides and 26
        other partners, of which only 60 sequences are distinct.
    `non_kir_curated`
        The 22-chain fixture `histo_kir_hmm/tests/fixtures/non_kir_chains.json` holds: one
        representative of each structural partner class **plus** the UniProt LILR/LAIR and
        Ig-superfamily receptors, which are the hardest negatives and are deliberately *not* in any
        structure file. Kept as well as, not instead of, the 122.

`golden.json`
    Reference outputs for a small subset, recorded from `histo_kir_hmm` by `tools/make_golden.py`.
    Lets the plugin be tested on a machine without numba.

Run:  python tools/make_test_data.py --kirs-root ..
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "tests" / "data"


def parse_fasta(path: Path) -> list[dict]:
    """Parse IPD-KIR's `kir_prot.fasta`.

    Headers are `>KIR:KIR00001 KIR2DL1*0010101 348 bp` -- **description field 2 is the allele
    name**, and the gene is the part before `*`. Getting this wrong (taking field 1, the IPD
    accession) would silently give every allele the same "gene".
    """
    records: list[dict] = []
    ipd_id = allele = None
    chunks: list[str] = []

    def flush() -> None:
        if allele is None:
            return
        seq = "".join(chunks).upper().replace("-", "").replace(".", "").replace("*", "")
        if not seq:
            return
        records.append(
            {
                "ipd_id": ipd_id,
                "allele": allele,
                "gene": allele.split("*")[0],
                "sequence": seq,
                # IPD-KIR marks non-expressed alleles with a trailing N; their translations are
                # frameshifted fragments, so rejection is correct behaviour and accuracy must be
                # reported separately (histo_kir_hmm invariant 9).
                "is_null": allele.endswith("N"),
            }
        )

    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                fields = line[1:].split()
                ipd_id = fields[0] if fields else None
                allele = fields[1] if len(fields) > 1 else None
                chunks = []
            else:
                chunks.append(line.strip())
    flush()
    return records


def read_framed_non_kir(framed_dir: Path) -> list[dict]:
    """Every non-KIR chain of every framed file, in coordinate order.

    The framing step remapped chain letters so K is the KIR chain, M the MHC heavy chain, B
    beta-2 microglobulin, P the peptide and X/Y/Z/W the remaining partners. Those letters are used
    only to know which chain is the KIR one to exclude; the classifier is never told them, and it
    must reject all the rest on sequence alone.
    """
    from Bio.Data.IUPACData import protein_letters_3to1
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa

    three_to_one = {k.upper(): v for k, v in protein_letters_3to1.items()}
    expected = {
        "M": "MHC class I heavy chain",
        "B": "beta-2 microglobulin",
        "P": "peptide",
        "X": "other partner (antibody chain / RIFIN)",
        "Y": "other partner (antibody chain / RIFIN)",
        "Z": "other partner (antibody chain / RIFIN)",
        "W": "other partner (antibody chain / RIFIN)",
    }
    parser = PDBParser(QUIET=True)
    out: list[dict] = []
    for path in sorted(framed_dir.glob("*.pdb")):
        structure = parser.get_structure(path.stem, str(path))
        for chain in next(structure.get_models()):
            letters = [
                three_to_one.get(res.get_resname().upper(), "X")
                for res in chain
                if is_aa(res, standard=False)
            ]
            if not letters or chain.id == "K":
                continue
            out.append(
                {
                    "file": path.name,
                    "pdb_id": path.stem.split("_")[0],
                    "chain": chain.id,
                    "identity": expected.get(chain.id, "unassigned partner"),
                    "n_res": len(letters),
                    "sequence": "".join(letters),
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--kirs-root",
        type=Path,
        default=ROOT.parent,
        help="the kirs/ directory holding validation/ and histo_kir_hmm/ (default: ..)",
    )
    args = ap.parse_args()

    kirs = args.kirs_root.resolve()
    fasta = kirs / "validation" / "ref" / "kir_prot.fasta"
    fixtures = kirs / "histo_kir_hmm" / "tests" / "fixtures"

    for path in (fasta, fixtures / "structural_chains.json", fixtures / "non_kir_chains.json"):
        if not path.exists():
            raise SystemExit(f"missing source: {path}")

    DEST.mkdir(parents=True, exist_ok=True)

    alleles = parse_fasta(fasta)
    genes = sorted({a["gene"] for a in alleles})
    if len(genes) != 15:
        raise SystemExit(f"expected 15 KIR genes in the FASTA, found {len(genes)}: {genes}")
    (DEST / "ipdkir_alleles.json").write_text(
        json.dumps(
            {
                "_provenance": {
                    "source": "IPD-KIR release 2.15.0, kir_prot.fasta",
                    "url": "https://raw.githubusercontent.com/ANHIG/IPDKIR/Latest/kir_prot.fasta",
                    "local_copy": str(fasta),
                    "note": (
                        "Full-length precursor translations. Description field 2 is the allele "
                        "name; the gene is the part before '*'. is_null flags the trailing-N "
                        "non-expressed alleles, whose translations are frameshifted fragments."
                    ),
                    "n_alleles": len(alleles),
                    "n_genes": len(genes),
                },
                "alleles": alleles,
            },
            indent=1,
        )
        + "\n"
    )

    # KIR chains and tied ectodomains: copied, not re-derived, so both packages test the same bytes.
    structural = json.loads((fixtures / "structural_chains.json").read_text())
    curated = json.loads((fixtures / "non_kir_chains.json").read_text())

    framed_dir = kirs / "validation" / "structures" / "framed"
    if not framed_dir.is_dir():
        raise SystemExit(f"missing framed structures: {framed_dir}")
    non_kir = read_framed_non_kir(framed_dir)
    if len(non_kir) != 122:
        raise SystemExit(
            f"expected 122 non-KIR chains across the framed files (the count "
            f"histo_kir_hmm/VERIFICATION_REPORT.md reports), found {len(non_kir)}"
        )

    combined = {
        "_provenance": {
            "kir_chains": structural["_provenance"],
            "non_kir_chains": {
                "source": f"{framed_dir}/*.pdb, every chain other than the framed KIR chain K",
                "note": (
                    "All 122 non-KIR chains across the 54 framed files -- the scope "
                    "histo_kir_hmm/VERIFICATION_REPORT.md reports as 122/122 rejected. Only 60 of "
                    "the sequences are distinct, because the same MHC heavy chain, b2m and peptide "
                    "recur across entries; the duplicates are kept so the count matches the "
                    "reference's."
                ),
            },
            "non_kir_curated": curated["_provenance"],
            "note": (
                "KIR chains and tied ectodomains copied verbatim from "
                "histo_kir_hmm/tests/fixtures/. The non-KIR chains are read from the same framed "
                "coordinates rather than from the 22-chain fixture, which is a curated subset plus "
                "UniProt receptors that appear in no structure."
            ),
        },
        "kir_chains": structural["kir_chains"],
        "tied_ectodomains": structural["tied_ectodomains"],
        "non_kir_chains": non_kir,
        "non_kir_curated": curated["chains"],
    }
    (DEST / "structural_chains.json").write_text(json.dumps(combined, indent=1) + "\n")

    print(f"{len(alleles)} alleles across {len(genes)} genes -> tests/data/ipdkir_alleles.json")
    n_null = sum(1 for a in alleles if a["is_null"])
    print(f"  {n_null} null (trailing-N) alleles, {len(alleles) - n_null} expressed")
    n_unique = len({c["sequence"] for c in combined["non_kir_chains"]})
    print(
        f"{len(combined['kir_chains'])} KIR chains + {len(combined['non_kir_chains'])} non-KIR "
        f"chains ({n_unique} distinct sequences) + {len(combined['non_kir_curated'])} curated "
        f"negatives + {len(combined['tied_ectodomains'])} tied ectodomains -> "
        "tests/data/structural_chains.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
