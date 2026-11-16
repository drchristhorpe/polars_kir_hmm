"""polars-kir-hmm -- type KIR chains inside Polars.

Importing this module registers a ``.kir`` namespace on Polars expressions::

    import polars as pl
    import polars_kir_hmm  # noqa: F401  (import for the side effect)

    df.with_columns(call=pl.col("sequence").kir.classify())

The semantics are those of ``histo_kir_hmm``'s ``KIRClassifier``, reproduced field-for-field
against the same 18 trained profile HMMs (15 gene + D0/D1/D2) -- including its sequence cleaning,
its per-match-state ``is_kir`` test, its two tie mechanisms, its sliding-window scan for
constructs, and its domain-segmentation DP.

**Ties are ties.** ``gene`` is null and ``genes`` holds both names whenever the call is tied. 127
IPD-KIR alleles have a mature ectodomain byte-identical to an allele of a different gene --
KIR2DL5A/KIR2DL5B (71 alleles), KIR2DL3/KIR2DS2 (54), KIR2DL2/KIR2DL3 (2) -- and one of them,
``KIR2DS2*0010101``, is a reference allele appearing in four deposited structures. Nothing in this
package picks a winner, and code consuming it must not either: never ``genes[0]``, never treat a
tie as a detection failure. Filter with ``pl.col("call").struct.field("is_kir")``, and read
``genes`` when ``gene`` is null.

See the README for the parity measurements and for what ``region_start``/``region_end`` index.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars._typing import IntoExpr

__all__ = [
    "KIRNameSpace",
    "classify",
    "domains",
    "genes",
    "model_dir",
    "models",
    "score",
    "__version__",
]
__version__ = "0.1.0"

_PLUGIN_PATH = Path(__file__).parent
_MODEL_DIR = _PLUGIN_PATH / "models"


def model_dir() -> Path:
    """Path to the bundled models, vendored from ``histo_kir_hmm``.

    Pass it to ``classify(model_dir=...)`` explicitly, or point that argument at your own
    directory of ``.npz`` models -- the format is ``histo_kir_hmm``'s, unchanged.
    """
    return _MODEL_DIR


@lru_cache(maxsize=None)
def _manifest(directory: str) -> dict:
    path = Path(directory) / "manifest.json"
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(
            f"no manifest.json in {directory!r}; a model directory holds one .npz per model, "
            "a background.npy and a manifest.json (as written by histo_kir_hmm's save_models)"
        ) from None
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed manifest.json in {directory!r}: {e}") from None


def _resolve_model_dir(directory: str | Path | None) -> str:
    """Resolve to a concrete directory, failing here rather than inside the query engine."""
    if directory is None:
        if not (_MODEL_DIR / "manifest.json").exists():
            raise RuntimeError(
                f"the bundled models are missing from {_MODEL_DIR}. If you are working from a "
                "source checkout, run:\n  python codegen/vendor_models.py"
            )
        return str(_MODEL_DIR)

    path = Path(directory)
    if not path.is_dir():
        raise ValueError(f"model_dir {str(directory)!r} is not a directory")
    _manifest(str(path))  # surfaces a missing or malformed manifest now
    return str(path)


def genes(model_dir: str | Path | None = None) -> pl.DataFrame:
    """The 15 KIR genes this build can assign, as a DataFrame.

    Columns: ``gene``, ``architecture`` (``D1D2``/``D0D2``/``D0D1D2``), ``length`` (the profile
    HMM's match-state count), ``n_training_alleles``, and ``has_structure`` -- False for the eight
    genes with no deposited structure anywhere, whose loop windows downstream are alignment
    predictions rather than observations.
    """
    directory = _resolve_model_dir(model_dir)
    manifest = _manifest(directory)
    block = manifest.get("models", {})
    names = manifest["gene_models"]
    no_structure = {
        "KIR2DL5A",
        "KIR2DL5B",
        "KIR2DS1",
        "KIR2DS3",
        "KIR2DS5",
        "KIR3DL2",
        "KIR3DL3",
        "KIR3DS1",
    }
    return pl.DataFrame(
        {
            "gene": names,
            "architecture": [block.get(n, {}).get("architecture") for n in names],
            "length": [block.get(n, {}).get("length") for n in names],
            "n_training_alleles": [block.get(n, {}).get("n_training_sequences") for n in names],
            "has_structure": [n not in no_structure for n in names],
        },
        schema={
            "gene": pl.String,
            "architecture": pl.String,
            "length": pl.UInt32,
            "n_training_alleles": pl.UInt32,
            "has_structure": pl.Boolean,
        },
    )


def models(model_dir: str | Path | None = None) -> pl.DataFrame:
    """Every model in the bundle -- the 15 gene models and the 3 domain models.

    Columns: ``model``, ``kind`` (``gene``/``domain``), ``length``. The ``model`` values are what
    :func:`score` accepts.
    """
    directory = _resolve_model_dir(model_dir)
    manifest = _manifest(directory)
    block = manifest.get("models", {})
    names = manifest["classes"]
    return pl.DataFrame(
        {
            "model": names,
            "kind": [block.get(n, {}).get("kind", "gene") for n in names],
            "length": [block.get(n, {}).get("length") for n in names],
        },
        schema={"model": pl.String, "kind": pl.String, "length": pl.UInt32},
    )


def calibration(model_dir: str | Path | None = None) -> dict:
    """The stored calibration block, for reporting and reproducibility.

    These are measured numbers from training, not tuning knobs: ``per_residue_threshold`` is the
    ``is_kir`` decision boundary on the per-match-state scale, ``tie_tolerance`` the log-odds
    separation below which a gene call is reported as a tie, ``domain_threshold`` the floor a
    domain placement must clear.
    """
    return dict(_manifest(_resolve_model_dir(model_dir)).get("calibration", {}))


def _validate_model_name(name: str, directory: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"model must be a str, got {type(name).__name__}")
    classes = _manifest(directory)["classes"]
    if name not in classes:
        import difflib

        close = difflib.get_close_matches(name, classes, n=3)
        hint = f"; did you mean {', '.join(repr(c) for c in close)}?" if close else ""
        raise ValueError(
            f"unknown model {name!r}{hint} -- call polars_kir_hmm.models() to list the "
            f"{len(classes)} available models"
        )
    return name


def _validate_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


@pl.api.register_expr_namespace("kir")
class KIRNameSpace:
    """The ``.kir`` expression namespace."""

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def classify(
        self,
        with_architecture: bool = True,
        model_dir: str | Path | None = None,
    ) -> pl.Expr:
        """Type a protein-sequence column as KIR and assign its gene.

        Returns a struct column mirroring ``histo_kir_hmm``'s ``ChainType``:

        ==================== ==================================== ==============================
        field                dtype                                meaning
        ==================== ==================================== ==============================
        ``chain_type``       ``String``                           ``"kir"`` or ``"non_kir"``
        ``is_kir``           ``Boolean``                          ``chain_type == "kir"``
        ``gene``             ``String``                           gene name, **null when tied**
        ``genes``            ``List(String)``                     the tie set, sorted
        ``is_tie``           ``Boolean``                          ``len(genes) > 1``
        ``tie_kind``         ``String``                           ``none``/``exact``/``score``
        ``architecture``     ``String``                           observed domain string
        ``domains``          ``List(Struct{domain,start,end,score})`` ordered, non-overlapping
        ``confidence``       ``Float64``                          0-1 in the chain-type call
        ``gene_confidence``  ``Float64``                          probability mass on ``genes``
        ``typicality``       ``Float64``                          percentile under the gene's own
                                                                  training distribution
        ``score``            ``Float64``                          log-odds of the winning gene
        ``region_start``     ``UInt32``                           start of the scored region
        ``region_end``       ``UInt32``                           end of the region (exclusive)
        ``notes``            ``List(String)``                     tie / partial-chain / no-structure
        ``raw_scores``       ``List(Struct{gene,score})``         log-odds against all 15 genes
        ==================== ==================================== ==============================

        with_architecture
            Run domain segmentation. On by default, matching ``type_chain``. Turning it off makes
            the expression several times cheaper when only the gene call is wanted; then
            ``architecture`` is null and ``domains`` empty. Use the :meth:`domains` expression if
            you want segmentation without a gene call.
        model_dir
            A directory of models in ``histo_kir_hmm``'s format. Defaults to the bundled ones.

        Three things about the output are contract, not implementation detail:

        * **A tied call has ``gene`` null.** Read ``genes``; do not take ``genes[0]``. The
          information that would separate KIR2DL5A from KIR2DL5B, or KIR2DL3 from KIR2DS2, is in
          the full-length precursor and not in the ectodomain a structure contains.
        * **``typicality`` saturates on structural input** and must not be thresholded. Within a
          gene, IPD-KIR alleles differ at a handful of positions, so any chain with missing
          residues sits tens of sigma below the training mean. ``confidence`` and
          ``gene_confidence`` are the fields that carry information there.
        * **``architecture`` reports what is observed, not what the gene implies.** A 167-residue
          KIR2DL3 chain segments as ``"D1"``, and a note says the gene call implies ``"D1D2"``.
          That is a correct observation of an incomplete chain.

        ``region_start``/``region_end`` and the domain indices index the **cleaned** sequence --
        uppercased and stripped of everything that is not one of the 20 amino acids or ``X`` -- not
        the input string. If your input contains gaps or punctuation, the offsets will not line up
        with it.

        Null inputs produce null outputs. A sequence with fewer than 40 scorable residues is
        rejected without scoring, with ``score = -inf`` and an explanatory note.
        """
        directory = _resolve_model_dir(model_dir)
        with_architecture = _validate_bool(with_architecture, "with_architecture")

        return register_plugin_function(
            plugin_path=_PLUGIN_PATH,
            function_name="classify_expr",
            args=self._expr,
            kwargs={"with_architecture": with_architecture, "model_dir": directory},
            is_elementwise=True,
        )

    def score(
        self,
        model: str,
        log_odds: bool = True,
        model_dir: str | Path | None = None,
    ) -> pl.Expr:
        """Log-odds of each sequence against a single model's profile HMM.

        Mirrors ``ProfileHMM.log_odds_score``: ``log P(seq|model) - log P(seq|null)``. Positive
        means the sequence is more likely under this model than under the background. ``model``
        is a gene name (``"KIR3DL1"``) or a domain name (``"D0"``); :func:`models` lists them.

        The sequence is cleaned exactly as :meth:`classify` cleans it, but there is no
        sliding-window scan -- the whole cleaned sequence is scored against the one model.

        Scores are **not comparable across architectures**: model lengths are 199 (D1D2), 194
        (D0D2) and 294 (D0D1D2), and log-odds accumulates per match state. Compare within an
        architecture, or divide by ``min(len(sequence), length)`` first -- which is what
        :meth:`classify`'s ``is_kir`` test does.

        log_odds
            Set False for the raw Viterbi score, without the null-model subtraction.

        Null inputs produce null outputs.
        """
        directory = _resolve_model_dir(model_dir)
        model = _validate_model_name(model, directory)
        log_odds = _validate_bool(log_odds, "log_odds")

        return register_plugin_function(
            plugin_path=_PLUGIN_PATH,
            function_name="score_expr",
            args=self._expr,
            kwargs={"model": model, "log_odds": log_odds, "model_dir": directory},
            is_elementwise=True,
        )

    def domains(self, model_dir: str | Path | None = None) -> pl.Expr:
        """Segment each sequence into Ig domains, independently of any gene call.

        Mirrors ``KIRClassifier.call_architecture``. Returns a struct of ``architecture``
        (``String``), ``domains`` (``List(Struct{domain, start, end, score})``) and ``confidence``
        (``Float64``, how decisively each placed domain's own model beat the other two on the same
        span, averaged).

        This consults only the D0/D1/D2 models and never the gene models, so its answer can be
        cross-checked against :meth:`classify`'s gene call. A disagreement is informative rather
        than an error -- it usually means an incomplete chain.

        Sequences with fewer than 60 scorable residues return a null architecture, no domains and
        confidence 0.0. Null inputs produce null outputs.
        """
        directory = _resolve_model_dir(model_dir)
        return register_plugin_function(
            plugin_path=_PLUGIN_PATH,
            function_name="domains_expr",
            args=self._expr,
            kwargs={"model_dir": directory},
            is_elementwise=True,
        )

def classify(expr: IntoExpr, /, **kwargs) -> pl.Expr:
    """Function form of :meth:`KIRNameSpace.classify`, for when you prefer it."""
    e = pl.col(expr) if isinstance(expr, str) else expr
    return e.kir.classify(**kwargs)


def score(expr: IntoExpr, /, model: str, **kwargs) -> pl.Expr:
    """Function form of :meth:`KIRNameSpace.score`, for when you prefer it."""
    e = pl.col(expr) if isinstance(expr, str) else expr
    return e.kir.score(model, **kwargs)


def domains(expr: IntoExpr, /, **kwargs) -> pl.Expr:
    """Function form of :meth:`KIRNameSpace.domains`, for when you prefer it."""
    e = pl.col(expr) if isinstance(expr, str) else expr
    return e.kir.domains(**kwargs)
