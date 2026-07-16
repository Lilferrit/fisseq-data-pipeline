"""Per-variant cell-level feature aggregation strategies.

Defines 8 concrete :class:`BaseAggregator` implementations (mean, median, MAD, std,
EMD, KS, QQ, AUROC) and two Hydra entry points: ``fisseq-aggregate`` (standalone,
normalizes to synonymous baseline and attaches metadata) and
``fisseq-aggregate-feature-type`` (Nextflow processes ``AGGREGATE_FEATURE_TYPE`` and
``AGGREGATE_HALF`` — lean per-feature-type aggregation used by the feature-selection
branch).
"""

import abc
import dataclasses
import logging
import pathlib
import re
import time
from typing import ClassVar, Optional

import hydra
import numpy as np
import polars as pl
import scipy.stats
import sklearn.metrics
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import LabeledInputConfig
from .normalize import Normalizer
from .utils.batches import load_batches
from .utils.constants import CONTROL_COLUMN, CONTROL_COLUMN_NAME, FEATURE_SELECTOR
from .utils.log import setup_logging
from .utils.metadata import get_aggregate_meta_data
from .utils.splits import filter_by_index_file
from .utils.vectors import compute_impact_score


@dataclasses.dataclass
class AggregateConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the aggregation entry point.

    Attributes
    ----------
    aggregator : str
        Aggregation method. One of: ``mean``, ``median``, ``MAD``, ``std``,
        ``EMD``, ``KS``, ``QQ``, ``AUROC``. Required.
    save_normalizer : bool
        If ``True``, persist the fitted :class:`.normalize.Normalizer` alongside
        the output. Defaults to ``True``.
    block_list_file : str or None
        Optional path to a parquet file with at least ``feature`` (str) and
        ``feature_ok`` (bool) columns. Features where ``feature_ok`` is
        ``False`` are excluded from aggregation. Defaults to ``None`` (no
        features blocked).
    feature_batch_size : int or None
        If set, split features into chunks of at most this many columns per
        aggregation pass to bound peak memory. Each chunk collects its
        variant and reference rows into numpy arrays, so batching bounds how
        much of that data is materialized in memory at once. Defaults to
        ``200``. Set to ``None`` for a single unbatched pass.
    """

    aggregator: str = MISSING
    save_normalizer: bool = True
    block_list_file: Optional[str] = None
    compute_impact_score: bool = True
    feature_batch_size: Optional[int] = 200


_cs = ConfigStore.instance()
_cs.store(name="aggregate_main", node=AggregateConfig)


def _is_synonymous(v: str) -> bool:
    """Return True iff v encodes a synonymous (same-amino-acid) substitution."""
    if "fs" in v or v.endswith("-") or "X" in v or "*" in v or "WT" in v:
        return False
    match = re.match(r"([A-Z])(\d+)([A-Z])", v)
    return match is not None and match.group(1) == match.group(3)


def variant_classification(lf: pl.LazyFrame, label_col: str) -> pl.LazyFrame:
    """
    Add a boolean ``CONTROL_COLUMN_NAME`` column to a LazyFrame.

    The added column is ``True`` for rows whose variant label (in ``label_col``)
    encodes a synonymous amino-acid substitution — i.e. the same amino acid
    appears before and after the position — as determined by
    :func:`_is_synonymous`.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input LazyFrame.
    label_col : str
        Name of the column containing variant label strings.

    Returns
    -------
    pl.LazyFrame
        The input frame with an additional boolean ``CONTROL_COLUMN_NAME``
        column.
    """
    return lf.with_columns(
        pl.col(label_col)
        .map_elements(_is_synonymous, return_dtype=pl.Boolean)
        .alias(CONTROL_COLUMN_NAME)
    )


def _collect_reference_pool(
    lf: pl.LazyFrame, feature_cols: list[str]
) -> dict[str, np.ndarray]:
    """
    Collect the control/reference pool for ``feature_cols`` once, as plain
    numpy arrays shared by reference (not copied) across every variant
    group's stat computation.

    The reference pool is shared by every variant label (a single global
    control group, not split per-label), so collecting it once here and
    looking it up by feature name avoids replicating it once per label
    (which a join into the per-label LazyFrame would do).

    Values are raw (nulls become NaN via ``to_numpy()``); cleaning is the
    caller's responsibility via :func:`_clean`, which already drops
    non-finite values, so the null->NaN conversion is a no-op for
    correctness.
    """
    if not feature_cols:
        return {}
    ref_df = lf.filter(CONTROL_COLUMN).select(feature_cols).collect()
    return {f: ref_df[f].to_numpy() for f in feature_cols}


def _native_clean(feat: str) -> pl.Expr:
    """
    Native-Polars equivalent of :func:`_clean` applied to a per-label list
    column: drop null, NaN, and Inf entries from the list for ``feat`` before
    any further native list-based computation. ``is_finite()`` on a Float64
    element is ``False`` for null/NaN/Inf alike, matching ``_clean``'s
    ``v is not None and np.isfinite(v)`` filter exactly (verified against
    empty, single-value, null-mixed, NaN, and Inf cases).
    """
    return pl.col(feat).list.eval(pl.element().filter(pl.element().is_finite()))


def _chunk(items: list[str], size: Optional[int]) -> list[list[str]]:
    """
    Split ``items`` into chunks of at most ``size`` elements. ``size`` of
    ``None`` or ``<= 0`` means "no batching" (a single chunk containing all
    items) — a defensive no-op rather than a footgun for ``size=0``.
    """
    if size is None or size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def _clean(values) -> np.ndarray:
    """Drop None and non-finite entries from a value list or numpy array."""
    if values is None:
        values = []
    return np.fromiter(
        (v for v in values if v is not None and np.isfinite(v)), dtype=float
    )


def _finalize(result: float) -> Optional[float]:
    """None if non-finite, else a plain python float."""
    return float(result) if np.isfinite(result) else None


def _native_aggregate_feature_batch(
    lf: pl.LazyFrame, label_col: str, feature_cols: list[str], exprs: list[pl.Expr]
) -> pl.LazyFrame:
    """
    Shared group_by/select boilerplate for native aggregators: group
    non-control rows by ``label_col`` into per-label list columns, then
    select ``exprs`` (one aliased native-Polars-expression per feature)
    against those list columns. Stays entirely in Arrow — no Python boxing.
    """
    variant_lists = (
        lf.filter(~CONTROL_COLUMN)
        .group_by(label_col)
        .agg([pl.col(f) for f in feature_cols])
    )
    return variant_lists.select([label_col] + exprs)


class BaseAggregator(abc.ABC):
    """
    Base class for all aggregators.

    Subclasses declare :attr:`_stat_suffix` (e.g. ``"_mean"``, ``"_EMD"``).
    :meth:`aggregate` is the public orchestration entry point: it chunks
    features (via :func:`_chunk`), computes each chunk via
    :meth:`_aggregate_feature_batch`, and joins per-chunk results back
    together on ``label_col``.

    The default :meth:`_aggregate_feature_batch` is a numpy-vectorized path:
    it collects each chunk's variant and reference rows into numpy arrays
    once, then loops over (group, feature) pairs calling :meth:`_compute_stat`
    with already-cleaned numpy arrays — no Arrow struct boxing, no
    per-row Python calls. Subclasses that need SciPy/scikit-learn or the
    reference pool (EMD, KS, QQ, AUROC) implement only :meth:`_compute_stat`.

    Subclasses that can express their statistic as a native Polars
    list-expression (mean, median, std, MAD) instead override
    :meth:`_aggregate_feature_batch` directly, bypassing the numpy path and
    the reference pool entirely — Polars-native list expressions are faster
    than anything the numpy path could offer for these. Such subclasses do
    not implement :meth:`_compute_stat`.

    Parameters
    ----------
    label_col : str
        Name of the column used to identify variant groups. Defaults to
        ``"meta_aa_changes"``.
    block_list : set[str] or None
        Aggregated output column names to skip (e.g. ``"f1_EMD"``). Blocked
        statistics are not computed. Defaults to ``None``.
    """

    _stat_suffix: ClassVar[str]

    def __init__(
        self,
        label_col: str = "meta_aa_changes",
        block_list: Optional[set[str]] = None,
    ) -> None:
        self.label_col = label_col
        self.block_list = block_list

    def _compute_stat(
        self, group_vals: np.ndarray, ref_vals: np.ndarray
    ) -> Optional[float]:
        """
        Compute a single statistic from a label's cleaned variant values
        (``group_vals``) and the cleaned reference/control pool
        (``ref_vals``) for one feature. Called only by the default numpy
        :meth:`_aggregate_feature_batch`, and only with non-empty,
        already-``_clean``ed arrays (the caller owns the empty-array
        short-circuit). Only implemented by aggregators that use the
        default numpy path (EMD, KS, QQ, AUROC); native aggregators
        (mean/median/std/MAD) override :meth:`_aggregate_feature_batch`
        instead and never call this.
        """
        raise NotImplementedError

    def _feature_columns(self, lf: pl.LazyFrame) -> list[str]:
        return [
            f
            for f in lf.select(FEATURE_SELECTOR).collect_schema().names()
            if f"{f}{self._stat_suffix}" not in (self.block_list or set())
        ]

    def aggregate(
        self, lf: pl.LazyFrame, feature_batch_size: Optional[int] = None
    ) -> pl.LazyFrame:
        """
        Compute per-label statistics for every non-block-listed feature.

        Parameters
        ----------
        lf : pl.LazyFrame
            Input LazyFrame containing the label column, a ``CONTROL_COLUMN``
            boolean column, and feature columns.
        feature_batch_size : int or None
            If set, features are processed in chunks of at most this many
            columns at a time (each chunk performs its own collection and
            computation), and per-chunk results are joined back together on
            ``label_col``. This trades some redundant work for a lower
            peak-memory materialization per chunk. ``None`` (default)
            processes all features in a single pass, identical to unbatched
            behavior. The module-level :func:`aggregate` function and the
            Hydra configs default this to ``200`` instead, to bound memory
            in production pipeline runs.

        Returns
        -------
        pl.LazyFrame
            One row per non-control variant group with computed statistics.
        """
        feature_cols = self._feature_columns(lf)
        batches = _chunk(feature_cols, feature_batch_size)
        logging.info(
            "%s: %d feature(s) to aggregate in %d chunk(s) (feature_batch_size=%s)",
            type(self).__name__,
            len(feature_cols),
            len(batches),
            feature_batch_size,
        )
        results = [
            self._aggregate_feature_batch(
                lf, cols, batch_idx=i, total_batches=len(batches)
            )
            for i, cols in enumerate(batches, start=1)
        ]

        out = results[0]
        for r in results[1:]:
            out = out.join(r, on=self.label_col, how="inner")
        return out

    def _aggregate_feature_batch(
        self,
        lf: pl.LazyFrame,
        feature_cols: list[str],
        batch_idx: int,
        total_batches: int,
    ) -> pl.LazyFrame:
        """
        Compute this chunk's statistics for every feature in
        ``feature_cols``, one row per non-control variant group. Default
        implementation: the numpy path (see module docstring / class
        docstring). Native aggregators override this entirely.
        """
        t0 = time.perf_counter()

        reference_pool = _collect_reference_pool(lf, feature_cols)
        variant_df = (
            lf.filter(~CONTROL_COLUMN).select([self.label_col] + feature_cols).collect()
        )

        labels = variant_df[self.label_col].to_numpy()
        order = np.argsort(labels, kind="stable")
        sorted_labels = labels[order]
        unique_labels, group_starts = np.unique(sorted_labels, return_index=True)
        n_groups = len(unique_labels)

        cleaned_ref = {f: _clean(reference_pool[f]) for f in feature_cols}
        n_ref_rows = len(next(iter(reference_pool.values()))) if reference_pool else 0

        columns: dict[str, object] = {
            self.label_col: pl.Series(self.label_col, unique_labels)
        }
        for feat in feature_cols:
            feat_sorted = variant_df[feat].to_numpy()[order]
            groups = np.split(feat_sorted, group_starts[1:]) if n_groups else []
            ref_vals = cleaned_ref[feat]
            stats: list[Optional[float]] = []
            for group_vals in groups:
                cleaned_group = _clean(group_vals)
                if len(cleaned_group) == 0 or len(ref_vals) == 0:
                    stats.append(None)
                else:
                    stats.append(self._compute_stat(cleaned_group, ref_vals))
            columns[f"{feat}{self._stat_suffix}"] = pl.Series(
                f"{feat}{self._stat_suffix}", stats, dtype=pl.Float64
            )

        elapsed = time.perf_counter() - t0
        logging.info(
            "%s chunk %d/%d: %d feature(s), %d variant group(s), %d reference row(s), %.3fs",
            type(self).__name__,
            batch_idx,
            total_batches,
            len(feature_cols),
            n_groups,
            n_ref_rows,
            elapsed,
        )
        return pl.DataFrame(columns).lazy()


class MeanAggregator(BaseAggregator):
    """Computes per-group mean for each feature column."""

    _stat_suffix = "_mean"

    def _aggregate_feature_batch(
        self,
        lf: pl.LazyFrame,
        feature_cols: list[str],
        batch_idx: int,
        total_batches: int,
    ) -> pl.LazyFrame:
        exprs = [
            _native_clean(f).list.mean().alias(f"{f}{self._stat_suffix}")
            for f in feature_cols
        ]
        logging.info(
            "%s chunk %d/%d: %d feature(s) (native)",
            type(self).__name__,
            batch_idx,
            total_batches,
            len(feature_cols),
        )
        return _native_aggregate_feature_batch(lf, self.label_col, feature_cols, exprs)


class MedianAggregator(BaseAggregator):
    """Computes per-group median for each feature column."""

    _stat_suffix = "_median"

    def _aggregate_feature_batch(
        self,
        lf: pl.LazyFrame,
        feature_cols: list[str],
        batch_idx: int,
        total_batches: int,
    ) -> pl.LazyFrame:
        exprs = [
            _native_clean(f).list.median().alias(f"{f}{self._stat_suffix}")
            for f in feature_cols
        ]
        logging.info(
            "%s chunk %d/%d: %d feature(s) (native)",
            type(self).__name__,
            batch_idx,
            total_batches,
            len(feature_cols),
        )
        return _native_aggregate_feature_batch(lf, self.label_col, feature_cols, exprs)


class MADAggregator(BaseAggregator):
    """Computes per-group median absolute deviation (MAD) for each feature column."""

    _stat_suffix = "_MAD"

    def _aggregate_feature_batch(
        self,
        lf: pl.LazyFrame,
        feature_cols: list[str],
        batch_idx: int,
        total_batches: int,
    ) -> pl.LazyFrame:
        # Verified against np.median(np.abs(vals - np.median(vals))) for
        # empty, single-value, normal, and NaN/Inf-present cases — matches
        # exactly (see test_mad_native_matches_numpy_with_nan_inf_present).
        exprs = [
            _native_clean(f)
            .list.eval((pl.element() - pl.element().median()).abs())
            .list.median()
            .alias(f"{f}{self._stat_suffix}")
            for f in feature_cols
        ]
        logging.info(
            "%s chunk %d/%d: %d feature(s) (native)",
            type(self).__name__,
            batch_idx,
            total_batches,
            len(feature_cols),
        )
        return _native_aggregate_feature_batch(lf, self.label_col, feature_cols, exprs)


class StdAggregator(BaseAggregator):
    """Computes per-group standard deviation for each feature column."""

    _stat_suffix = "_std"

    def _aggregate_feature_batch(
        self,
        lf: pl.LazyFrame,
        feature_cols: list[str],
        batch_idx: int,
        total_batches: int,
    ) -> pl.LazyFrame:
        # Confirmed: list.std(ddof=1) returns null for lists of length < 2
        # (see test_std_native_single_value_group_returns_null).
        exprs = [
            _native_clean(f).list.std(ddof=1).alias(f"{f}{self._stat_suffix}")
            for f in feature_cols
        ]
        logging.info(
            "%s chunk %d/%d: %d feature(s) (native)",
            type(self).__name__,
            batch_idx,
            total_batches,
            len(feature_cols),
        )
        return _native_aggregate_feature_batch(lf, self.label_col, feature_cols, exprs)


class EMDAggregator(BaseAggregator):
    """
    Computes per-group 1D Wasserstein distances (Earth Mover's Distance)
    against the reference distribution for each feature column.
    """

    _stat_suffix = "_EMD"

    def _compute_stat(
        self, group_vals: np.ndarray, ref_vals: np.ndarray
    ) -> Optional[float]:
        return _finalize(scipy.stats.wasserstein_distance(group_vals, ref_vals))


class KSAggregator(BaseAggregator):
    """
    Computes per-group two-sample Kolmogorov-Smirnov statistics against
    the reference distribution for each feature column.
    """

    _stat_suffix = "_KS"

    def _compute_stat(
        self, group_vals: np.ndarray, ref_vals: np.ndarray
    ) -> Optional[float]:
        return _finalize(scipy.stats.ks_2samp(group_vals, ref_vals).statistic)


class QQCorrelationAggregator(BaseAggregator):
    """
    Computes per-group Q-Q correlation against the reference distribution for
    each feature column.

    Parameters
    ----------
    label_col : str
        Variant label column name.
    n_quantiles : int, optional
        Number of quantile points to evaluate. Defaults to ``100``.
    """

    _stat_suffix = "_QQ"

    def __init__(
        self,
        label_col: str = "meta_aa_changes",
        n_quantiles: int = 100,
        block_list: Optional[set[str]] = None,
    ) -> None:
        super().__init__(label_col, block_list)
        self.quantile_points = np.linspace(0, 1, n_quantiles)

    def _compute_stat(
        self, group_vals: np.ndarray, ref_vals: np.ndarray
    ) -> Optional[float]:
        variant_q = np.quantile(group_vals, self.quantile_points)
        reference_q = np.quantile(ref_vals, self.quantile_points)
        return _finalize(scipy.stats.pearsonr(variant_q, reference_q).statistic)


class AUROCAggregator(BaseAggregator):
    """
    Computes per-group AUROC against the reference distribution for each
    feature column.

    Variant samples are labelled ``1`` and reference samples ``0``.
    ``0.5`` indicates identical distributions; ``1.0`` indicates perfect
    separability.
    """

    _stat_suffix = "_AUROC"

    def _compute_stat(
        self, group_vals: np.ndarray, ref_vals: np.ndarray
    ) -> Optional[float]:
        concatenated = np.concatenate([ref_vals, group_vals])
        labels = np.concatenate([np.zeros(len(ref_vals)), np.ones(len(group_vals))])
        auroc = sklearn.metrics.roc_auc_score(labels, concatenated)
        if auroc < 0.5:
            auroc = 1 - auroc
        return _finalize(auroc)


_AGGREGATORS: dict[str, type[BaseAggregator]] = {
    "mean": MeanAggregator,
    "median": MedianAggregator,
    "MAD": MADAggregator,
    "std": StdAggregator,
    "EMD": EMDAggregator,
    "KS": KSAggregator,
    "QQ": QQCorrelationAggregator,
    "AUROC": AUROCAggregator,
}


def aggregate(
    lf: pl.LazyFrame,
    label_col: str,
    aggregator_name: str,
    block_list: Optional[set[str]] = None,
    feature_batch_size: Optional[int] = 200,
) -> pl.LazyFrame:
    """
    Run the specified aggregator on cell-level data and return per-label statistics.

    Control rows (where ``CONTROL_COLUMN`` is ``True``) are passed as the
    reference distribution to all aggregators that require one.

    Parameters
    ----------
    lf : pl.LazyFrame
        Cell-level LazyFrame. Must contain a boolean ``CONTROL_COLUMN`` column.
    label_col : str
        Name of the column identifying variant labels.
    aggregator_name : str
        Aggregation method. One of: ``mean``, ``median``, ``MAD``, ``std``,
        ``EMD``, ``KS``, ``QQ``, ``AUROC``.
    block_list : set[str] or None
        Aggregated output column names to skip (e.g. ``"f1_EMD"``). Blocked
        statistics are not computed and do not appear in the output. Names
        that do not match any aggregated output are silently ignored. Defaults
        to ``None``.
    feature_batch_size : int or None
        If set, split features into chunks of at most this many columns per
        aggregation pass to bound peak memory. Each chunk collects its
        variant and reference rows into numpy arrays, so batching bounds how
        much of that data is materialized in memory at once. Defaults to
        ``200``. Set to ``None`` for a single unbatched pass.

    Returns
    -------
    pl.LazyFrame
        One row per non-control variant group with computed statistics.
    """
    valid = set(_AGGREGATORS)
    if aggregator_name not in valid:
        raise ValueError(
            f"Unknown aggregator {aggregator_name!r}. Choose from: {sorted(valid)}"
        )

    agg = _AGGREGATORS[aggregator_name](label_col=label_col, block_list=block_list)
    return agg.aggregate(lf, feature_batch_size=feature_batch_size)


@hydra.main(version_base=None, config_path=None, config_name="aggregate_main")
def main(cfg: DictConfig) -> None:
    """
    Aggregate cell-level features, z-score normalize to synonymous baseline,
    and attach per-variant metadata.

    ``input_file`` is interpreted as a glob pattern via
    :func:`.utils.load_batches`; each matching file becomes one batch, with
    ``meta_batch`` set to the filename stem. A concrete (non-glob) path is
    treated as a single-file pattern.

    Runs the configured aggregator to produce one row per variant, marks
    synonymous variants as the normalization reference via
    :func:`variant_classification`, fits a :class:`.normalize.Normalizer` on
    those rows, applies it, joins per-variant metadata from
    :func:`get_aggregate_meta_data`, and writes the result.

    Output path
    -----------
    - Glob input: ``{output_root}.output.parquet`` or ``{output_dir}/output.parquet``
    - Single-file input: ``{output_root}.{stem}.{ext}`` or
      ``{output_dir}/{filename}`` (same name as the input file)

    If ``save_normalizer`` is ``True``, the fitted :class:`.normalize.Normalizer`
    is also written alongside the output as ``normalizer.parquet``.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.aggregate \\
            output_dir=./out \\
            'input_file=data/batches/*.parquet' \\
            aggregator=EMD
    """
    agg_cfg: AggregateConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(agg_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agg_cfg.output_dir = output_dir
    setup_logging(agg_cfg, "aggregate")

    logging.info("Loading input from %s", agg_cfg.input_file)
    lf, output_stem = load_batches(agg_cfg.input_file)

    block_list: Optional[set[str]] = None
    if agg_cfg.block_list_file is not None:
        logging.info("Loading block list from %s", agg_cfg.block_list_file)
        bl_df = pl.read_parquet(agg_cfg.block_list_file)
        block_list = set(bl_df.filter(~pl.col("feature_ok"))["feature"].to_list())

    logging.info("Running %s aggregator", agg_cfg.aggregator)
    logging.info(
        "Classifying variants and marking synonymous as normalization reference"
    )
    agg_lf = variant_classification(
        aggregate(
            lf,
            label_col=agg_cfg.label_column,
            aggregator_name=agg_cfg.aggregator,
            block_list=block_list,
            feature_batch_size=agg_cfg.feature_batch_size,
        ),
        agg_cfg.label_column,
    )

    logging.info("Fitting normalizer on synonymous rows")
    normalizer = Normalizer.from_lazyframe(agg_lf, fit_only_on_control=True)

    logging.info("Applying normalizer")
    normalized_lf = normalizer.apply(agg_lf)

    if agg_cfg.output_root is not None:
        out_path = pathlib.Path(f"{agg_cfg.output_root}.{output_stem}.parquet")
    else:
        out_path = output_dir / f"{output_stem}.parquet"

    logging.info("Adding queries to retrieve metadata")
    meta_lf = get_aggregate_meta_data(lf, agg_cfg.label_column)
    normalized_lf = normalized_lf.join(meta_lf, on=agg_cfg.label_column)

    if cfg.compute_impact_score:
        logging.info("Computing impact scores")
        normalized_lf = compute_impact_score(normalized_lf)

    logging.info("Writing output to %s", out_path)
    normalized_lf.sink_parquet(out_path)

    if agg_cfg.save_normalizer:
        if agg_cfg.output_root is not None:
            norm_path = pathlib.Path(f"{agg_cfg.output_root}.normalizer.parquet")
        else:
            norm_path = output_dir / "normalizer.parquet"
        logging.info("Saving normalizer to %s", norm_path)
        normalizer.save(norm_path)

    logging.info("Done")


@dataclasses.dataclass
class FeatureTypeAggregateConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the lean per-feature-type aggregation
    entry point.

    Shared by the feature-selection pipeline's stage 1 (full aggregation)
    and stage 2b (per-pseudo-replicate-half aggregation).

    Attributes
    ----------
    aggregator : str
        A concrete key in ``_AGGREGATORS`` (``mean``, ``median``, ``MAD``,
        ``std``, ``EMD``, ``KS``, ``QQ``, ``AUROC``). Required.
    index_file : str or None
        Optional path to a single-column ``TMP_IDX_COL`` parquet file (as
        written by :func:`fisseq_data_pipeline.features.generate_split_main`)
        naming a subset of cell-level rows to aggregate over (e.g. one
        pseudo-replicate half). When ``None``, all rows are aggregated.
        Defaults to ``None``.
    feature_batch_size : int or None
        If set, split features into chunks of at most this many columns per
        aggregation pass to bound peak memory. Each chunk collects its
        variant and reference rows into numpy arrays, so batching bounds how
        much of that data is materialized in memory at once. Defaults to
        ``200``. Set to ``None`` for a single unbatched pass.
    """

    aggregator: str = MISSING
    index_file: Optional[str] = None
    feature_batch_size: Optional[int] = 200


_cs.store(name="aggregate_feature_type_main", node=FeatureTypeAggregateConfig)


@hydra.main(
    version_base=None, config_path=None, config_name="aggregate_feature_type_main"
)
def feature_type_main(cfg: DictConfig) -> None:
    """
    Hydra entry point: aggregate cell-level features for one feature type.

    ``input_file`` is interpreted as a glob pattern via :func:`load_batches`
    (a concrete non-glob path is a single-file pattern). Rows are optionally
    filtered to ``index_file`` via :func:`.utils.splits.filter_by_index_file`.
    Runs the configured single aggregator via :func:`aggregate` and writes a
    lean output containing only ``[label_column] + <feature type's stat
    columns>`` — no normalizer, no metadata join, no impact score (those
    happen once, later, in the final feature-selection stage).

    Relies on the ``meta_is_control`` column already present on the input
    (set upstream by ``normalize.py``'s WT-based ``control_sample_query``) as
    the aggregator's reference/control group; does not call
    :func:`variant_classification`.

    Output path
    -----------
    - Glob input: ``{output_root}.output.parquet`` or ``{output_dir}/output.parquet``
    - Single-file input: ``{output_root}.{stem}.parquet`` or
      ``{output_dir}/{stem}.parquet``

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.aggregate \\
            output_dir=./out \\
            input_file=data/normalized.parquet \\
            aggregator=mean \\
            index_file=./half1.parquet
    """
    ft_cfg: FeatureTypeAggregateConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(ft_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ft_cfg.output_dir = output_dir
    setup_logging(ft_cfg, "aggregate_feature_type")

    logging.info("Loading input from %s", ft_cfg.input_file)
    lf, output_stem = load_batches(ft_cfg.input_file)

    logging.info("Filtering by index_file=%s", ft_cfg.index_file)
    lf = filter_by_index_file(lf, ft_cfg.index_file)

    logging.info("Running %s aggregator", ft_cfg.aggregator)
    agg_lf = aggregate(
        lf,
        label_col=ft_cfg.label_column,
        aggregator_name=ft_cfg.aggregator,
        feature_batch_size=ft_cfg.feature_batch_size,
    )

    if ft_cfg.output_root is not None:
        out_path = pathlib.Path(f"{ft_cfg.output_root}.{output_stem}.parquet")
    else:
        out_path = output_dir / f"{output_stem}.parquet"

    logging.info("Writing output to %s", out_path)
    agg_lf.sink_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
