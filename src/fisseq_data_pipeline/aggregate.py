"""Per-variant cell-level feature aggregation strategies.

Defines 7 concrete :class:`BaseAggregator` implementations (mean, median, MAD, std,
KS, QQ, AUROC) and two Hydra entry points: ``fisseq-aggregate`` (standalone,
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
        ``KS``, ``QQ``, ``AUROC``. Required.
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
        ``500``. Set to ``None`` for a single unbatched pass.
    """

    aggregator: str = MISSING
    save_normalizer: bool = True
    block_list_file: Optional[str] = None
    compute_impact_score: bool = True
    feature_batch_size: Optional[int] = 500


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


def _collect_clean_reference_pool(
    lf: pl.LazyFrame, feature_cols: list[str]
) -> dict[str, np.ndarray]:
    """
    :func:`_collect_reference_pool` followed by :func:`_clean` per feature —
    the reference-based native aggregators (KS, AUROC, QQ) all need the
    same "collect once per chunk, clean once per feature" reference array
    to embed as a Polars literal, so this is the shared entry point for all
    three rather than each repeating the two-step pattern.
    """
    raw = _collect_reference_pool(lf, feature_cols)
    return {f: _clean(raw[f]) for f in feature_cols}


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


def _native_quantiles(list_expr: pl.Expr, probs: np.ndarray) -> pl.Expr:
    """
    Per-row quantiles of a list column at fixed probability points
    ``probs``, matching ``np.quantile(..., interpolation="linear")``
    bit-for-bit (verified across 200 randomized trials, including
    single-element and empty lists).

    Computed via one sort + a per-row index gather rather than one
    ``list.eval(...quantile...)`` call per probability point: with 100
    quantile points and hundreds of features per chunk, the naive
    one-expression-per-point approach makes the chunk's ``.select()`` call
    contain ``n_quantiles * feature_batch_size`` sub-expressions (50,000 at
    the defaults), which measured *slower* than the numpy loop it replaces
    (247.9s vs a 157.4s baseline at 500 labels x 500 features x 100
    quantiles). This version measured 22.5s at the same scale.

    ``null_on_oob=True`` on the gathers is required, not defensive
    padding: an empty list (``n=0``) produces negative/garbage indices
    after the ``(n - 1)`` scaling, and Polars' ``when/otherwise`` does not
    short-circuit — the ``otherwise`` branch is evaluated for every row
    before the mask is applied, so an out-of-bounds gather raises
    regardless of whether that row's result is later discarded (confirmed
    directly: without ``null_on_oob``, a single empty group anywhere in the
    chunk crashes the whole query with ``OutOfBoundsError``).
    """
    n = list_expr.list.len()
    sorted_list = list_expr.list.eval(pl.element().sort())
    probs_lit = pl.lit(pl.Series([probs.tolist()]))
    idx = probs_lit * (n - 1).cast(pl.Float64)
    lower_f = idx.list.eval(pl.element().floor())
    upper_f = idx.list.eval(
        (pl.element().floor() + 1).clip(upper_bound=pl.element().max())
    )
    frac = idx.list.eval(pl.element() - pl.element().floor())
    lower_idx = lower_f.list.eval(pl.element().cast(pl.UInt32))
    upper_idx = upper_f.list.eval(pl.element().cast(pl.UInt32))
    lower_vals = sorted_list.list.gather(lower_idx, null_on_oob=True)
    upper_vals = sorted_list.list.gather(upper_idx, null_on_oob=True)
    return lower_vals + frac * (upper_vals - lower_vals)


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

    Subclasses declare :attr:`_stat_suffix` (e.g. ``"_mean"``, ``"_KS"``).
    :meth:`aggregate` is the public orchestration entry point: it chunks
    features (via :func:`_chunk`), computes each chunk via
    :meth:`_aggregate_feature_batch`, and joins per-chunk results back
    together on ``label_col``.

    Every concrete aggregator implements :meth:`_aggregate_feature_batch`
    directly as native Polars list expressions over per-label list columns
    (built by grouping non-control rows on ``label_col``) — no numpy
    materialization, no per-(group, feature) Python loop. Aggregators that
    need the reference/control pool (KS, QQ, AUROC) collect it once per
    chunk via :func:`_collect_reference_pool` and embed it as a broadcast
    Polars literal, rather than replicating it per variant group.

    Parameters
    ----------
    label_col : str
        Name of the column used to identify variant groups. Defaults to
        ``"meta_aa_changes"``.
    block_list : set[str] or None
        Aggregated output column names to skip (e.g. ``"f1_KS"``). Blocked
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
            Hydra configs default this to ``500`` instead, to bound memory
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

    @abc.abstractmethod
    def _aggregate_feature_batch(
        self,
        lf: pl.LazyFrame,
        feature_cols: list[str],
        batch_idx: int,
        total_batches: int,
    ) -> pl.LazyFrame:
        """
        Compute this chunk's statistics for every feature in
        ``feature_cols``, one row per non-control variant group. Every
        concrete subclass implements this as native Polars list
        expressions (see class docstring) — there is no shared default.
        """
        raise NotImplementedError

        labels = variant_df[self.label_col].to_numpy()
        order = np.argsort(labels, kind="stable")
        sorted_labels = labels[order]
        unique_labels, group_starts = np.unique(sorted_labels, return_index=True)
        n_groups = len(unique_labels)

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


class KSAggregator(BaseAggregator):
    """
    Computes per-group two-sample Kolmogorov-Smirnov statistics against
    the reference distribution for each feature column.
    """

    _stat_suffix = "_KS"

    def _aggregate_feature_batch(
        self,
        lf: pl.LazyFrame,
        feature_cols: list[str],
        batch_idx: int,
        total_batches: int,
    ) -> pl.LazyFrame:
        cleaned_ref = _collect_clean_reference_pool(lf, feature_cols)
        exprs = [self._ks_expr(feat, cleaned_ref[feat]) for feat in feature_cols]
        logging.info(
            "%s chunk %d/%d: %d feature(s) (native)",
            type(self).__name__,
            batch_idx,
            total_batches,
            len(feature_cols),
        )
        return _native_aggregate_feature_batch(lf, self.label_col, feature_cols, exprs)

    def _ks_expr(self, feat: str, ref_vals: np.ndarray) -> pl.Expr:
        alias = f"{feat}{self._stat_suffix}"
        n_ref = len(ref_vals)
        if n_ref == 0:
            return pl.lit(None, dtype=pl.Float64).alias(alias)

        # Signed-weight cumulative-sum KS statistic: +1/n_group per variant
        # value, -1/n_ref per reference value. Sort the combined values,
        # cumsum the weights, and take the max |cumsum| — but only at the
        # LAST position of each run of tied values (ties must be resolved
        # together, not mid-tie, or a spurious intermediate extremum can
        # exceed the true statistic). Verified against
        # scipy.stats.ks_2samp across 500 randomized trials, including
        # tie-heavy integer data and n=1 groups.
        group_list = _native_clean(feat)
        ref_val_lit = pl.lit(pl.Series([ref_vals.tolist()]))
        ref_w_lit = pl.lit(pl.Series([[-1.0 / n_ref] * n_ref]))

        g_weight = group_list.list.eval((pl.element() * 0 + 1.0) / pl.element().count())
        combined_val = pl.concat_list([group_list, ref_val_lit])
        combined_w = pl.concat_list([g_weight, ref_w_lit])

        order = combined_val.list.eval(pl.element().arg_sort())
        val_sorted = combined_val.list.gather(order)
        w_sorted = combined_w.list.gather(order)

        cumsum = w_sorted.list.eval(pl.element().cum_sum())
        next_val = val_sorted.list.shift(-1)
        # `!=` between two List columns is whole-list equality, not
        # element-wise — subtract instead (arithmetic *is* element-wise
        # for List columns) and compare each element to 0 inside list.eval.
        diff = val_sorted - next_val
        is_last_f = diff.list.eval(
            ((pl.element() != 0) | pl.element().is_null()).cast(pl.Float64)
        )
        candidate = cumsum.list.eval(pl.element().abs())
        # Non-last positions become 0.0 (multiply, not pl.when — when/then
        # does not broadcast element-wise over a List(Boolean) predicate),
        # which never wins the subsequent max since a KS statistic is >= 0.
        ks_stat = (candidate * is_last_f).list.max()

        result = pl.when(group_list.list.len() == 0).then(None).otherwise(ks_stat)
        return result.alias(alias)


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

    def _aggregate_feature_batch(
        self,
        lf: pl.LazyFrame,
        feature_cols: list[str],
        batch_idx: int,
        total_batches: int,
    ) -> pl.LazyFrame:
        cleaned_ref = _collect_clean_reference_pool(lf, feature_cols)
        exprs = [self._qq_expr(feat, cleaned_ref[feat]) for feat in feature_cols]
        logging.info(
            "%s chunk %d/%d: %d feature(s) (native)",
            type(self).__name__,
            batch_idx,
            total_batches,
            len(feature_cols),
        )
        return _native_aggregate_feature_batch(lf, self.label_col, feature_cols, exprs)

    def _qq_expr(self, feat: str, ref_vals: np.ndarray) -> pl.Expr:
        alias = f"{feat}{self._stat_suffix}"
        if len(ref_vals) == 0:
            return pl.lit(None, dtype=pl.Float64).alias(alias)

        # Reference-side quantiles and their centering are fixed per
        # feature (the reference pool doesn't vary by group), so compute
        # them once here in plain numpy rather than per group.
        ref_q = np.quantile(ref_vals, self.quantile_points)
        if ref_q.max() == ref_q.min():
            # Constant reference quantile profile (e.g. a single distinct
            # reference value) -> correlation is undefined for every group
            # regardless of its own data (scipy.stats.pearsonr would raise
            # ConstantInputWarning and return nan here).
            return pl.lit(None, dtype=pl.Float64).alias(alias)
        y_mean = ref_q.mean()
        y_centered_lit = pl.lit(pl.Series([(ref_q - y_mean).tolist()]))
        var_y_sum = float(np.sum((ref_q - y_mean) ** 2))

        group_list = _native_clean(feat)
        variant_q = _native_quantiles(group_list, self.quantile_points)

        # Pearson r, computed manually (mean-center each side, then
        # dot-product over sqrt of sum-of-squares) rather than via
        # pl.corr, which correlates two *columns across rows* — not the
        # paired elements *within* one row's two quantile lists.
        x_centered = variant_q.list.eval(pl.element() - pl.element().mean())
        num = (x_centered * y_centered_lit).list.sum()
        denom_x = x_centered.list.eval(pl.element() ** 2).list.sum()
        denom = (denom_x * var_y_sum).sqrt()
        r = pl.when(denom == 0).then(None).otherwise(num / denom)

        # A group with a single distinct value has an exactly constant
        # quantile profile (quantile(0) == quantile(1) == that value, since
        # the grid includes both endpoints), so its correlation with the
        # reference is mathematically undefined — same case scipy flags via
        # ConstantInputWarning. This is checked on
        # group_list.list.n_unique(), not solely on `denom == 0`, because
        # mean-centering a repeated constant is *not* guaranteed to cancel
        # to bit-exact zero in floating point (e.g. 100 copies of
        # 0.09048978162787422 summed then divided by 100 comes back
        # ~2.8e-17 off), which would otherwise leak a bogus near-zero
        # correlation instead of null — caught directly via a real n=1
        # group during verification.
        is_constant_group = group_list.list.n_unique() <= 1
        r = pl.when(is_constant_group).then(None).otherwise(r)

        result = pl.when(group_list.list.len() == 0).then(None).otherwise(r)
        return result.alias(alias)


class AUROCAggregator(BaseAggregator):
    """
    Computes per-group AUROC against the reference distribution for each
    feature column.

    Variant samples are labelled ``1`` and reference samples ``0``. ``0.5``
    indicates identical distributions; ``1.0`` indicates the variant
    group's values are consistently higher than the reference; ``0.0``
    indicates they are consistently lower. Unlike a typical classification
    AUROC, this value is *not* symmetrized to ``[0.5, 1]`` — it reports
    ``P(variant > reference) + 0.5 * P(variant == reference)`` directly, so
    the sign of separation is preserved in the value itself.
    """

    _stat_suffix = "_AUROC"

    def _aggregate_feature_batch(
        self,
        lf: pl.LazyFrame,
        feature_cols: list[str],
        batch_idx: int,
        total_batches: int,
    ) -> pl.LazyFrame:
        cleaned_ref = _collect_clean_reference_pool(lf, feature_cols)
        exprs = [self._auroc_expr(feat, cleaned_ref[feat]) for feat in feature_cols]
        logging.info(
            "%s chunk %d/%d: %d feature(s) (native)",
            type(self).__name__,
            batch_idx,
            total_batches,
            len(feature_cols),
        )
        return _native_aggregate_feature_batch(lf, self.label_col, feature_cols, exprs)

    def _auroc_expr(self, feat: str, ref_vals: np.ndarray) -> pl.Expr:
        alias = f"{feat}{self._stat_suffix}"
        n_ref = len(ref_vals)
        if n_ref == 0:
            return pl.lit(None, dtype=pl.Float64).alias(alias)

        # Rank-sum (Mann-Whitney U) identity: rank the combined pool with
        # average ranks for ties (matches sklearn.metrics.roc_auc_score's
        # own tie handling — verified across 300 randomized trials,
        # continuous and tied data). concat_list preserves element order,
        # so the group's own ranks are exactly the first n_group entries
        # of the ranked combined list.
        group_list = _native_clean(feat)
        ref_lit = pl.lit(pl.Series([ref_vals.tolist()]))
        combined = pl.concat_list([group_list, ref_lit])
        ranks = combined.list.eval(pl.element().rank(method="average"))
        n_group = group_list.list.len()
        group_rank_sum = ranks.list.slice(0, n_group).list.sum()
        u = group_rank_sum - n_group * (n_group + 1) / 2
        auroc = u / (n_group * n_ref)

        result = pl.when(n_group == 0).then(None).otherwise(auroc)
        return result.alias(alias)


_AGGREGATORS: dict[str, type[BaseAggregator]] = {
    "mean": MeanAggregator,
    "median": MedianAggregator,
    "MAD": MADAggregator,
    "std": StdAggregator,
    "KS": KSAggregator,
    "QQ": QQCorrelationAggregator,
    "AUROC": AUROCAggregator,
}


def aggregate(
    lf: pl.LazyFrame,
    label_col: str,
    aggregator_name: str,
    block_list: Optional[set[str]] = None,
    feature_batch_size: Optional[int] = 500,
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
        ``KS``, ``QQ``, ``AUROC``.
    block_list : set[str] or None
        Aggregated output column names to skip (e.g. ``"f1_KS"``). Blocked
        statistics are not computed and do not appear in the output. Names
        that do not match any aggregated output are silently ignored. Defaults
        to ``None``.
    feature_batch_size : int or None
        If set, split features into chunks of at most this many columns per
        aggregation pass to bound peak memory. Each chunk collects its
        variant and reference rows into numpy arrays, so batching bounds how
        much of that data is materialized in memory at once. Defaults to
        ``500``. Set to ``None`` for a single unbatched pass.

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
            aggregator=KS
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
        ``std``, ``KS``, ``QQ``, ``AUROC``). Required.
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
        ``500``. Set to ``None`` for a single unbatched pass.
    """

    aggregator: str = MISSING
    index_file: Optional[str] = None
    feature_batch_size: Optional[int] = 500


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
