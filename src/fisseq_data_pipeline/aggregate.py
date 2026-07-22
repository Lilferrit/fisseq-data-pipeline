"""Per-variant cell-level feature aggregation strategies.

Defines 7 concrete :class:`BaseAggregator` implementations (mean, median, MAD, std,
KS, QQ, AUROC) and two Hydra entry points: ``fisseq-aggregate`` (standalone,
normalizes to synonymous baseline and attaches metadata) and
``fisseq-aggregate-feature-type`` (Nextflow processes ``AGGREGATE_FEATURE_TYPE`` and
``AGGREGATE_HALF`` — lean per-feature-type aggregation used by the feature-selection
branch). ``fisseq-aggregate-feature-type`` also supports optionally downsampling
control (wildtype) rows before aggregation via ``downsample_wt``/``seed`` — see
:func:`downsample_control`.
"""

import abc
import dataclasses
import logging
import pathlib
from typing import ClassVar, Optional, Union

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
from .utils.variant import classify_variant
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
    """

    aggregator: str = MISSING
    save_normalizer: bool = True
    block_list_file: Optional[str] = None
    compute_impact_score: bool = True


_cs = ConfigStore.instance()
_cs.store(name="aggregate_main", node=AggregateConfig)


def variant_classification(lf: pl.LazyFrame, label_col: str) -> pl.LazyFrame:
    """
    Add a boolean ``CONTROL_COLUMN_NAME`` column to a LazyFrame.

    The added column is ``True`` for rows whose variant label (in ``label_col``)
    encodes a synonymous amino-acid substitution — i.e. the same amino acid
    appears before and after the position — as determined by
    :func:`classify_variant`.

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
        .map_elements(
            lambda v: classify_variant(v) == "Synonymous", return_dtype=pl.Boolean
        )
        .alias(CONTROL_COLUMN_NAME)
    )


class BaseAggregator(abc.ABC):
    """
    Base class for all aggregators.

    Subclasses declare :attr:`_stat_suffix` (e.g. ``"_mean"``, ``"_KS"``) and
    implement :meth:`_feature_expr`, a native Polars list expression for one
    feature. :meth:`aggregate` handles everything else: resolving feature
    columns, building the reference pool (only for :class:`ReferenceBasedAggregator`
    subclasses — see :meth:`_reference_lf`), grouping non-control rows into
    per-label list columns, and assembling the final per-feature
    expressions — entirely in Arrow, no numpy materialization, no
    per-(group, feature) Python loop.

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

    @staticmethod
    def _native_clean(feat: str) -> pl.Expr:
        """
        Drop null, NaN, and Inf entries from a per-label list column for
        ``feat`` before any further native list-based computation.
        ``is_finite()`` on a Float64 element is ``False`` for null/NaN/Inf
        alike (verified against empty, single-value, null-mixed, NaN, and
        Inf cases) — the same filter :meth:`ReferenceBasedAggregator._reference_lf`
        applies to the flat reference-pool columns.
        """
        return pl.col(feat).list.eval(pl.element().filter(pl.element().is_finite()))

    @staticmethod
    def _reference_lf(
        lf: pl.LazyFrame, feature_cols: list[str]
    ) -> Optional[pl.LazyFrame]:
        """
        Reference frame to cross-join before computing :meth:`_feature_expr`,
        or ``None``. Overridden by :class:`ReferenceBasedAggregator`; plain
        aggregators (mean/median/MAD/std) don't need a reference pool at
        all.
        """
        return None

    def _native_aggregate_feature_batch(
        self,
        lf: pl.LazyFrame,
        feature_cols: list[str],
        exprs: list[pl.Expr],
        reference_lf: Optional[pl.LazyFrame],
    ) -> pl.LazyFrame:
        """
        Shared group_by/select boilerplate: group non-control rows by
        ``self.label_col`` into per-label list columns, then select
        ``exprs`` (one aliased native-Polars-expression per feature)
        against those list columns. Stays entirely in Arrow — no Python
        boxing.

        ``reference_lf``, if given, is the single-row
        :meth:`ReferenceBasedAggregator._reference_lf` output; it's
        cross-joined onto the per-label list frame so every variant-group
        row also carries each feature's ``{feat}_ref`` reference list
        column. A single-row cross join only broadcasts the reference row
        onto every existing group row — it does not multiply row count.
        """
        variant_lists = (
            lf.filter(~CONTROL_COLUMN)
            .group_by(self.label_col)
            .agg([pl.col(f) for f in feature_cols])
        )
        if reference_lf is not None:
            variant_lists = variant_lists.join(reference_lf, how="cross")
        return variant_lists.select([self.label_col] + exprs)

    def aggregate(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Compute per-label statistics for every non-block-listed feature.

        Parameters
        ----------
        lf : pl.LazyFrame
            Input LazyFrame containing the label column, a ``CONTROL_COLUMN``
            boolean column, and feature columns.

        Returns
        -------
        pl.LazyFrame
            One row per non-control variant group with computed statistics.
        """
        feature_cols = self._feature_columns(lf)
        logging.info(
            "%s: %d feature(s) to aggregate",
            type(self).__name__,
            len(feature_cols),
        )
        reference_lf = self._reference_lf(lf, feature_cols)
        exprs = [self._feature_expr(f) for f in feature_cols]
        return self._native_aggregate_feature_batch(
            lf, feature_cols, exprs, reference_lf
        )

    @abc.abstractmethod
    def _feature_expr(self, feat: str) -> pl.Expr:
        """
        Native Polars expression computing this aggregator's statistic for
        one feature, evaluated against the per-label list columns (and, for
        :class:`ReferenceBasedAggregator` subclasses, the cross-joined
        ``{feat}_ref`` column) built by :meth:`aggregate`.
        """
        raise NotImplementedError


class ReferenceBasedAggregator(BaseAggregator):
    """
    Base for aggregators that compare each variant group against a shared
    control/reference pool (KS, QQ, AUROC): builds the single-row reference
    frame and lets :meth:`BaseAggregator.aggregate` cross-join it in
    automatically.
    """

    @staticmethod
    def _reference_lf(lf: pl.LazyFrame, feature_cols: list[str]) -> pl.LazyFrame:
        """
        Single-row LazyFrame holding one ``{feat}_ref`` list column per
        feature with the finite control-row values for that feature.

        The reference pool is shared by every variant label (a single
        global control group, not split per-label), so this stays a single
        row and is cross-joined onto the per-label variant-list frame
        rather than collected eagerly: the reference pool and the
        per-group variant lists then live in the same lazy query graph,
        letting the streaming engine manage memory instead of Python
        holding numpy arrays for the whole aggregation call.

        ``is_finite()`` is ``False`` for null/NaN/Inf alike, so filtering
        on it drops all three in one pass — the same set
        :meth:`BaseAggregator._native_clean` drops from per-group list
        columns.
        """
        exprs = [
            pl.col(f).filter(pl.col(f).is_finite()).implode().alias(f"{f}_ref")
            for f in feature_cols
        ]
        return lf.filter(CONTROL_COLUMN).select(exprs)


class MeanAggregator(BaseAggregator):
    """Computes per-group mean for each feature column."""

    _stat_suffix = "_mean"

    def _feature_expr(self, feat: str) -> pl.Expr:
        return self._native_clean(feat).list.mean().alias(f"{feat}{self._stat_suffix}")


class MedianAggregator(BaseAggregator):
    """Computes per-group median for each feature column."""

    _stat_suffix = "_median"

    def _feature_expr(self, feat: str) -> pl.Expr:
        return (
            self._native_clean(feat).list.median().alias(f"{feat}{self._stat_suffix}")
        )


class MADAggregator(BaseAggregator):
    """Computes per-group median absolute deviation (MAD) for each feature column."""

    _stat_suffix = "_MAD"

    def _feature_expr(self, feat: str) -> pl.Expr:
        # Verified against np.median(np.abs(vals - np.median(vals))) for
        # empty, single-value, normal, and NaN/Inf-present cases — matches
        # exactly (see test_mad_native_matches_numpy_with_nan_inf_present).
        return (
            self._native_clean(feat)
            .list.eval((pl.element() - pl.element().median()).abs())
            .list.median()
            .alias(f"{feat}{self._stat_suffix}")
        )


class StdAggregator(BaseAggregator):
    """Computes per-group standard deviation for each feature column."""

    _stat_suffix = "_std"

    def _feature_expr(self, feat: str) -> pl.Expr:
        # Confirmed: list.std(ddof=1) returns null for lists of length < 2
        # (see test_std_native_single_value_group_returns_null).
        return (
            self._native_clean(feat)
            .list.std(ddof=1)
            .alias(f"{feat}{self._stat_suffix}")
        )


class KSAggregator(ReferenceBasedAggregator):
    """
    Computes per-group two-sample Kolmogorov-Smirnov statistics against
    the reference distribution for each feature column.
    """

    _stat_suffix = "_KS"

    def _feature_expr(self, feat: str) -> pl.Expr:
        alias = f"{feat}{self._stat_suffix}"
        ref_list = pl.col(f"{feat}_ref")
        n_ref = ref_list.list.len()

        # Signed-weight cumulative-sum KS statistic: +1/n_group per variant
        # value, -1/n_ref per reference value. Sort the combined values,
        # cumsum the weights, and take the max |cumsum| — but only at the
        # LAST position of each run of tied values (ties must be resolved
        # together, not mid-tie, or a spurious intermediate extremum can
        # exceed the true statistic). Verified against
        # scipy.stats.ks_2samp across 500 randomized trials, including
        # tie-heavy integer data and n=1 groups.
        group_list = self._native_clean(feat)

        g_weight = group_list.list.eval((pl.element() * 0 + 1.0) / pl.element().count())
        ref_weight = ref_list.list.eval((pl.element() * 0 - 1.0) / pl.element().count())
        combined_val = pl.concat_list([group_list, ref_list])
        combined_w = pl.concat_list([g_weight, ref_weight])

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

        result = (
            pl.when(n_ref == 0)
            .then(None)
            .when(group_list.list.len() == 0)
            .then(None)
            .otherwise(ks_stat)
        )
        return result.alias(alias)


class QQCorrelationAggregator(ReferenceBasedAggregator):
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

    @staticmethod
    def _native_quantiles(list_expr: pl.Expr, probs: np.ndarray) -> pl.Expr:
        """
        Per-row quantiles of a list column at fixed probability points
        ``probs``, matching ``np.quantile(..., interpolation="linear")``
        bit-for-bit (verified across 200 randomized trials, including
        single-element and empty lists).

        Computed via one sort + a per-row index gather rather than one
        ``list.eval(...quantile...)`` call per probability point: with 100
        quantile points and hundreds of features per aggregation call, the
        naive one-expression-per-point approach makes the ``.select()``
        call contain ``n_quantiles * n_features`` sub-expressions (50,000
        at the defaults), which measured *slower* than the numpy loop it
        replaces (247.9s vs a 157.4s baseline at 500 labels x 500 features
        x 100 quantiles). This version measured 22.5s at the same scale.

        ``null_on_oob=True`` on the gathers is required, not defensive
        padding: an empty list (``n=0``) produces negative/garbage indices
        after the ``(n - 1)`` scaling, and Polars' ``when/otherwise`` does
        not short-circuit — the ``otherwise`` branch is evaluated for every
        row before the mask is applied, so an out-of-bounds gather raises
        regardless of whether that row's result is later discarded
        (confirmed directly: without ``null_on_oob``, a single empty group
        anywhere in the data crashes the whole query with
        ``OutOfBoundsError``).
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

    def _feature_expr(self, feat: str) -> pl.Expr:
        alias = f"{feat}{self._stat_suffix}"
        ref_list = pl.col(f"{feat}_ref")
        n_ref = ref_list.list.len()

        # Reference-side quantiles and their centering are fixed per
        # feature (the reference pool doesn't vary by group) but are now
        # computed per row via _native_quantiles rather than once in plain
        # numpy, since the reference pool lives in the query graph — every
        # row gets the identical broadcast reference list post-cross-join,
        # so this recomputes the same quantiles redundantly per group
        # rather than once per feature.
        ref_q = self._native_quantiles(ref_list, self.quantile_points)
        # Constant reference quantile profile (e.g. a single distinct
        # reference value) -> correlation is undefined for every group
        # regardless of its own data (scipy.stats.pearsonr would raise
        # ConstantInputWarning and return nan here).
        ref_constant = ref_q.list.max() == ref_q.list.min()
        y_centered = ref_q.list.eval(pl.element() - pl.element().mean())
        var_y_sum = y_centered.list.eval(pl.element() ** 2).list.sum()

        group_list = self._native_clean(feat)
        variant_q = self._native_quantiles(group_list, self.quantile_points)

        # Pearson r, computed manually (mean-center each side, then
        # dot-product over sqrt of sum-of-squares) rather than via
        # pl.corr, which correlates two *columns across rows* — not the
        # paired elements *within* one row's two quantile lists.
        x_centered = variant_q.list.eval(pl.element() - pl.element().mean())
        num = (x_centered * y_centered).list.sum()
        denom_x = x_centered.list.eval(pl.element() ** 2).list.sum()
        denom = (denom_x * var_y_sum).sqrt()

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

        result = (
            pl.when(n_ref == 0)
            .then(None)
            .when(ref_constant)
            .then(None)
            .when(group_list.list.len() == 0)
            .then(None)
            .when(is_constant_group)
            .then(None)
            .when(denom == 0)
            .then(None)
            .otherwise(num / denom)
        )
        return result.alias(alias)


class AUROCAggregator(ReferenceBasedAggregator):
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

    def _feature_expr(self, feat: str) -> pl.Expr:
        alias = f"{feat}{self._stat_suffix}"
        ref_list = pl.col(f"{feat}_ref")
        n_ref = ref_list.list.len()

        # Rank-sum (Mann-Whitney U) identity: rank the combined pool with
        # average ranks for ties (matches sklearn.metrics.roc_auc_score's
        # own tie handling — verified across 300 randomized trials,
        # continuous and tied data). concat_list preserves element order,
        # so the group's own ranks are exactly the first n_group entries
        # of the ranked combined list.
        group_list = self._native_clean(feat)
        combined = pl.concat_list([group_list, ref_list])
        ranks = combined.list.eval(pl.element().rank(method="average"))
        n_group = group_list.list.len()
        group_rank_sum = ranks.list.slice(0, n_group).list.sum()
        u = group_rank_sum - n_group * (n_group + 1) / 2
        auroc = u / (n_group * n_ref)

        result = (
            pl.when(n_ref == 0)
            .then(None)
            .when(n_group == 0)
            .then(None)
            .otherwise(auroc)
        )
        return result.alias(alias)


def downsample_control(
    lf: pl.LazyFrame, downsample_wt: Union[float, int], seed: int
) -> pl.LazyFrame:
    """
    Reproducibly downsample control (wildtype) rows to a target size.

    Non-control rows are left untouched. Selection is deterministic given
    ``seed``: each control row gets a seeded hash of a fresh row index, rows
    are ranked by that hash, and the lowest ``target`` ranks are kept — the
    same lazy hash-and-rank idiom used by
    :func:`fisseq_data_pipeline.input.add_downsampled_pseudo_variants`, which
    avoids collecting the full control pool up front and reproduces exactly
    across runs for a fixed seed.

    Parameters
    ----------
    lf : pl.LazyFrame
        Cell-level LazyFrame containing a boolean ``CONTROL_COLUMN`` column.
    downsample_wt : float or int
        Target control-pool size. A float in ``(0, 1)`` is interpreted as
        the fraction of control rows to keep; an int is interpreted as an
        absolute target count (a no-op if the control pool is already at or
        below the target).
    seed : int
        Random seed for the downsample draw.

    Returns
    -------
    pl.LazyFrame
        ``lf`` with control rows downsampled to the target size.
    """
    control = lf.filter(CONTROL_COLUMN).with_row_index("_tmp_row_idx")
    non_control = lf.filter(~CONTROL_COLUMN)

    ranked = control.with_columns(
        pl.col("_tmp_row_idx").hash(seed=seed).rank(method="ordinal").alias("_rank"),
    )
    if isinstance(downsample_wt, float):
        target = (pl.len() * downsample_wt).floor()
    else:
        target = pl.lit(downsample_wt)

    downsampled_control = ranked.filter(pl.col("_rank") <= target).drop(
        ["_tmp_row_idx", "_rank"]
    )
    return pl.concat([non_control, downsampled_control])


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
    return agg.aggregate(lf)


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
    downsample_wt : float, int, or None
        Optional downsampling of control (wildtype) rows before aggregation.
        A float in ``(0, 1)`` keeps that fraction of control rows; an int
        keeps that many. ``None`` disables downsampling. Defaults to
        ``None``.
    seed : int
        Random seed for the ``downsample_wt`` draw. Ignored when
        ``downsample_wt`` is ``None``. Defaults to ``0``.
    """

    aggregator: str = MISSING
    index_file: Optional[str] = None
    downsample_wt: Optional[Union[float, int]] = None
    seed: int = 0


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
            index_file=./half1.parquet \\
            downsample_wt=0.5 \\
            seed=1
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

    if ft_cfg.downsample_wt is not None:
        if isinstance(ft_cfg.downsample_wt, float) and not (
            0 < ft_cfg.downsample_wt < 1
        ):
            raise ValueError(
                f"downsample_wt float must satisfy 0 < x < 1, got {ft_cfg.downsample_wt}"
            )
        if isinstance(ft_cfg.downsample_wt, int) and ft_cfg.downsample_wt <= 0:
            raise ValueError(
                f"downsample_wt int must be positive, got {ft_cfg.downsample_wt}"
            )
        logging.info(
            "Downsampling control rows: downsample_wt=%s, seed=%d",
            ft_cfg.downsample_wt,
            ft_cfg.seed,
        )
        lf = downsample_control(lf, ft_cfg.downsample_wt, ft_cfg.seed)

    logging.info("Running %s aggregator", ft_cfg.aggregator)
    agg_lf = aggregate(
        lf,
        label_col=ft_cfg.label_column,
        aggregator_name=ft_cfg.aggregator,
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
