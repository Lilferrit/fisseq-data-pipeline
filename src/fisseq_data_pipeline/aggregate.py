import abc
import dataclasses
import logging
import pathlib
import re
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
    """

    aggregator: str = MISSING
    save_normalizer: bool = True
    block_list_file: Optional[str] = None
    compute_impact_score: bool = True


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


class BaseAggregator(abc.ABC):
    """
    Abstract base class for all aggregators.

    Parameters
    ----------
    label_col : str
        Name of the column used to identify variant groups. Defaults to
        ``"meta_aa_changes"``.
    block_list : set[str] or None
        Aggregated output column names to skip (e.g. ``"f1_EMD"``). Blocked
        statistics are not computed. Defaults to ``None``.
    """

    def __init__(
        self,
        label_col: str = "meta_aa_changes",
        block_list: Optional[set[str]] = None,
    ) -> None:
        self.label_col = label_col
        self.block_list = block_list

    @abc.abstractmethod
    def aggregate(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Compute per-label statistics.

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
        raise NotImplementedError


def _reference_alias(feat: str) -> str:
    """Column name for a feature's label-independent reference-pool list."""
    return f"__reference__{feat}"


def _variant_reference_lists(
    lf: pl.LazyFrame, label_col: str, feature_cols: list[str]
) -> pl.LazyFrame:
    """
    Build one row per variant label with, for each feature, a list column of
    that label's own (non-control) values and a label-independent list
    column of the full control/reference pool for that feature.

    The reference pool is shared by every variant label (a single global
    control group, not split per-label), so it is aggregated once and
    cross-joined in rather than replicated per label before grouping.
    """
    variant_lists = (
        lf.filter(~CONTROL_COLUMN)
        .group_by(label_col)
        .agg([pl.col(f) for f in feature_cols])
    )
    reference_lists = lf.filter(CONTROL_COLUMN).select(
        [pl.col(f).implode().alias(_reference_alias(f)) for f in feature_cols]
    )
    return variant_lists.join(reference_lists, how="cross")


def _clean(values: Optional[list]) -> np.ndarray:
    """Drop None and non-finite entries from a struct-unpacked value list."""
    return np.fromiter(
        (v for v in (values or []) if v is not None and np.isfinite(v)), dtype=float
    )


def _finalize(result: float) -> Optional[float]:
    """None if non-finite, else a plain python float."""
    return float(result) if np.isfinite(result) else None


class FeatureStatAggregator(BaseAggregator):
    """
    Base for aggregators that compute one scalar statistic per (label,
    feature) pair from that label's variant values and the shared reference
    (control) pool.

    Subclasses declare :attr:`_stat_suffix` (e.g. ``"_mean"``, ``"_EMD"``) and
    implement :meth:`_compute_feature_stat`. The boilerplate of grouping
    variant values and the reference pool into list columns, filtering the
    block list, and constructing the per-feature query is handled here.
    """

    _stat_suffix: ClassVar[str]

    @abc.abstractmethod
    def _compute_feature_stat(self, values: dict) -> Optional[float]:
        """
        Compute a single statistic from ``values``, a dict with keys
        ``"variant"`` and ``"reference"`` each mapping to a list of feature
        values (possibly containing ``None``). Stats that don't need the
        reference pool (e.g. mean, median) simply ignore that key.
        """
        raise NotImplementedError

    def _feature_columns(self, lf: pl.LazyFrame) -> list[str]:
        return [
            f
            for f in lf.select(FEATURE_SELECTOR).collect_schema().names()
            if f"{f}{self._stat_suffix}" not in (self.block_list or set())
        ]

    def _stat_expr(self, feat: str) -> pl.Expr:
        return (
            pl.struct(
                [
                    pl.col(feat).alias("variant"),
                    pl.col(_reference_alias(feat)).alias("reference"),
                ]
            )
            .map_elements(self._compute_feature_stat, return_dtype=pl.Float64)
            .alias(f"{feat}{self._stat_suffix}")
        )

    def aggregate(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        feature_cols = self._feature_columns(lf)
        base = _variant_reference_lists(lf, self.label_col, feature_cols)
        return base.select(
            [self.label_col] + [self._stat_expr(f) for f in feature_cols]
        )


class MeanAggregator(FeatureStatAggregator):
    """Computes per-group mean for each feature column."""

    _stat_suffix = "_mean"

    def _compute_feature_stat(self, values: dict) -> Optional[float]:
        vals = _clean(values["variant"])
        if len(vals) == 0:
            return None
        return _finalize(np.mean(vals))


class MedianAggregator(FeatureStatAggregator):
    """Computes per-group median for each feature column."""

    _stat_suffix = "_median"

    def _compute_feature_stat(self, values: dict) -> Optional[float]:
        vals = _clean(values["variant"])
        if len(vals) == 0:
            return None
        return _finalize(np.median(vals))


class MADAggregator(FeatureStatAggregator):
    """Computes per-group median absolute deviation (MAD) for each feature column."""

    _stat_suffix = "_MAD"

    def _compute_feature_stat(self, values: dict) -> Optional[float]:
        vals = _clean(values["variant"])
        if len(vals) == 0:
            return None
        return _finalize(np.median(np.abs(vals - np.median(vals))))


class StdAggregator(FeatureStatAggregator):
    """Computes per-group standard deviation for each feature column."""

    _stat_suffix = "_std"

    def _compute_feature_stat(self, values: dict) -> Optional[float]:
        vals = _clean(values["variant"])
        if len(vals) < 2:
            return None
        return _finalize(np.std(vals, ddof=1))


class EMDAggregator(FeatureStatAggregator):
    """
    Computes per-group 1D Wasserstein distances (Earth Mover's Distance)
    against the reference distribution for each feature column.
    """

    _stat_suffix = "_EMD"

    def _compute_feature_stat(self, values: dict) -> Optional[float]:
        group_vals = _clean(values["variant"])
        ref_vals = _clean(values["reference"])
        if len(group_vals) == 0 or len(ref_vals) == 0:
            return None
        return _finalize(scipy.stats.wasserstein_distance(group_vals, ref_vals))


class KSAggregator(FeatureStatAggregator):
    """
    Computes per-group two-sample Kolmogorov-Smirnov statistics against
    the reference distribution for each feature column.
    """

    _stat_suffix = "_KS"

    def _compute_feature_stat(self, values: dict) -> Optional[float]:
        group_vals = _clean(values["variant"])
        ref_vals = _clean(values["reference"])
        if len(group_vals) == 0 or len(ref_vals) == 0:
            return None
        return _finalize(scipy.stats.ks_2samp(group_vals, ref_vals).statistic)


class QQCorrelationAggregator(FeatureStatAggregator):
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

    def _compute_feature_stat(self, values: dict) -> Optional[float]:
        group_vals = _clean(values["variant"])
        ref_vals = _clean(values["reference"])
        if len(group_vals) == 0 or len(ref_vals) == 0:
            return None
        variant_q = np.quantile(group_vals, self.quantile_points)
        reference_q = np.quantile(ref_vals, self.quantile_points)
        return _finalize(scipy.stats.pearsonr(variant_q, reference_q).statistic)


class AUROCAggregator(FeatureStatAggregator):
    """
    Computes per-group AUROC against the reference distribution for each
    feature column.

    Variant samples are labelled ``1`` and reference samples ``0``.
    ``0.5`` indicates identical distributions; ``1.0`` indicates perfect
    separability.
    """

    _stat_suffix = "_AUROC"

    def _compute_feature_stat(self, values: dict) -> Optional[float]:
        group_vals = _clean(values["variant"])
        ref_vals = _clean(values["reference"])
        if len(group_vals) == 0 or len(ref_vals) == 0:
            return None
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
    """

    aggregator: str = MISSING
    index_file: Optional[str] = None


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
        lf, label_col=ft_cfg.label_column, aggregator_name=ft_cfg.aggregator
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
