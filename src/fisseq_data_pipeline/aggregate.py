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
from omegaconf import DictConfig, OmegaConf

from .config import LabeledInputConfig
from .constants import CONTROL_COLUMN, CONTROL_COLUMN_NAME, FEATURE_SELECTOR
from .normalize import Normalizer
from .utils import (
    compute_impact_score,
    get_aggregate_meta_data,
    load_batches,
    setup_logging,
)


@dataclasses.dataclass
class AggregateConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the aggregation entry point.

    Attributes
    ----------
    aggregator : str
        Aggregation method. One of: ``mean``, ``median``, ``MAD``, ``std``,
        ``EMD``, ``KS``, ``QQ``, ``AUROC``. Defaults to ``"multi"``.
    save_normalizer : bool
        If ``True``, persist the fitted :class:`.normalize.Normalizer` alongside
        the output. Defaults to ``True``.
    block_list_file : str or None
        Optional path to a parquet file with at least ``feature`` (str) and
        ``feature_ok`` (bool) columns. Features where ``feature_ok`` is
        ``False`` are excluded from aggregation. Defaults to ``None`` (no
        features blocked).
    """

    aggregator: str = "multi"
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


class NativeAggregator(BaseAggregator):
    """
    Base for aggregators that express their statistic as a single Polars expression.

    Subclasses declare :attr:`_stat_suffix` and implement :meth:`_expr`. The
    boilerplate of filtering controls, grouping, filtering the block list, and
    collecting feature columns is handled here.
    """

    _stat_suffix: ClassVar[str]

    @abc.abstractmethod
    def _expr(self, feat: str) -> pl.Expr:
        raise NotImplementedError

    def aggregate(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        feature_cols = [
            f
            for f in lf.select(FEATURE_SELECTOR).collect_schema().names()
            if f"{f}{self._stat_suffix}" not in (self.block_list or set())
        ]
        return (
            lf.filter(~CONTROL_COLUMN)
            .group_by(self.label_col)
            .agg([self._expr(f) for f in feature_cols])
        )


class MeanAggregator(NativeAggregator):
    """Computes per-group mean for each feature column."""

    _stat_suffix = "_mean"

    def _expr(self, feat: str) -> pl.Expr:
        return pl.col(feat).mean().alias(f"{feat}_mean")


class MedianAggregator(NativeAggregator):
    """Computes per-group median for each feature column."""

    _stat_suffix = "_median"

    def _expr(self, feat: str) -> pl.Expr:
        return pl.col(feat).median().alias(f"{feat}_median")


class MADAggregator(NativeAggregator):
    """Computes per-group median absolute deviation (MAD) for each feature column."""

    _stat_suffix = "_MAD"

    def _expr(self, feat: str) -> pl.Expr:
        return (
            (pl.col(feat) - pl.col(feat).median()).abs().median().alias(f"{feat}_MAD")
        )


class StdAggregator(NativeAggregator):
    """Computes per-group standard deviation for each feature column."""

    _stat_suffix = "_std"

    def _expr(self, feat: str) -> pl.Expr:
        return pl.col(feat).std().alias(f"{feat}_std")


class ReferenceBasedAggregator(BaseAggregator):
    """
    Base for aggregators that compare each variant group against a reference
    distribution using a scalar per-feature statistic.

    Subclasses declare :attr:`_stat_suffix` (e.g. ``"_EMD"``) and implement
    :meth:`_compute_feature_stat`. The boilerplate of checking for a reference
    DataFrame, filtering controls, dispatching parallel tasks, null-guarding,
    and building the output DataFrame is handled here.
    """

    _stat_suffix: ClassVar[str]

    def aggregate(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        all_cols = lf.collect_schema().names()
        feature_cols = [
            f
            for f in lf.select(FEATURE_SELECTOR).collect_schema().names()
            if f"{f}{self._stat_suffix}" not in (self.block_list or set())
        ]
        variant_labels = lf.filter(~CONTROL_COLUMN).select(self.label_col).unique()
        controls = lf.filter(CONTROL_COLUMN).drop(self.label_col)
        expanded = pl.concat(
            [
                lf.filter(~CONTROL_COLUMN),
                controls.join(variant_labels, how="cross").select(all_cols),
            ]
        )
        # Group each label's rows (variant + replicated controls) into lists so
        # that map_elements below can split them by the control flag.
        grouped = expanded.group_by(self.label_col).agg(
            [pl.col(f) for f in feature_cols] + [CONTROL_COLUMN]
        )
        compute = self._compute_feature_stat
        stat_exprs = []
        for feat in feature_cols:

            def _stat_fn(
                s: dict,
                _feat: str = feat,
                _fn=compute,
            ) -> Optional[float]:
                is_ctrl = s[CONTROL_COLUMN_NAME]
                vals = s[_feat]
                group_vals = np.fromiter(
                    (
                        v
                        for v, c in zip(vals, is_ctrl)
                        if not c and v is not None and np.isfinite(v)
                    ),
                    dtype=float,
                )
                ref_vals = np.fromiter(
                    (
                        v
                        for v, c in zip(vals, is_ctrl)
                        if c and v is not None and np.isfinite(v)
                    ),
                    dtype=float,
                )
                if len(group_vals) == 0 or len(ref_vals) == 0:
                    return None
                result = _fn(group_vals, ref_vals)
                return float(result) if np.isfinite(result) else None

            stat_exprs.append(
                pl.struct([pl.col(feat), CONTROL_COLUMN])
                .map_elements(_stat_fn, return_dtype=pl.Float64)
                .alias(f"{feat}{self._stat_suffix}")
            )
        return grouped.with_columns(stat_exprs).select(
            [self.label_col] + [f"{f}{self._stat_suffix}" for f in feature_cols]
        )

    @abc.abstractmethod
    def _compute_feature_stat(
        self, group_vals: np.ndarray, ref_vals: np.ndarray
    ) -> float:
        raise NotImplementedError


class EMDAggregator(ReferenceBasedAggregator):
    """
    Computes per-group 1D Wasserstein distances (Earth Mover's Distance)
    against the reference distribution for each feature column.

    Requires ``reference_df`` to be set at construction time.
    """

    _stat_suffix = "_EMD"

    def _compute_feature_stat(
        self, group_vals: np.ndarray, ref_vals: np.ndarray
    ) -> float:
        return scipy.stats.wasserstein_distance(group_vals, ref_vals)


class KSAggregator(ReferenceBasedAggregator):
    """
    Computes per-group two-sample Kolmogorov-Smirnov statistics against
    the reference distribution for each feature column.

    Requires ``reference_df`` to be set at construction time.
    """

    _stat_suffix = "_KS"

    def _compute_feature_stat(
        self, group_vals: np.ndarray, ref_vals: np.ndarray
    ) -> float:
        return scipy.stats.ks_2samp(group_vals, ref_vals).statistic


class QQCorrelationAggregator(ReferenceBasedAggregator):
    """
    Computes per-group Q-Q correlation against the reference distribution for
    each feature column.

    Requires ``reference_df`` to be set at construction time.

    Parameters
    ----------
    reference_df : pl.DataFrame or None
        Reference DataFrame (typically control rows).
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

    def _compute_feature_stat(
        self, group_vals: np.ndarray, ref_vals: np.ndarray
    ) -> float:
        variant_q = np.quantile(group_vals, self.quantile_points)
        reference_q = np.quantile(ref_vals, self.quantile_points)
        return scipy.stats.pearsonr(variant_q, reference_q).statistic


class AUROCAggregator(ReferenceBasedAggregator):
    """
    Computes per-group AUROC against the reference distribution for each
    feature column.

    Requires ``reference_df`` to be set at construction time.

    Variant samples are labelled ``1`` and reference samples ``0``.
    ``0.5`` indicates identical distributions; ``1.0`` indicates perfect
    separability.
    """

    _stat_suffix = "_AUROC"

    def _compute_feature_stat(
        self, group_vals: np.ndarray, ref_vals: np.ndarray
    ) -> float:
        values = np.concatenate([ref_vals, group_vals])
        labels = np.concatenate([np.zeros(len(ref_vals)), np.ones(len(group_vals))])
        auroc = sklearn.metrics.roc_auc_score(labels, values)
        if auroc < 0.5:
            auroc = 1 - auroc
        return auroc


class MultiAggregator(BaseAggregator):
    """
    Applies a list of aggregators and joins their results on the label column.

    Each sub-aggregator produces a per-label DataFrame. The results are joined
    sequentially on ``label_col`` so the label column appears exactly once in
    the output.

    Parameters
    ----------
    aggregators : list[BaseAggregator]
        Pre-configured aggregator instances to run. Must share the same
        ``label_col``.
    """

    def __init__(self, aggregators: list[BaseAggregator]) -> None:
        label_col = aggregators[0].label_col if aggregators else "meta_aa_changes"
        super().__init__(label_col=label_col)
        self.aggregators = aggregators

    def aggregate(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        if not self.aggregators:
            raise ValueError("MultiAggregator requires at least one aggregator")
        results = [agg.aggregate(lf) for agg in self.aggregators]
        combined = results[0]
        for other in results[1:]:
            combined = combined.join(other, on=self.label_col, how="inner")
        return combined


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

_MULTI_DEFAULT: list[str] = ["mean", "median", "MAD", "std", "KS", "QQ", "AUROC"]


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
        ``EMD``, ``KS``, ``QQ``, ``AUROC``, or ``multi`` (all except EMD).
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
    valid = set(_AGGREGATORS) | {"multi"}
    if aggregator_name not in valid:
        raise ValueError(
            f"Unknown aggregator {aggregator_name!r}. Choose from: {sorted(valid)}"
        )

    if aggregator_name == "multi":
        sub_aggs = [
            _AGGREGATORS[name](label_col=label_col, block_list=block_list)
            for name in _MULTI_DEFAULT
        ]
        agg: BaseAggregator = MultiAggregator(sub_aggs)
    else:
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


if __name__ == "__main__":
    main()
