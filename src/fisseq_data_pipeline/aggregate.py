import abc
import dataclasses
import logging
import pathlib
import re
from typing import Any, Optional

import hydra
import joblib
import numpy as np
import polars as pl
import scipy.stats
import sklearn.metrics
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .constants import CONTROL_COLUMN, CONTROL_COLUMN_NAME, FEATURE_SELECTOR
from .normalize import Normalizer
from .utils import setup_logging


@dataclasses.dataclass
class AggregateConfig(AppConfig):
    """
    Hydra structured configuration for the aggregation entry point.

    Attributes
    ----------
    input_file : str
        Path to the input parquet file (cell-level normalized data). Required.
    label_column : str
        Name of the column identifying variant labels. Defaults to
        ``"meta_aa_changes"``. Used for group-by and synonymous classification.
    aggregator : str
        Aggregation method. One of: ``mean``, ``median``, ``MAD``, ``std``,
        ``EMD``, ``KS``, ``QQ``, ``AUROC``. Defaults to ``"EMD"``.
    save_normalizer : bool
        If ``True``, persist the fitted :class:`.normalize.Normalizer` alongside
        the output. Defaults to ``True``.
    """

    input_file: str = MISSING
    label_column: str = "meta_aa_changes"
    aggregator: str = "multi"
    save_normalizer: bool = True


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

    The constructor stores an optional reference DataFrame and label column
    name. Subclasses that require ``reference_df`` raise :exc:`ValueError`
    inside :meth:`aggregate` when it is ``None``.

    Parameters
    ----------
    reference_df : pl.DataFrame or None
        Optional reference (control) DataFrame. Reference-based aggregators
        require this; native aggregators ignore it.
    label_col : str
        Name of the column used to identify variant groups. Defaults to
        ``"meta_aa_changes"``.
    """

    def __init__(
        self,
        reference_df: Optional[pl.DataFrame] = None,
        label_col: str = "meta_aa_changes",
    ) -> None:
        self.reference_df = reference_df
        self.label_col = label_col

    @abc.abstractmethod
    def aggregate(self, agg_df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute per-label statistics.

        Parameters
        ----------
        agg_df : pl.DataFrame
            Input DataFrame containing the label column, a ``CONTROL_COLUMN``
            boolean column, and feature columns.

        Returns
        -------
        pl.DataFrame
            One row per non-control variant group with computed statistics.
        """
        raise NotImplementedError


class MeanAggregator(BaseAggregator):
    """Computes per-group mean for each feature column."""

    def aggregate(self, agg_df: pl.DataFrame) -> pl.DataFrame:
        feature_cols = agg_df.select(FEATURE_SELECTOR).columns
        return (
            agg_df.filter(~CONTROL_COLUMN)
            .group_by(self.label_col)
            .agg([pl.col(f).mean().alias(f"{f}_mean") for f in feature_cols])
        )


class MedianAggregator(BaseAggregator):
    """Computes per-group median for each feature column."""

    def aggregate(self, agg_df: pl.DataFrame) -> pl.DataFrame:
        feature_cols = agg_df.select(FEATURE_SELECTOR).columns
        return (
            agg_df.filter(~CONTROL_COLUMN)
            .group_by(self.label_col)
            .agg([pl.col(f).median().alias(f"{f}_median") for f in feature_cols])
        )


class MADAggregator(BaseAggregator):
    """Computes per-group median absolute deviation (MAD) for each feature column."""

    def aggregate(self, agg_df: pl.DataFrame) -> pl.DataFrame:
        feature_cols = agg_df.select(FEATURE_SELECTOR).columns
        return (
            agg_df.filter(~CONTROL_COLUMN)
            .group_by(self.label_col)
            .agg(
                [
                    (pl.col(f) - pl.col(f).median()).abs().median().alias(f"{f}_MAD")
                    for f in feature_cols
                ]
            )
        )


class StdAggregator(BaseAggregator):
    """Computes per-group standard deviation for each feature column."""

    def aggregate(self, agg_df: pl.DataFrame) -> pl.DataFrame:
        feature_cols = agg_df.select(FEATURE_SELECTOR).columns
        return (
            agg_df.filter(~CONTROL_COLUMN)
            .group_by(self.label_col)
            .agg([pl.col(f).std().alias(f"{f}_std") for f in feature_cols])
        )


class EMDAggregator(BaseAggregator):
    """
    Computes per-group 1D Wasserstein distances (Earth Mover's Distance)
    against the reference distribution for each feature column.

    Requires ``reference_df`` to be set at construction time.
    """

    def aggregate(
        self,
        agg_df: pl.DataFrame,
        n_jobs: int = -1,
        backend: str = "threading",
        verbose: int = 0,
    ) -> pl.DataFrame:
        if self.reference_df is None:
            raise ValueError("EMDAggregator requires a reference_df")
        feature_cols = self.reference_df.select(FEATURE_SELECTOR).columns
        groups = agg_df.filter(~CONTROL_COLUMN).group_by(self.label_col)
        tasks = [(keys[0], group_df) for keys, group_df in groups]
        dicts = joblib.Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
            joblib.delayed(self._compute_statistic)(label, group_df, feature_cols)
            for label, group_df in tasks
        )
        return pl.DataFrame(dicts)

    def _compute_statistic(
        self, label: str, group_df: pl.DataFrame, feature_cols: list[str]
    ) -> dict[str, Any]:
        row: dict[str, Any] = {self.label_col: label}
        for feat in feature_cols:
            row[f"{feat}_EMD"] = scipy.stats.wasserstein_distance(
                group_df.get_column(feat).to_numpy(),
                self.reference_df.get_column(feat).to_numpy(),
            )
        return row


class KSAggregator(BaseAggregator):
    """
    Computes per-group two-sample Kolmogorov-Smirnov statistics against
    the reference distribution for each feature column.

    Requires ``reference_df`` to be set at construction time.
    """

    def aggregate(
        self,
        agg_df: pl.DataFrame,
        n_jobs: int = -1,
        backend: str = "threading",
        verbose: int = 0,
    ) -> pl.DataFrame:
        if self.reference_df is None:
            raise ValueError("KSAggregator requires a reference_df")
        feature_cols = self.reference_df.select(FEATURE_SELECTOR).columns
        groups = agg_df.filter(~CONTROL_COLUMN).group_by(self.label_col)
        tasks = [(keys[0], group_df) for keys, group_df in groups]
        dicts = joblib.Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
            joblib.delayed(self._compute_statistic)(label, group_df, feature_cols)
            for label, group_df in tasks
        )
        return pl.DataFrame(dicts)

    def _compute_statistic(
        self, label: str, group_df: pl.DataFrame, feature_cols: list[str]
    ) -> dict[str, Any]:
        row: dict[str, Any] = {self.label_col: label}
        for feat in feature_cols:
            row[f"{feat}_KS"] = scipy.stats.ks_2samp(
                group_df.get_column(feat).to_numpy(),
                self.reference_df.get_column(feat).to_numpy(),
            ).statistic
        return row


class QQCorrelationAggregator(BaseAggregator):
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

    def __init__(
        self,
        reference_df: Optional[pl.DataFrame] = None,
        label_col: str = "meta_aa_changes",
        n_quantiles: int = 100,
    ) -> None:
        super().__init__(reference_df, label_col)
        self.quantile_points = np.linspace(0, 1, n_quantiles)

    def aggregate(
        self,
        agg_df: pl.DataFrame,
        n_jobs: int = -1,
        backend: str = "threading",
        verbose: int = 0,
    ) -> pl.DataFrame:
        if self.reference_df is None:
            raise ValueError("QQCorrelationAggregator requires a reference_df")
        feature_cols = self.reference_df.select(FEATURE_SELECTOR).columns
        groups = agg_df.filter(~CONTROL_COLUMN).group_by(self.label_col)
        tasks = [(keys[0], group_df) for keys, group_df in groups]
        dicts = joblib.Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
            joblib.delayed(self._compute_statistic)(label, group_df, feature_cols)
            for label, group_df in tasks
        )
        return pl.DataFrame(dicts)

    def _compute_statistic(
        self, label: str, group_df: pl.DataFrame, feature_cols: list[str]
    ) -> dict[str, Any]:
        row: dict[str, Any] = {self.label_col: label}
        for feat in feature_cols:
            variant_q = np.quantile(
                group_df.get_column(feat).to_numpy(), self.quantile_points
            )
            reference_q = np.quantile(
                self.reference_df.get_column(feat).to_numpy(), self.quantile_points
            )
            row[f"{feat}_QQ"] = scipy.stats.pearsonr(variant_q, reference_q).statistic
        return row


class AUROCAggregator(BaseAggregator):
    """
    Computes per-group AUROC against the reference distribution for each
    feature column.

    Requires ``reference_df`` to be set at construction time.

    Variant samples are labelled ``1`` and reference samples ``0``.
    ``0.5`` indicates identical distributions; ``1.0`` indicates perfect
    separability.
    """

    def aggregate(
        self,
        agg_df: pl.DataFrame,
        n_jobs: int = -1,
        backend: str = "threading",
        verbose: int = 0,
    ) -> pl.DataFrame:
        if self.reference_df is None:
            raise ValueError("AUROCAggregator requires a reference_df")
        feature_cols = self.reference_df.select(FEATURE_SELECTOR).columns
        groups = agg_df.filter(~CONTROL_COLUMN).group_by(self.label_col)
        tasks = [(keys[0], group_df) for keys, group_df in groups]
        dicts = joblib.Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
            joblib.delayed(self._compute_statistic)(label, group_df, feature_cols)
            for label, group_df in tasks
        )
        return pl.DataFrame(dicts)

    def _compute_statistic(
        self, label: str, group_df: pl.DataFrame, feature_cols: list[str]
    ) -> dict[str, Any]:
        row: dict[str, Any] = {self.label_col: label}
        for feat in feature_cols:
            variant = group_df.get_column(feat).to_numpy()
            reference = self.reference_df.get_column(feat).to_numpy()
            values = np.concatenate([reference, variant])
            labels = np.concatenate([np.zeros(len(reference)), np.ones(len(variant))])
            auroc = sklearn.metrics.roc_auc_score(labels, values)
            if auroc < 0.5:
                auroc = 1 - auroc
            row[f"{feat}_AUROC"] = auroc
        return row


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
        super().__init__(reference_df=None, label_col=label_col)
        self.aggregators = aggregators

    def aggregate(self, agg_df: pl.DataFrame) -> pl.DataFrame:
        if not self.aggregators:
            raise ValueError("MultiAggregator requires at least one aggregator")
        results = [agg.aggregate(agg_df) for agg in self.aggregators]
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
) -> pl.DataFrame:
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

    Returns
    -------
    pl.DataFrame
        One row per non-control variant group with computed statistics.
    """
    valid = set(_AGGREGATORS) | {"multi"}
    if aggregator_name not in valid:
        raise ValueError(
            f"Unknown aggregator {aggregator_name!r}. Choose from: {sorted(valid)}"
        )

    df = lf.collect()
    control_df = df.filter(CONTROL_COLUMN)

    if aggregator_name == "multi":
        sub_aggs = [
            _AGGREGATORS[name](control_df, label_col=label_col)
            for name in _MULTI_DEFAULT
        ]
        agg: BaseAggregator = MultiAggregator(sub_aggs)
    else:
        agg = _AGGREGATORS[aggregator_name](control_df, label_col=label_col)

    return agg.aggregate(df)


@hydra.main(version_base=None, config_path=None, config_name="aggregate_main")
def main(cfg: DictConfig) -> None:
    """
    Aggregate cell-level features and z-score normalize to synonymous baseline.

    Reads the input file at ``input_file``, runs the configured aggregator to
    produce one row per variant, marks synonymous variants as the normalization
    reference via :func:`variant_classification`, fits a
    :class:`.normalize.Normalizer` on those rows, applies it, and writes the
    result.

    Output path
    -----------
    - If ``output_root`` is set: ``{output_root}.{stem}.{ext}``
    - Otherwise: ``{output_dir}/{filename}`` (same name as the input file)

    If ``save_normalizer`` is ``True``, the fitted :class:`.normalize.Normalizer`
    is also written using the same root/dir convention with the name
    ``normalizer.parquet``.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.aggregate \\
            output_dir=./out \\
            input_file=data/cells_normalized.parquet \\
            aggregator=EMD
    """
    agg_cfg: AggregateConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(agg_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agg_cfg.output_dir = output_dir
    setup_logging(agg_cfg, "aggregate")

    input_path = pathlib.Path(agg_cfg.input_file)
    logging.info("Loading input from %s", input_path)
    lf = pl.scan_parquet(input_path)

    logging.info("Running %s aggregator", agg_cfg.aggregator)
    agg_df = aggregate(
        lf, label_col=agg_cfg.label_column, aggregator_name=agg_cfg.aggregator
    )

    logging.info(
        "Classifying variants and marking synonymous as normalization reference"
    )
    agg_lf = variant_classification(agg_df.lazy(), agg_cfg.label_column)

    logging.info("Fitting normalizer on synonymous rows")
    normalizer = Normalizer.from_lazyframe(agg_lf, fit_only_on_control=True)

    logging.info("Applying normalizer")
    normalized_lf = normalizer.apply(agg_lf)

    stem = input_path.stem
    ext = input_path.suffix.lstrip(".")
    if agg_cfg.output_root is not None:
        out_path = pathlib.Path(f"{agg_cfg.output_root}.{stem}.{ext}")
    else:
        out_path = output_dir / input_path.name

    logging.info("Writing output to %s", out_path)
    normalized_lf.collect().write_parquet(out_path)

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
