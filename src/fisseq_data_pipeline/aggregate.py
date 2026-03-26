import abc
import logging
import pathlib
import re
from os import PathLike
from typing import Any, Optional

import fire
import joblib
import numpy as np
import polars as pl
import scipy.stats
import sklearn.metrics

from .normalize import Normalizer, fit_normalizer, normalize
from .utils import get_feature_cols, setup_logging


class BaseAggregator(abc.ABC):
    """
    Abstract base class for all aggregators.

    All subclasses expose a single ``aggregate(agg_df)`` method that accepts
    an input DataFrame and returns a per-(batch, label) summary DataFrame.
    """

    @abc.abstractmethod
    def aggregate(self, agg_df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute per-(batch, label) statistics.

        Parameters
        ----------
        agg_df : pl.DataFrame
            Input DataFrame containing ``_meta_batch``, ``_meta_label``, and
            feature columns.

        Returns
        -------
        pl.DataFrame
            One row per (batch, label) group with computed statistics.
        """
        raise NotImplementedError


class ReferenceBaseAggregator(BaseAggregator):
    """
    Base class for aggregators that compare each group against a reference
    distribution, typically built from control rows.

    Subclasses override :meth:`compute_statistic` to implement a specific
    scalar summary (e.g. EMD, KS statistic) for each feature.

    Parameters
    ----------
    reference_df : pl.DataFrame
        Reference DataFrame (typically control rows) used as the comparison
        distribution in :meth:`compute_statistic`.
    """

    def __init__(self, reference_df: pl.DataFrame) -> None:
        self.feature_cols = get_feature_cols(reference_df, as_string=True)
        self.reference_df = reference_df

    @abc.abstractmethod
    def compute_statistic(
        self, label: str, batch: str, group_df: pl.DataFrame, ref_df: pl.DataFrame
    ) -> dict[str, Any]:
        """
        Compute per-feature statistics for a single (batch, label) group.

        Parameters
        ----------
        label : str
            The variant label for this group.
        batch : str
            The batch identifier for this group.
        group_df : pl.DataFrame
            The subset of the input DataFrame corresponding to this
            (batch, label) group.
        ref_df : pl.DataFrame
            The subset of the reference DataFrame corresponding to this batch.

        Returns
        -------
        dict[str, Any]
            A dict with at minimum ``_meta_label`` and ``_meta_batch`` keys,
            plus one entry per feature column containing the computed statistic.
        """
        raise NotImplementedError

    def aggregate(
        self,
        agg_df: pl.DataFrame,
        n_jobs: int = -1,
        backend: str = "threading",
        verbose: int = 0,
    ) -> pl.DataFrame:
        """
        Compute per-(batch, label) statistics for all features using joblib.

        Filters out wildtype rows, groups by ``_meta_label`` and
        ``_meta_batch``, and dispatches one :meth:`compute_statistic` call
        per group in parallel.

        Parameters
        ----------
        agg_df : pl.DataFrame
            Input DataFrame containing ``_meta_batch``, ``_meta_label``, and
            feature columns.
        n_jobs : int, optional
            Number of parallel jobs passed to joblib.
        backend : str, optional
            Joblib backend.
        verbose : int, optional
            Joblib verbosity level. Defaults to ``0``.

        Returns
        -------
        pl.DataFrame
            A DataFrame with one row per (batch, label) group and one column
            per feature containing the computed statistic.
        """
        groups = agg_df.filter(pl.col("_meta_label") != "WT").group_by(
            "_meta_label", "_meta_batch"
        )
        tasks = [((label, batch), group_df) for (label, batch), group_df in groups]

        dicts = joblib.Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
            joblib.delayed(self.compute_statistic)(
                label=label,
                batch=batch,
                group_df=group_df,
                ref_df=self.reference_df.filter(pl.col("_meta_batch") == batch),
            )
            for (label, batch), group_df in tasks
        )

        return pl.DataFrame(dicts)


class NativeAggregator(BaseAggregator):
    """
    Base class for aggregators implemented as native Polars group-by expressions.

    Subclasses override :meth:`polars_exprs` to return a list of Polars
    expressions that are passed directly to ``group_by().agg()``.
    """

    @abc.abstractmethod
    def polars_exprs(self, feature_cols: list[str]) -> list[pl.Expr]:
        """
        Return Polars aggregation expressions for the given feature columns.

        Parameters
        ----------
        feature_cols : list[str]
            Names of the feature columns to aggregate.

        Returns
        -------
        list[pl.Expr]
            Polars expressions to pass to ``group_by().agg()``.
        """
        raise NotImplementedError

    def aggregate(self, agg_df: pl.DataFrame) -> pl.DataFrame:
        feature_cols = get_feature_cols(agg_df, as_string=True)
        return (
            agg_df.filter(pl.col("_meta_label") != "WT")
            .group_by("_meta_label", "_meta_batch")
            .agg(self.polars_exprs(feature_cols))
        )


class MeanAggregator(NativeAggregator):
    """Computes per-group mean for each feature column."""

    def polars_exprs(self, feature_cols: list[str]) -> list[pl.Expr]:
        return [pl.col(f).mean().alias(f"{f}_mean") for f in feature_cols]


class MedianAggregator(NativeAggregator):
    """Computes per-group median for each feature column."""

    def polars_exprs(self, feature_cols: list[str]) -> list[pl.Expr]:
        return [pl.col(f).median().alias(f"{f}_median") for f in feature_cols]


class MADAggregator(NativeAggregator):
    """Computes per-group median absolute deviation (MAD) for each feature column."""

    def polars_exprs(self, feature_cols: list[str]) -> list[pl.Expr]:
        return [
            (pl.col(f) - pl.col(f).median()).abs().median().alias(f"{f}_MAD")
            for f in feature_cols
        ]


class StdAggregator(NativeAggregator):
    """Computes per-group standard deviation for each feature column."""

    def polars_exprs(self, feature_cols: list[str]) -> list[pl.Expr]:
        return [pl.col(f).std().alias(f"{f}_std") for f in feature_cols]


class EMDAggregator(ReferenceBaseAggregator):
    """
    Computes per-group 1D Wasserstein distances (Earth Mover's Distance)
    against a reference distribution for each feature column.
    """

    def compute_statistic(
        self, label: str, batch: str, group_df: pl.DataFrame, ref_df: pl.DataFrame
    ) -> dict[str, Any]:
        row: dict[str, Any] = {"_meta_label": label, "_meta_batch": batch}
        for feat in self.feature_cols:
            variant = group_df.get_column(feat).to_numpy()
            reference = ref_df.get_column(feat).to_numpy()
            row[f"{feat}_EMD"] = scipy.stats.wasserstein_distance(variant, reference)
        return row


class KSAggregator(ReferenceBaseAggregator):
    """
    Computes per-group two-sample Kolmogorov-Smirnov statistics against
    a reference distribution for each feature column.
    """

    def compute_statistic(
        self, label: str, batch: str, group_df: pl.DataFrame, ref_df: pl.DataFrame
    ) -> dict[str, Any]:
        row: dict[str, Any] = {"_meta_label": label, "_meta_batch": batch}
        for feat in self.feature_cols:
            variant = group_df.get_column(feat).to_numpy()
            reference = ref_df.get_column(feat).to_numpy()
            row[f"{feat}_KS"] = scipy.stats.ks_2samp(variant, reference).statistic
        return row


class QQCorrelationAggregator(ReferenceBaseAggregator):
    """
    Computes per-group Q-Q correlation against a reference distribution for
    each feature column.

    Parameters
    ----------
    reference_df : pl.DataFrame
        Reference DataFrame (typically control rows).
    n_quantiles : int, optional
        Number of quantile points to evaluate. Defaults to ``100``.
    """

    def __init__(self, reference_df: pl.DataFrame, n_quantiles: int = 100) -> None:
        super().__init__(reference_df)
        self.quantile_points = np.linspace(0, 1, n_quantiles)

    def compute_statistic(
        self, label: str, batch: str, group_df: pl.DataFrame, ref_df: pl.DataFrame
    ) -> dict[str, Any]:
        row: dict[str, Any] = {"_meta_label": label, "_meta_batch": batch}
        for feat in self.feature_cols:
            variant = group_df.get_column(feat).to_numpy()
            reference = ref_df.get_column(feat).to_numpy()
            variant_quantiles = np.quantile(variant, self.quantile_points)
            reference_quantiles = np.quantile(reference, self.quantile_points)
            row[f"{feat}_QQ"] = scipy.stats.pearsonr(
                variant_quantiles, reference_quantiles
            ).statistic
        return row


class AUROCAggregator(ReferenceBaseAggregator):
    """
    Computes per-group AUROC against a reference distribution for each
    feature column.

    Variant samples are labelled ``1`` and reference samples are labelled
    ``0``. ``0.5`` indicates identical distributions and ``1.0`` indicates
    perfect separability.
    """

    def compute_statistic(
        self, label: str, batch: str, group_df: pl.DataFrame, ref_df: pl.DataFrame
    ) -> dict[str, Any]:
        row: dict[str, Any] = {"_meta_label": label, "_meta_batch": batch}
        for feat in self.feature_cols:
            variant = group_df.get_column(feat).to_numpy()
            reference = ref_df.get_column(feat).to_numpy()
            values = np.concatenate([reference, variant])
            labels = np.concatenate(
                [
                    np.zeros(len(reference)),
                    np.ones(len(variant)),
                ]
            )

            auroc = sklearn.metrics.roc_auc_score(labels, values)
            if auroc < 0.5:
                auroc = 1 - auroc

            row[f"{feat}_AUROC"] = auroc

        return row


def variant_classification(v: str) -> str:
    """Thanks Sriram"""
    if "fs" in v:
        classification = "Frameshift"
    elif v[-1] == "-":
        vs = v.split("|")
        ncodons_aff = len(vs)
        if ncodons_aff > 2:
            classification = "Other"
        else:
            if ncodons_aff == 1:
                classification = "3nt Deletion"
            elif ncodons_aff == 2:
                if int(vs[0][1:-1]) == (int(vs[1][1:-1]) - 1):
                    classification = "3nt Deletion"
                else:
                    classification = "Other"
            else:
                classification = "Other"
    elif ("X" in v) | ("*" in v):
        classification = "Nonsense"
    elif "WT" in v:
        classification = "WT"
    else:
        regex_match = re.match(r"([A-Z])(\d+)([A-Z])", v)
        if regex_match is None:
            classification = "Other"
        elif regex_match.group(1) == regex_match.group(3):
            classification = "Synonymous"
        else:
            classification = "Single Missense"

    return classification


def aggregate(
    norm_df: pl.DataFrame | PathLike,
    normalize_emds: bool = True,
    norm_only_to_synonymous: bool = False,
) -> tuple[pl.DataFrame, Optional[Normalizer]]:
    """
    Aggregate a normalized feature dataframe by (batch, label), computing both
    per-feature medians and per-feature distances-to-control (EMD/Wasserstein).

    This function performs two complementary per-group summaries over groups
    defined by (`_meta_batch`, `_meta_label`):

      1) **Location summary (median):**
         Computes `median(feature)` for each feature column within the group.

      2) **Distribution-shift summary (EMD/Wasserstein):**
         For each feature, computes the 1D Wasserstein distance between the
         distribution of values in the group and the cached reference
         distribution for the same batch. The reference distributions are
         built from rows where `_meta_is_control == True`.

      3) **Optional EMD normalization:**
         If `normalize_emds=True`, fits a batch-wise z-score normalizer on the
         aggregated EMD columns and returns normalized EMDs.

         If `norm_only_to_synonymous=True`, the EMD normalizer is fit **only**
         on groups whose `_meta_label` classifies as "Synonymous" via
         `variant_classification(...)`. In this mode, a synthetic
         `_meta_is_control` column is created on the aggregated EMD dataframe
         to mark "Synonymous" groups as controls for fitting.

    Parameters
    ----------
    norm_df : pl.DataFrame | PathLike
        Either an in-memory Polars DataFrame or a path to a parquet file.

        The dataframe must include:
          - `_meta_batch` (batch identifier)
          - `_meta_label` (group label identifier)
          - `_meta_is_control` (boolean control indicator used to build the
            per-batch reference distributions)
          - feature columns returned by `get_feature_cols(...)`

    normalize_emds : bool, default=True
        If True, fit a batch-wise normalizer on the aggregated EMD columns and
        return normalized EMDs along with the fitted normalizer. If False,
        EMD columns are returned unnormalized and the returned normalizer is
        `None`.

    norm_only_to_synonymous : bool, default=False
        If True, fit the EMD normalizer only on aggregated groups classified as
        "Synonymous" by `variant_classification(_meta_label)`. This is useful
        when you want the EMD z-scoring baseline to be defined by synonymous
        variants rather than all groups.

        Notes:
        - This option has an effect only when `normalize_emds=True`.
        - It does *not* change which rows are used to build the EMD reference
          distributions; those always come from the input rows where
          `_meta_is_control == True`.

    Returns
    -------
    tuple[pl.DataFrame, Optional[Normalizer]]
        A tuple `(agg_df, normalizer)`.

        - `agg_df` contains one row per (`_meta_batch`, `_meta_label`) group and
            - per-feature medians (native Polars group-by median)
            - per-feature EMD columns named `{feature}_EMD` (raw or normalized)

        - `normalizer` is:
            - `None` if `normalize_emds=False`
            - a fitted `Normalizer` if `normalize_emds=True`
              (possibly fit only on synonymous groups if
               `norm_only_to_synonymous=True`)
    """
    if not isinstance(norm_df, pl.DataFrame):
        logging.info(
            "Loading normalized feature dataframe from parquet: %s", str(norm_df)
        )
        norm_df = pl.read_parquet(norm_df)
    else:
        logging.info(
            "Using in-memory normalized feature dataframe: %d rows × %d cols",
            norm_df.height,
            norm_df.width,
        )

    required_meta = {"_meta_batch", "_meta_label", "_meta_is_control"}
    missing = required_meta - set(norm_df.columns)
    if missing:
        raise ValueError(
            f"norm_df missing required metadata columns: {sorted(missing)}"
        )

    logging.info("Selecting control rows for EMD reference cache")
    control_df = norm_df.filter(pl.col("_meta_is_control"))
    logging.info("Control df: %d rows", control_df.height)

    logging.info("Initializing EMD aggregator (reference cache)")
    aggregator = EMDAggregator(control_df)

    feature_cols = get_feature_cols(norm_df, as_string=True)
    meta_cols = ["_meta_label", "_meta_batch"]
    logging.info(
        "Aggregating medians for %d features over %d meta columns",
        len(feature_cols),
        len(meta_cols),
    )

    median_df = norm_df.select(meta_cols + feature_cols).group_by(meta_cols).median()
    logging.info(
        "Median dataframe shape: %d rows × %d cols", median_df.height, median_df.width
    )

    logging.info("Computing EMD dataframe")
    emd_df = aggregator.aggregate(norm_df)
    logging.info("EMD dataframe shape: %d rows × %d cols", emd_df.height, emd_df.width)

    if not normalize_emds:
        logging.info("Joining median + raw EMD dataframes")
        agg_df = median_df.join(emd_df, on=meta_cols)
        logging.info(
            "Aggregated dataframe shape: %d rows × %d cols", agg_df.height, agg_df.width
        )
        return agg_df, None

    if norm_only_to_synonymous:
        emd_df = emd_df.with_columns(
            (
                pl.col("_meta_label").map_elements(
                    variant_classification, return_dtype=pl.Utf8
                )
                == "Synonymous"
            ).alias("_meta_is_control")
        )

    logging.info("Fitting EMD normalizer (batch-wise)")
    emd_lf = emd_df.lazy()
    normalizer = fit_normalizer(
        emd_lf, fit_batch_wise=True, fit_only_on_control=norm_only_to_synonymous
    )

    logging.info("Normalizing EMD dataframe")
    emd_df = normalize(emd_lf, normalizer).collect()

    logging.info("Joining median + normalized EMD dataframes")
    agg_df = median_df.join(emd_df, on=meta_cols)
    logging.info(
        "Aggregated dataframe shape: %d rows × %d cols", agg_df.height, agg_df.width
    )

    logging.info("Aggregation complete")
    return agg_df, normalizer


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


def compute_cli(
    norm_df: PathLike,
    out_dir: PathLike,
    aggregator: str,
) -> None:
    """
    Compute per-(batch, label) aggregate statistics and write the result.

    Reads a normalized feature parquet, runs the specified aggregator over all
    (batch, label) groups (excluding WT), and writes the result — containing
    only the aggregate feature columns — to ``<out_dir>/aggregated.parquet``.

    For reference-based aggregators (EMD, KS, QQ, AUROC) the per-batch
    reference distributions are built from rows where
    ``_meta_is_control == True``.

    Parameters
    ----------
    norm_df : PathLike
        Path to the input normalized feature parquet. Must contain
        ``_meta_batch``, ``_meta_label``, and feature columns.  Reference-based
        aggregators also require ``_meta_is_control``.
    out_dir : PathLike
        Output directory. Created if it does not exist.
    aggregator : str
        Aggregation method.  One of: ``mean``, ``median``, ``MAD``, ``std``,
        ``EMD``, ``KS``, ``QQ``, ``AUROC``.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    if aggregator not in _AGGREGATORS:
        raise ValueError(
            f"Unknown aggregator {aggregator!r}. "
            f"Choose from: {sorted(_AGGREGATORS)}"
        )

    logging.info("Loading normalized dataframe from %s", str(norm_df))
    df = pl.read_parquet(norm_df)
    logging.info("Loaded: %d rows × %d cols", df.height, df.width)

    agg_cls = _AGGREGATORS[aggregator]
    if issubclass(agg_cls, ReferenceBaseAggregator):
        if "_meta_is_control" not in df.columns:
            raise ValueError(
                f"Aggregator {aggregator!r} requires '_meta_is_control' column."
            )
        control_df = df.filter(pl.col("_meta_is_control"))
        logging.info("Building reference from %d control rows", control_df.height)
        agg = agg_cls(control_df)
    else:
        agg = agg_cls()

    logging.info("Running %s aggregator", aggregator)
    result = agg.aggregate(df)
    logging.info("Result: %d rows × %d cols", result.height, result.width)

    out_path = out_dir / "aggregated.parquet"
    logging.info("Writing aggregated dataframe to %s", str(out_path))
    result.write_parquet(out_path)
    logging.info("Done")


def normalize_cli(
    agg_df: PathLike,
    out_dir: PathLike,
) -> None:
    """
    Normalize an aggregate dataframe to the synonymous-variant baseline.

    Reads an aggregate parquet (produced by ``compute``), classifies each row's
    ``_meta_label`` with :func:`variant_classification`, marks "Synonymous"
    rows as the reference population, fits a batch-wise z-score normalizer on
    those rows, and writes the normalized result.

    Parameters
    ----------
    agg_df : PathLike
        Path to the aggregate feature parquet. Must contain ``_meta_batch``,
        ``_meta_label``, and aggregate feature columns.
    out_dir : PathLike
        Output directory. Created if it does not exist.  Writes:

        - ``normalized.parquet`` — the normalized aggregate dataframe
        - ``normalizer.pkl`` — the fitted :class:`~.normalize.Normalizer`
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    logging.info("Loading aggregate dataframe from %s", str(agg_df))
    df = pl.read_parquet(agg_df)
    logging.info("Loaded: %d rows × %d cols", df.height, df.width)

    logging.info("Marking synonymous rows as normalization reference")
    df = df.with_columns(
        (
            pl.col("_meta_label").map_elements(
                variant_classification, return_dtype=pl.Utf8
            )
            == "Synonymous"
        ).alias("_meta_is_control")
    )

    lf = df.lazy()
    logging.info("Fitting batch-wise normalizer on synonymous rows")
    normalizer = fit_normalizer(lf, fit_batch_wise=True, fit_only_on_control=True)

    logging.info("Normalizing")
    normalized_df = normalize(lf, normalizer).collect()
    normalized_df = normalized_df.drop("_meta_is_control")

    out_path = out_dir / "normalized.parquet"
    logging.info("Writing normalized dataframe to %s", str(out_path))
    normalized_df.write_parquet(out_path)

    norm_path = out_dir / "normalizer.pkl"
    logging.info("Writing normalizer to %s", str(norm_path))
    normalizer.save(norm_path)
    logging.info("Done")


if __name__ == "__main__":
    fire.Fire({"compute": compute_cli, "normalize": normalize_cli})
