import dataclasses
import logging
import pickle
from os import PathLike

import numpy as np
import polars as pl

from .utils import get_feature_cols


@dataclasses.dataclass
class Normalizer:
    """
    Container object storing per-feature normalization statistics.

    Attributes
    ----------
    means : pl.DataFrame
        A DataFrame of shape (n_batches, n_features) containing the mean value
        of each feature for each batch. When batch-wise normalization is not
        used, this has shape (1, n_features).
    stds : pl.DataFrame
        A DataFrame of shape (n_batches, n_features) containing the standard
        deviation of each feature for each batch. When batch-wise normalization
        is not used, this has shape (1, n_features).
    is_batch_wise : dict[str, int] or None
        Whether statistics were computed batch-wise or globally
    """

    means: pl.DataFrame
    stds: pl.DataFrame
    is_batch_wise: bool

    def save(self, save_path: PathLike) -> None:
        """
        Serialize and save the Normalizer object to disk using pickle.

        This method stores the fitted normalization statistics — per-feature
        means, standard deviations, and batch-wise configuration flag — as a
        single binary file that can later be reloaded to reproduce the same
        normalization behavior.

        Parameters
        ----------
        save_path : PathLike
            Destination file path for the serialized Normalizer object. The file
            is written in binary format (typically named ``normalizer.pkl``).

        Notes
        -----
        - The file can be reloaded using ``pickle.load(open(path, "rb"))``.
        - Only the Normalizer object and its attributes are serialized; any
        external references (e.g., LazyFrames) are not included.
        - The resulting file is Python-version dependent and not guaranteed
        to be portable across major interpreter versions.

        Examples
        --------
        >>> normalizer = fit_normalizer(feature_df, meta_data_df)
        >>> normalizer.save("output/normalizer.pkl")

        To reload later:
        >>> with open("output/normalizer.pkl", "rb") as f:
        ...     normalizer = pickle.load(f)
        """
        with open(save_path, "wb") as f:
            pickle.dump(self, f)


def fit_normalizer(
    data_lf: pl.LazyFrame,
    fit_only_on_control: bool = False,
    fit_batch_wise: bool = True,
) -> Normalizer:
    """
    Compute per-feature mean and standard deviation statistics for
    z-score normalization.

    The normalizer can operate in two modes:

    **1. Global normalization (`fit_batch_wise=False`)**
       All samples are assigned to a single synthetic batch
       (`_meta_batch = 0`). A single mean and standard deviation
       are computed per feature.

    **2. Batch-wise normalization (`fit_batch_wise=True`)**
       Means and standard deviations are computed independently for
       each batch defined by the existing `_meta_batch` column.

    If `fit_only_on_control=True`, rows are filtered using the
    boolean `_meta_is_control` column before statistics are computed.

    Columns with zero or near-zero variance are identified and dropped
    from the returned statistics to avoid division-by-zero during
    normalization.

    Parameters
    ----------
    data_lf : pl.LazyFrame
        A LazyFrame containing:
          - numerical feature columns
          - a `_meta_batch` column (if `fit_batch_wise=True`)
          - optionally a `_meta_is_control` column
    fit_only_on_control : bool, default False
        Whether to compute statistics only from rows where
        `_meta_is_control == True`.
    fit_batch_wise : bool, default True
        Whether to compute statistics separately per batch. If False,
        all samples are assigned to a single batch.

    Returns
    -------
    Normalizer
        A dataclass holding:
          - `means` : per-batch feature means
          - `stds` : per-batch feature standard deviations
          - `is_batch_wise` : boolean flag matching input
    """
    if fit_only_on_control:
        logging.info("Adding query to filter for control samples")
        data_lf = data_lf.filter(pl.col("_meta_is_control"))

    if not fit_batch_wise:
        data_lf = data_lf.with_columns(pl.lit(0).alias("_meta_batch"))

    # Handle batch column (batch-wise vs global)
    logging.info("Adding normalization queries")
    agg_exprs = []
    for c in get_feature_cols(data_lf, as_string=True):
        agg_exprs.append(pl.col(c).mean().alias(f"{c}_mean"))
        agg_exprs.append(pl.col(c).std().alias(f"{c}_std"))

    agg_df = data_lf.group_by("_meta_batch").agg(agg_exprs)
    mean_cols = [c for c in agg_df.columns if c.endswith("_mean")]
    std_cols = [c for c in agg_df.columns if c.endswith("_std")]

    logging.info("Computing feature means")
    means = agg_df.select(
        pl.col("_meta_batch"),
        *[pl.col(c).alias(c.removesuffix("_mean")) for c in mean_cols],
    ).collect()

    logging.info("Computing feature standard deviations")
    stds = agg_df.select(
        pl.col("_meta_batch"),
        *[pl.col(c).alias(c.removesuffix("_std")) for c in std_cols],
    ).collect()

    logging.info("Scanning for zero variance columns")
    zero_var_cols = [
        c
        for c in get_feature_cols(data_lf, as_string=True)
        if (stds[c] <= np.finfo(np.float32).eps).any()
    ]

    if len(zero_var_cols) != 0:
        logging.warning("Dropping %d zero-variance columns", len(zero_var_cols))
        means = means.select(pl.exclude(zero_var_cols))
        stds = stds.select(pl.exclude(zero_var_cols))

    logging.info("Normalization statistics computed successfully")
    return Normalizer(means=means, stds=stds, is_batch_wise=fit_batch_wise)


def normalize(data_lf: pl.LazyFrame, normalizer: Normalizer) -> pl.LazyFrame:
    """
    Apply z-score normalization to a LazyFrame using precomputed statistics.

    Each feature column `f` is transformed into:

        z_f = (f - f_mean) / f_std

    where `f_mean` and `f_std` are taken from the corresponding row of
    `normalizer.means` and `normalizer.stds`.

    If the normalizer was fitted batch-wise, the `_meta_batch` column of
    `data_lf` determines which batch statistics to apply. If the normalizer
    was fitted globally (`is_batch_wise=False`), a dummy batch index 0 is
    applied to all rows.

    Columns that were removed during fitting (e.g., zero-variance features)
    are automatically dropped from `data_lf` prior to normalization.

    Parameters
    ----------
    data_lf : pl.LazyFrame
        A LazyFrame containing numerical feature columns and a `_meta_batch`
        column (or none if global normalization is desired).
    normalizer : Normalizer
        The object returned by :func:`fit_normalizer`, containing the
        per-feature mean and standard deviation tables.

    Returns
    -------
    pl.LazyFrame
        A LazyFrame where all feature columns have been z-score normalized.
        Columns not present in the normalizer (e.g., removed features) are
        excluded.
    """
    logging.info("Creating normalization query")
    if not normalizer.is_batch_wise:
        data_lf = data_lf.with_columns(pl.lit(0).alias("_meta_batch"))

    feature_cols = set(get_feature_cols(data_lf, as_string=True))
    norm_cols = set(get_feature_cols(normalizer.stds, as_string=True))
    bad_cols = feature_cols - norm_cols
    data_lf = data_lf.select(pl.exclude(bad_cols))

    if len(bad_cols) > 0:
        logging.warning(
            "Dropped %d columns from feature_df not present in normalizer",
            len(bad_cols),
        )

    feature_columns = feature_cols.intersection(norm_cols)
    for suffix, df in [("_mean", normalizer.means), ("_std", normalizer.stds)]:
        data_lf = data_lf.join(df.lazy(), on="_meta_batch", how="left", suffix=suffix)

    meta_columns = [c for c in data_lf.columns if c.startswith("_meta")]
    data_lf = data_lf.with_columns(
        [
            ((pl.col(c) - pl.col(f"{c}_mean")) / pl.col(f"{c}_std")).alias(c)
            for c in feature_columns
        ]
    ).select(list(feature_columns) + meta_columns)

    return data_lf
