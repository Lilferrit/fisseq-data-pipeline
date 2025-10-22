import dataclasses
import logging
from typing import Dict, Optional

import numpy as np
import polars as pl


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
    mapping : dict[str, int] or None
        Optional mapping from batch label strings (from the `_batch` column of
        the metadata) to the corresponding integer batch indices used in
        `means` and `stds`. If `None`, all samples are assumed to belong to
        a single batch.
    """

    means: pl.DataFrame
    stds: pl.DataFrame
    mapping: Optional[Dict[str, int]]


def fit_normalizer(
    feature_df: pl.DataFrame,
    meta_data_df: Optional[pl.DataFrame] = None,
    fit_only_on_control: bool = False,
    fit_batch_wise: bool = True,
) -> Normalizer:
    """
    Compute per-feature means and standard deviations for z-score normalization.

    The function can operate in two modes:
      * **Global normalization** - if `fit_batch_wise=False`, compute one mean
        and standard deviation per feature across all samples.
      * **Batch-wise normalization** - if `fit_batch_wise=True`,
        compute separate statistics for each batch defined by the `_batch`
        column of `meta_data_df`.

    Optionally, the statistics may be estimated only from control samples
    indicated by a boolean `_is_control` column in the metadata.

    Columns with (near) zero variance are automatically excluded from the
    resulting normalizer to avoid divide-by-zero errors during scaling.

    Parameters
    ----------
    feature_df : pl.DataFrame
        Feature matrix of shape (n_samples, n_features). Each column represents
        a quantitative feature to be normalized.
    meta_data_df : pl.DataFrame, optional
        Metadata DataFrame aligned row-wise with `feature_df`. Must contain a
        `_batch` column if `fit_batch_wise=True`, and an `_is_control` column
        if `fit_only_on_control=True`.
    fit_only_on_control : bool, default=False
        If True, compute normalization statistics using only rows where
        `meta_data_df["_is_control"]` is True.
    fit_batch_wise : bool, default=True
        If True, compute means and standard deviations separately for each
        batch. Requires that `meta_data_df` include a `_batch` column.

    Returns
    -------
    Normalizer
        A dataclass containing the per-feature means, standard deviations,
        and optional batch mapping for use in downstream normalization.
    """
    if (fit_only_on_control or fit_batch_wise) and meta_data_df is None:
        raise ValueError("Meta data required to fit to control samples or by batch")

    if fit_only_on_control:
        logging.info(
            "Filtering control samples, number of samples before filtering=%d",
            len(feature_df),
        )
        feature_df = feature_df.filter(meta_data_df.get_column("_is_control"))
        meta_data_df = meta_data_df.filter(meta_data_df.get_column("_is_control"))
        logging.info(
            "Filtering complete, remaining train set samples shape=%s",
            len(feature_df.shape),
        )

    logging.info("Fitting Normalizer")
    if fit_batch_wise:
        batch_col = meta_data_df.get_column("_batch").cast(pl.Categorical)
        mapping = {cat: i for i, cat in enumerate(batch_col.cat.get_categories())}
        batch_col = batch_col.to_physical().alias("_batch_idx")
    else:
        mapping = None
        batch_col = pl.repeat(0, n=len(feature_df), dtype=pl.Int8)

    feature_df = feature_df.with_columns(batch_col.alias("_batch_idx"))
    batch_group_by = feature_df.group_by(by="_batch_idx")

    logging.info("Calculating feature means")
    means = (
        batch_group_by.mean()
        .sort(by="_batch_idx")
        .select(pl.exclude(["_batch_idx", "by"]))
    )

    logging.info("Calculating feature deviations")
    stds = (
        batch_group_by.agg([pl.col(col).std() for col in means.columns])
        .sort(by="by")
        .select(pl.exclude("by"))
    )

    logging.info("Checking for zero variance columns")
    is_zero_var = stds.select(
        [pl.col(col) <= np.finfo(np.float32).eps for col in stds.columns]
    )
    zero_var_cols = [col for col in is_zero_var.columns if is_zero_var[col].any()]

    if len(zero_var_cols) > 0:
        logging.warning("Dropping %d zero variance columns", len(zero_var_cols))
        means = means.select(pl.exclude(zero_var_cols))
        stds = stds.select(pl.exclude(zero_var_cols))

    logging.info("Done")
    normalizer = Normalizer(means=means, stds=stds, mapping=mapping)
    return normalizer


def normalize(
    feature_df: pl.DataFrame,
    normalizer: Normalizer,
    meta_data_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Apply z-score normalization to a feature matrix.

    Each feature value is standardized as:
        z = (x - mean) / std
    where `mean` and `std` are drawn from the appropriate batch in
    `normalizer.means` and `normalizer.stds`.

    When the normalizer was fitted batch-wise, the `_batch` column in
    `meta_data_df` determines which row of the stored statistics applies
    to each sample.

    Columns absent from the normalizer (e.g., zero-variance columns that were
    dropped during fitting) are automatically removed prior to scaling.

    Parameters
    ----------
    feature_df : pl.DataFrame
        Feature matrix of shape (n_samples, n_features) to be normalized.
    normalizer : Normalizer
        Object containing the means, standard deviations, and batch mapping
        produced by :func:`fit_normalizer`.
    meta_data_df : pl.DataFrame, optional
        Metadata aligned row-wise with `feature_df`. Required when the
        normalizer was fitted batch-wise (i.e., `normalizer.mapping` is not
        None), and must include a `_batch` column.

    Returns
    -------
    pl.DataFrame
        Normalized feature matrix of shape (n_samples, n_retained_features),
        where columns with zero variance during fitting are omitted.
    """
    if normalizer.mapping is not None and meta_data_df is None:
        raise ValueError("Meta data required to use batch-wise normalizer")

    logging.info("Setting up normalization, data shape=%s", feature_df.shape)
    if normalizer.mapping is None:
        batch_idx = pl.Series("batch_idx", [0] * len(feature_df), dtype=pl.UInt8)
    else:
        try:
            batch_idx = meta_data_df.get_column("_batch").map_elements(
                lambda x: normalizer.mapping[x], return_dtype=pl.UInt32
            )
        except KeyError as e:
            new_exc = KeyError(
                "Batch row mapping file - this is likely caused by a batch that is in"
                " meta_data_df but was not in the meta data used to fit the normalizer."
            )
            raise new_exc from e

    if set(normalizer.stds.columns) != set(feature_df.columns):
        n_cols = feature_df.width
        feature_df = feature_df.select(normalizer.stds.columns)
        logging.warning(
            "Dropped %d columns from feature_df that were not included in the"
            " normalizer",
            n_cols - feature_df.width,
        )

    logging.info("Subtracting means")
    op_df = normalizer.means[batch_idx, :]
    feature_df -= op_df

    logging.info("Standardizing")
    op_df = normalizer.stds[batch_idx, :]
    feature_df /= op_df

    logging.info("Finished normalization")
    return feature_df
