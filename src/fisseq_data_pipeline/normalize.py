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

    # Convert both to lazy frames immediately
    lf_features = feature_df.lazy()

    # Filter to control samples (if needed)
    if fit_only_on_control:
        logging.info("Filtering control samples")
        mask = meta_data_df.get_column("_is_control")
        lf_features = lf_features.filter(mask)
        meta_data_df = meta_data_df.filter(mask)

    # Handle batch column (batch-wise vs global)
    if fit_batch_wise:
        batch_col = meta_data_df["_batch"].cast(pl.Categorical)
        mapping = {cat: i for i, cat in enumerate(batch_col.cat.get_categories())}
        lf_features = lf_features.with_columns(
            pl.lit(batch_col.to_physical()).alias("_batch_idx")
        )
    else:
        mapping = None
        lf_features = lf_features.with_columns(
            pl.lit(0).cast(pl.Int8).alias("_batch_idx")
        )

    agg_exprs = []
    for c in feature_df.columns:
        if c in ("_batch_idx", "_is_control"):
            continue

        agg_exprs.append(pl.col(c).mean().alias(f"{c}_mean"))
        agg_exprs.append(pl.col(c).std().alias(f"{c}_std"))

    logging.info("Computing per-batch means and stds lazily")
    agg_df = (
        lf_features.group_by("_batch_idx").agg(agg_exprs).sort("_batch_idx").collect()
    )

    mean_cols = [c for c in agg_df.columns if c.endswith("_mean")]
    std_cols = [c for c in agg_df.columns if c.endswith("_std")]
    means = agg_df.select([pl.col(c).alias(c.removesuffix("_mean")) for c in mean_cols])
    stds = agg_df.select([pl.col(c).alias(c.removesuffix("_std")) for c in std_cols])

    zero_var_cols = [
        c for c in stds.columns if (stds[c] <= np.finfo(np.float32).eps).any()
    ]
    if zero_var_cols:
        logging.warning("Dropping %d zero-variance columns", len(zero_var_cols))
        means = means.select(pl.exclude(zero_var_cols))
        stds = stds.select(pl.exclude(zero_var_cols))

    logging.info("Normalization statistics computed successfully")
    return Normalizer(means=means, stds=stds, mapping=mapping)


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
        batch_idx = pl.Series("_batch_idx", [0] * len(feature_df), dtype=pl.UInt8)
    else:
        try:
            batch_idx = (
                meta_data_df.get_column("_batch")
                .map_elements(lambda x: normalizer.mapping[x], return_dtype=pl.UInt32)
                .alias("_batch_idx")
            )
        except KeyError as e:
            new_exc = KeyError(
                "Batch row mapping failed - this is likely caused by a batch that is in"
                " meta_data_df but was not in the meta data used to fit the normalizer."
            )
            raise new_exc from e

    logging.info("Creating normalization query")
    lf = feature_df.lazy()
    feature_cols = feature_df.columns
    norm_cols = normalizer.stds.columns

    if set(norm_cols) != set(feature_cols):
        lf = lf.select(norm_cols)
        logging.warning(
            "Dropped %d columns from feature_df not present in normalizer",
            len(feature_cols) - len(norm_cols),
        )

    lf = lf.with_columns(batch_idx)
    lf_means = normalizer.means.lazy().with_columns(pl.row_index("_batch_idx"))
    lf_stds = normalizer.stds.lazy().with_columns(pl.row_index("_batch_idx"))
    lf = lf.join(lf_means, on="_batch_idx", how="left", suffix="_mean")
    lf = lf.join(lf_stds, on="_batch_idx", how="left", suffix="_std")

    exprs = [
        ((pl.col(c) - pl.col(f"{c}_mean")) / pl.col(f"{c}_std")).alias(c)
        for c in norm_cols
    ]
    lf = lf.select(exprs)

    logging.info("Resolving normalization query")
    return lf.collect()
