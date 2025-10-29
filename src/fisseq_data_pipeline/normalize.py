import dataclasses
import logging
import pickle
from os import PathLike
from typing import Optional

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


def _add_batch_col(
    feature_df: pl.LazyFrame, meta_data_df: Optional[pl.LazyFrame], is_batch_wise: bool
) -> pl.LazyFrame:
    """
    Attach a `_batch` column to the feature LazyFrame.

    Parameters
    ----------
    feature_df : pl.LazyFrame
        LazyFrame containing per-sample features to which the `_batch`
        column should be added.
    meta_data_df : pl.LazyFrame or None
        Row-aligned metadata LazyFrame containing the `_batch` column.
        Required if `is_batch_wise` is True.
    is_batch_wise : bool
        If True, add the actual `_batch` column from the metadata.
        If False, assign a dummy `_batch = 0` to all rows.

    Returns
    -------
    pl.LazyFrame
        The input feature LazyFrame augmented with a `_batch` column,
        suitable for downstream grouping or normalization.
    """
    if is_batch_wise:
        feature_df = (
            feature_df.with_row_index(name="_idx")
            .join(
                meta_data_df.with_row_index(name="_idx").select(
                    pl.col("_idx"), pl.col("_batch")
                ),
                on="_idx",
            )
            .select(pl.exclude("_idx"))
        )
    else:
        feature_df = feature_df.with_columns(pl.lit(0).cast(pl.Int8).alias("_batch"))

    return feature_df


def fit_normalizer(
    feature_df: pl.LazyFrame,
    meta_data_df: Optional[pl.LazyFrame] = None,
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
        logging.info("Adding query to filter for control samples")
        feature_df = (
            feature_df.with_row_index(name="_idx")
            .join(
                meta_data_df.with_row_index(name="_idx").select(
                    pl.col("_idx"), pl.col("_is_control")
                ),
                on="_idx",
            )
            .filter(pl.col("_is_control"))
            .select(pl.exclude(["_idx", "_is_control"]))
        )
        meta_data_df = meta_data_df.filter(pl.col("_is_control"))

    # Handle batch column (batch-wise vs global)
    feature_df = _add_batch_col(feature_df, meta_data_df, fit_batch_wise)
    agg_exprs = []
    for c in feature_df.columns:
        if c == "_batch":
            continue
        agg_exprs.append(pl.col(c).mean().alias(f"{c}_mean"))
        agg_exprs.append(pl.col(c).std().alias(f"{c}_std"))

    agg_df = feature_df.group_by("_batch").agg(agg_exprs)
    mean_cols = [c for c in agg_df.columns if c.endswith("_mean")]
    std_cols = [c for c in agg_df.columns if c.endswith("_std")]

    means, stds = (
        agg_df.select(
            [pl.col(c).alias(c.removesuffix(sfx)) for c in cols] + [pl.col("_batch")]
        ).collect()
        for cols, sfx in [(mean_cols, "_mean"), (std_cols, "_std")]
    )

    zero_var_cols = [
        c
        for c in stds.columns
        if c != "_batch" and (stds[c] <= np.finfo(np.float32).eps).any()
    ]
    if zero_var_cols:
        logging.warning("Dropping %d zero-variance columns", len(zero_var_cols))
        means = means.select(pl.exclude(zero_var_cols))
        stds = stds.select(pl.exclude(zero_var_cols))

    logging.info("Normalization statistics computed successfully")
    return Normalizer(means=means, stds=stds, is_batch_wise=fit_batch_wise)


def normalize(
    feature_df: pl.LazyFrame,
    normalizer: Normalizer,
    meta_data_df: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
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
    if normalizer.is_batch_wise and meta_data_df is None:
        raise ValueError("Meta data required to use batch-wise normalizer")

    logging.info("Creating normalization query")
    feature_df = _add_batch_col(feature_df, meta_data_df, normalizer.is_batch_wise)
    feature_cols = feature_df.columns
    norm_cols = normalizer.stds.columns

    if set(norm_cols) != set(feature_cols):
        feature_df = feature_df.select(norm_cols)
        logging.warning(
            "Dropped %d columns from feature_df not present in normalizer",
            len(feature_cols) - len(norm_cols),
        )

    for suffix, df in [("_mean", normalizer.means), ("_std", normalizer.stds)]:
        feature_df = feature_df.join(df.lazy(), on="_batch", how="left", suffix=suffix)

    feature_df = feature_df.select(
        [
            ((pl.col(c) - pl.col(f"{c}_mean")) / pl.col(f"{c}_std")).alias(c)
            for c in norm_cols
            if c != "_batch"
        ]
    )

    return feature_df
