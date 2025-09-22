import dataclasses
import logging
from typing import Optional

import numpy as np
import polars as pl


@dataclasses.dataclass
class Normalizer:
    """
    Container for normalization statistics.

    Attributes
    ----------
    means : pl.DataFrame
        A 1xn DataFrame containing the mean value of each column.
    stds : pl.DataFrame
        A 1xn DataFrame containing the standard deviation of each column.
    """

    means: pl.DataFrame
    stds: pl.DataFrame


def fit_normalizer(
    feature_df: pl.DataFrame,
    meta_data_df: Optional[pl.DataFrame] = None,
    fit_only_on_control: bool = False,
) -> Normalizer:
    """
    Compute column-wise means and standard deviations for feature normalization.

    Zero variance columns will be dropped from the normalizer to avoid a
    divide by zero error.

    Parameters
    ----------
    feature_df : pl.DataFrame
        Feature matrix with samples as rows and features as columns.
    meta_data_df : Optional[pl.DataFrame], default=None
        Metadata aligned row-wise with `feature_df`. Required if
        ``fit_only_on_control=True``.
    fit_only_on_control : bool, default=False
        If True, compute normalization statistics only from control samples
        indicated by the ``_is_control`` column in `meta_data_df`.

    Returns
    -------
    Normalizer
        Object containing per-column means and standard deviations.
    """
    if fit_only_on_control and meta_data_df is None:
        raise ValueError("Meta data required to fit to control samples")
    elif fit_only_on_control:
        logging.info(
            "Filtering control samples, number of samples before filtering=%d",
            len(feature_df),
        )
        feature_df = feature_df.filter(meta_data_df.get_column("_is_control"))
        logging.info(
            "Filtering complete, remaining train set samples shape=%s",
            len(feature_df.shape),
        )

    logging.info("Fitting Normalizer")
    means = feature_df.mean()
    stds = feature_df.std()
    zero_var_cols = [
        k for k, v in stds.row(0, named=True).items() if v < np.finfo(np.float32).eps
    ]

    if len(zero_var_cols) > 0:
        logging.warning("Dropping %d zero variance columns", len(zero_var_cols))

    means = means.select(pl.exclude(zero_var_cols))
    stds = stds.select(pl.exclude(zero_var_cols))

    normalizer = Normalizer(means=means, stds=stds)
    logging.info("Done")

    return normalizer


def normalize(feature_df: pl.DataFrame, normalizer: Normalizer) -> pl.DataFrame:
    """
    Apply z-score normalization to features using precomputed statistics.

    Only columns included in the normalizer will be present in the output

    Parameters
    ----------
    feature_df : pl.DataFrame
        Feature matrix to normalize; shape (n_samples, n_features).
    normalizer : Normalizer
        Precomputed means and standard deviations to use for scaling.

    Returns
    -------
    pl.DataFrame
        Normalized feature matrix with the same shape as `feature_df`.
    """
    logging.info("Setting up normalization, data shape=%s", feature_df.shape)
    means = normalizer.means.row(0, named=True)
    stds = normalizer.stds.row(0, named=True)
    n_cols = feature_df.width

    logging.info("Running normalization")
    feature_df = feature_df.select(
        [((pl.col(c) - means[c]) / stds[c]).alias(c) for c in normalizer.stds.columns]
    )

    if feature_df.width < n_cols:
        logging.warning(
            "Dropped %d columns from feature_df that were not included in the"
            " normalizer",
            n_cols - feature_df.width,
        )

    logging.info("Done")
    return feature_df
