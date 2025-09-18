import dataclasses
from typing import Optional

import polars as pl


@dataclasses.dataclass
class Normalizer:
    """
    Container for normalization statistics.

    Attributes
    ----------
    means : pl.DataFrame
        A 1×n DataFrame containing the mean value of each column.
    stds : pl.DataFrame
        A 1×n DataFrame containing the standard deviation of each column.
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
        feature_df = feature_df.filter(meta_data_df.get_column("_is_control"))

    return Normalizer(means=feature_df.mean(), stds=feature_df.std())


def normalize(feature_df: pl.DataFrame, normalizer: Normalizer) -> pl.DataFrame:
    """
    Apply z-score normalization to features using precomputed statistics.

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
    means = normalizer.means.row(0, named=True)
    stds = normalizer.stds.row(0, named=True)
    feature_df = feature_df.with_columns(
        ((pl.col(c) - pl.lit(means[c])) / pl.lit(stds[c])).alias(c)
        for c in feature_df.columns
    )

    return feature_df
