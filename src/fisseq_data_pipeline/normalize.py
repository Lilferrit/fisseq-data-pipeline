from os import PathLike
from typing import Optional

import polars as pl
import sklearn.preprocessing

from .utils import Config, get_control_samples, get_feature_matrix, set_feature_matrix

Normalizer = sklearn.preprocessing.StandardScaler


def fit_normalizer(
    train_data_df: pl.LazyFrame,
    config: Optional[PathLike | Config],
) -> Normalizer:
    """
    Fit a normalization model (scikit-learn StandardScaler) using control
    samples from the training data.

    Parameters
    ----------
    train_data_df : pl.LazyFrame
        The training data as a Polars LazyFrame.
    config : PathLike or Config, optional
        Configuration object or path. Must provide the attribute
        ``control_sample_query`` to select control samples.

    Returns
    -------
    Normalizer : sklearn.preprocessing.StandardScaler
        A fitted StandardScaler instance that stores the mean and variance
        of the control feature matrix.
    """
    config = Config(config)
    normalizer = sklearn.preprocessing.StandardScaler()
    control_df = get_control_samples(train_data_df, config)
    _, control_feature_matrix = get_feature_matrix(control_df, config)
    normalizer.fit(control_feature_matrix)

    return normalizer


def normalize(
    data_df: pl.LazyFrame,
    config: Optional[PathLike | Config],
    normalizer: Normalizer,
) -> pl.LazyFrame:
    """
    Apply a fitted normalization model (StandardScaler) to feature columns
    of a LazyFrame.

    Parameters
    ----------
    data_df : pl.LazyFrame
        The input data as a Polars LazyFrame.
    config : PathLike or Config, optional
        Configuration object or path. Must provide the attribute
        ``feature_cols`` specifying the columns to normalize.
    normalizer : sklearn.preprocessing.StandardScaler
        A fitted StandardScaler returned by ``fit_normalizer``.

    Returns
    -------
    pl.LazyFrame
        A LazyFrame with normalized feature columns, where each feature
        has been centered and scaled according to the fitted normalizer.
    """
    config = Config(config)
    feature_cols, data_feature_matrix = get_feature_matrix(data_df, config)
    data_feature_matrix = normalizer.transform(data_feature_matrix)
    return set_feature_matrix(data_df, feature_cols, data_feature_matrix)
