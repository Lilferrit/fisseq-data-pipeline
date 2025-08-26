from typing import Optional
from os import PathLike

import polars as pl
import sklearn.preprocessing

from .utils import Config, get_control_samples, get_feature_matrix, set_feature_matrix

Normalizer = sklearn.preprocessing.StandardScaler


def fit_normalizer(
    train_data_df: pl.DataFrame,
    config: Optional[PathLike | Config],
) -> Normalizer:
    config = Config(config)
    normalizer = sklearn.preprocessing.StandardScaler()
    control_df = get_control_samples(train_data_df, config)
    _, control_feature_matrix = get_feature_matrix(control_df, config)
    normalizer.fit(control_feature_matrix)

    return normalizer


def normalize(
    data_df: pl.DataFrame,
    config: Optional[PathLike | Config],
    normalizer: Normalizer,
) -> pl.DataFrame:
    config = Config(config)
    feature_cols, data_feature_matrix = get_feature_matrix(data_df, config)
    data_feature_matrix = normalizer.transform(data_feature_matrix)
    return set_feature_matrix(data_df, feature_cols, data_feature_matrix)
