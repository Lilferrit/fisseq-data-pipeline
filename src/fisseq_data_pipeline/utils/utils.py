from typing import List, Tuple

import numpy as np
import polars as pl

from .config import Config


def get_control_samples(data_df: pl.DataFrame, config: Config) -> pl.DataFrame:
    return data_df.sql(f"SELECT * FROM self WHERE {config.control_sample_query}")


def get_feature_matrix(
    data_df: pl.DataFrame, config: Config
) -> Tuple[List[str], np.ndarray]:
    data_df = data_df.select(pl.col(config.feature_cols))
    return data_df.columns, data_df.to_numpy()


def set_feature_matrix(
    data_df: pl.DataFrame,
    feature_cols: List[str],
    new_features: np.ndarray,
) -> pl.DataFrame:
    feature_df = pl.DataFrame(new_features, schema=feature_cols)
    return data_df.drop(feature_df.columns).hstack(feature_df)
