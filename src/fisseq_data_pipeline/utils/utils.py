from typing import Collection, List, Tuple

import numpy as np
import polars as pl
import polars.selectors as cs
from polars.datatypes.classes import FloatType

from .config import Config


def get_control_samples(data_df: pl.LazyFrame, config: Config) -> pl.LazyFrame:
    return data_df.sql(f"SELECT * FROM self WHERE {config.control_sample_query}")


def get_feature_matrix(
    data_df: pl.LazyFrame, config: Config, dtype: FloatType = pl.Float32
) -> Tuple[List[str], np.ndarray]:
    if isinstance(config.feature_cols, str):
        selector = cs.matches(config.feature_cols)
    else:
        selector = pl.col(list(config.feature_cols))

    data_df = data_df.select(selector).cast(dtype)
    feature_cols = list(data_df.schema.keys())
    data_df = data_df.collect()

    return feature_cols, data_df.to_numpy()


def set_feature_matrix(
    data_df: pl.LazyFrame,
    feature_cols: List[str],
    new_features: np.ndarray,
) -> pl.LazyFrame:
    feature_df = pl.LazyFrame(new_features, schema=feature_cols)
    data_df = data_df.drop(feature_df.columns)
    return pl.concat((feature_df, data_df), how="horizontal")


def get_rows_by_idx(
    data_df: pl.LazyFrame,
    rows: Collection[int],
    idx_col_name: str = "_fisseq_data_pipeline_cell_idx",
) -> pl.LazyFrame:
    rows = set(rows)
    data_df = data_df.with_row_index(idx_col_name)
    data_df = data_df.filter(pl.col(idx_col_name).is_in(rows))
    return data_df.drop(idx_col_name)
