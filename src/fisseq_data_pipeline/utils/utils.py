from typing import Collection, List, Tuple

import numpy as np
import polars as pl
import polars.selectors as cs
from polars.datatypes.classes import FloatType

from .config import Config


def get_control_samples(data_df: pl.LazyFrame, config: Config) -> pl.LazyFrame:
    """
    Filter the input LazyFrame to include only control samples based on the query
    specified in the configuration.

    Parameters
    ----------
    data_df : pl.LazyFrame
        The input data as a Polars LazyFrame.
    config : Config
        Configuration object containing a ``control_sample_query`` attribute
        with a SQL WHERE clause fragment.

    Returns
    -------
    pl.LazyFrame
        A LazyFrame containing only the control samples.
    """
    return data_df.sql(f"SELECT * FROM self WHERE {config.control_sample_query}")


def get_feature_matrix(
    data_df: pl.LazyFrame, config: Config, dtype: FloatType = pl.Float32
) -> Tuple[List[str], np.ndarray]:
    """
    Extract the feature matrix from the LazyFrame based on columns defined in
    the configuration.

    Parameters
    ----------
    data_df : pl.LazyFrame
        The input data as a Polars LazyFrame.
    config : Config
        Configuration object containing a ``feature_cols`` attribute
        (string regex pattern or list of column names).
    dtype : polars.datatypes.DataType, default=pl.Float32
        The data type to cast the feature columns to.

    Returns
    -------
    feature_cols : list of str
        The list of selected feature column names.
    feature_matrix : numpy.ndarray
        A 2D NumPy array containing the feature values.
    """
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
    """
    Replace the feature columns in a LazyFrame with new values provided
    as a NumPy array.

    Parameters
    ----------
    data_df : pl.LazyFrame
        The input data as a Polars LazyFrame.
    feature_cols : list of str
        The names of the feature columns to replace.
    new_features : numpy.ndarray
        A 2D NumPy array of replacement feature values with shape
        (n_rows, len(feature_cols)).

    Returns
    -------
    pl.LazyFrame
        A LazyFrame containing the updated feature columns alongside
        all non-feature columns from the original input.
    """
    feature_df = pl.LazyFrame(new_features, schema=feature_cols)
    data_df = data_df.drop(feature_df.columns)
    return pl.concat((feature_df, data_df), how="horizontal")


def get_rows_by_idx(
    data_df: pl.LazyFrame,
    rows: Collection[int],
    idx_col_name: str = "_fisseq_data_pipeline_cell_idx",
) -> pl.LazyFrame:
    """
    Select specific rows from a LazyFrame by integer indices.

    Parameters
    ----------
    data_df : pl.LazyFrame
        The input data as a Polars LazyFrame.
    rows : collection of int
        The integer indices of the rows to retrieve.
    idx_col_name : str, default="_fisseq_data_pipeline_cell_idx"
        The temporary name for the index column added to the LazyFrame.

    Returns
    -------
    pl.LazyFrame
        A LazyFrame containing only the specified rows, with the
        temporary index column removed.
    """
    rows = set(rows)
    data_df = data_df.with_row_index(idx_col_name)
    data_df = data_df.filter(pl.col(idx_col_name).is_in(rows))
    return data_df.drop(idx_col_name)
