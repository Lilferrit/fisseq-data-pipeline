import logging
from typing import Optional
from os import PathLike

import polars as pl

from .utils import Config, get_feature_selector, get_feature_columns


def drop_feature_null(
    data_df: pl.LazyFrame, config: Optional[PathLike | Config]
) -> pl.LazyFrame:
    """
    Drop rows containing null values in feature columns.

    Parameters
    ----------
    data_df : pl.LazyFrame
        Input data frame in lazy mode.
    config : PathLike | Config, optional
        Pipeline configuration or path to a config file. Used to resolve
        the feature selector via `get_feature_selector`.

    Returns
    -------
    pl.LazyFrame
        A lazy frame with rows dropped if they contain NaNs in the
        selected feature columns.
    """
    config = Config(config)
    selector = get_feature_selector(config)
    return data_df.drop_nulls(subset=selector)


def drop_feature_zero_var(
    data_df: pl.LazyFrame, config: Optional[PathLike | Config]
) -> pl.LazyFrame:
    """
    Drop feature columns that contain no variance.

    Parameters
    ----------
    data_df : pl.LazyFrame
        Input data frame in lazy mode.
    config : PathLike | Config, optional
        Pipeline configuration or path to a config file. Used to resolve
        which columns are considered features.

    Returns
    -------
    pl.LazyFrame
        A lazy frame where constant-valued feature columns have been removed,
        and all other columns remain unchanged.
    """
    feature_df = get_feature_columns(data_df, config)
    feature_cols = feature_df.columns
    min_max = (
        data_df.select(
            [pl.col(c).min().alias(f"{c}_min") for c in feature_df.columns]
            + [pl.col(c).max().alias(f"{c}_max") for c in feature_df.columns]
        )
        .collect()
        .row(0, named=True)
    )

    zero_var_cols = [
        c for c in feature_df.columns if min_max[f"{c}_min"] == min_max[f"{c}_max"]
    ]

    if len(zero_var_cols) > 0:
        logging.info("Removing zero variance feature columns: %s", zero_var_cols)

    data_df = data_df.drop(feature_cols)
    return pl.concat((data_df, feature_df), how="horizontal")
