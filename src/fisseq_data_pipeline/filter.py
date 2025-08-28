import logging
from os import PathLike
from typing import Callable, List, Optional

import polars as pl

from .utils import Config, get_feature_columns, get_feature_selector

FilterFun = Callable[[pl.LazyFrame, Optional[PathLike | Config]], pl.LazyFrame]


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
    selector = get_feature_selector(data_df, config)
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


def run_sequential_filters(
    data_df: pl.LazyFrame,
    config: Optional[PathLike | Config],
    filter_funs: List[FilterFun] = [drop_feature_null, drop_feature_zero_var],
) -> pl.LazyFrame:
    """
    Apply a list of filter functions to a LazyFrame in order.

    Parameters
    ----------
    data_df : pl.LazyFrame
        The input dataset in lazy mode.
    config : PathLike | Config, optional
        Pipeline configuration (or path) to pass through to each filter
        function.
    filter_funs : list[FilterFun],
    default=[drop_feature_null, drop_feature_zero_var]
        An ordered list of filter functions to run.

    Returns
    -------
    pl.LazyFrame
        The transformed LazyFrame after all filters have been applied in
        sequence.
    """
    for curr_filter_fun in filter_funs:
        data_df = curr_filter_fun(data_df, config)
        print(data_df)

    return data_df
