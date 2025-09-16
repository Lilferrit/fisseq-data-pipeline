import logging
from typing import List, Optional, Tuple

import numpy as np
import polars as pl
import polars.selectors as cs

from .config import Config

PlSelector = cs.Selector | pl.Expr


def get_feature_selector(data_df: pl.LazyFrame, config: Config) -> PlSelector:
    """
    Get feature column feature selector based on config

    Parameters
    ----------
    data_df : pl.LazyFrame
        The input data as a Polars LazyFrame.
    config : Config
        Configuration object containing a ``feature_cols`` attribute, which may
        be either:
          - str : A regex pattern to match column names.
          - list[str] : A list of explicit column names to select.

    Returns
    -------
    PlSelector
        A selector that can be used to select feature columns in ``.select``
        call.
    """
    if isinstance(config.feature_cols, str):
        selector_type = "regex"
        selector = cs.matches(config.feature_cols)
    else:
        selector_type = "list"
        feature_cols = set(config.feature_cols)
        missing = feature_cols - set(data_df.columns)

        if len(missing) != 0:
            logging.warning(
                "Some columns are specified in the config but are not currently"
                " present in the dataframe, this can happen if columns are"
                " removed during data cleaning. The following columns will be"
                " ignored: %s",
                missing,
            )

        # Column order must be preserved
        not_missing = feature_cols - missing
        selector = pl.col(
            col for col in list(config.feature_cols) if col in not_missing
        )

    logging.debug("Using feature %s selector: %s", selector_type, config.feature_cols)
    return selector


def get_feature_columns(data_df: pl.LazyFrame, config: Config) -> pl.LazyFrame:
    """
    Select feature columns from a Polars LazyFrame based on the configuration

    Parameters
    ----------
    data_df : pl.LazyFrame
        The input data as a Polars LazyFrame.
    config : Config
        Configuration object containing a ``feature_cols`` attribute, which may
        be either:
          - str : A regex pattern to match column names.
          - list[str] : A list of explicit column names to select.

    Returns
    -------
    pl.LazyFrame
        A Polars LazyFrame containing only the selected feature columns, cast to
        the given dtype.
    """
    selector = get_feature_selector(data_df, config)
    data_df = data_df.select(selector)
    return data_df


def get_data_dfs(
    data_df: pl.LazyFrame,
    config: Config,
    dtype: pl.DataType = pl.Float32,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Construct a numeric feature matrix and a metadata DataFrame from a
    Polars LazyFrame.

    The computation graph:
      1. Adds a row index column (``_sample_idx``) for stable sample
         tracking.
      2. Selects feature columns defined in ``config.feature_cols`` and
         casts them to ``dtype``.
      3. Extracts the batch column (``_batch``), the label column
         (``_label``), and a boolean control mask (``_is_control``) using
         ``config.control_sample_query``.
      4. Collects the LazyFrame once and returns NumPy features plus
         eager Polars metadata.

    Parameters
    ----------
    data_df : pl.LazyFrame
        Input dataset as a Polars LazyFrame.
    config : Config
        Configuration object containing:
          - ``feature_cols``: regex or list of feature column names.
          - ``batch_col_name``: column name to be exposed as ``_batch``.
          - ``label_col_name``: column name to be exposed as ``_label``.
          - ``control_sample_query``: SQL WHERE clause fragment defining
            control samples (used to produce ``_is_control``).
    dtype : pl.DataType, optional
        Target dtype for feature columns (default: ``pl.Float32``).

    Returns
    -------
    Tuple[np.ndarray, pl.DataFrame]
        - Feature matrix: ``np.ndarray`` of shape ``(n_samples, n_features)``,
          with columns from ``config.feature_cols`` cast to ``dtype``.
        - Metadata DataFrame: eager ``pl.DataFrame`` with columns:
            * ``_batch``: values from ``config.batch_col_name``.
            * ``_label``: values from ``config.label_col_name``.
            * ``_is_control``: bool mask indicating control samples.
            * ``_sample_idx``: integer row index for reference.
    """
    logging.info(
        "Starting get_data_matrices for batch_col=%s, dtype=%s",
        config.batch_col_name,
        dtype,
    )

    # Attach row indices to preserve mapping later
    base = data_df.with_row_index(name="_sample_idx").cache()

    # Build feature selector
    feature_expr = get_feature_selector(base, config).cast(dtype=dtype)
    logging.debug("Feature selector resolved: %s", feature_expr)

    # Control mask expr
    logging.debug("Parsing control sample query: %s", config.control_sample_query)

    control_mask_expr = pl.sql_expr(config.control_sample_query).alias("_is_control")
    batch_expr = pl.col(config.batch_col_name).alias("_batch")
    label_expr = pl.col(config.label_col_name).alias("_label")

    # Execute the full plan
    logging.info("Collecting LazyFrame into DataFrame")
    df = (
        base.with_columns(
            label_expr,
            batch_expr,
            control_mask_expr,
        )
        .select(
            feature_expr,
            pl.col("_batch"),
            pl.col("_is_control"),
            pl.col("_sample_idx"),
            pl.col("_label"),
        )
        .collect()
    )
    logging.info("Collection complete: shape=%s", df.shape)

    # Get feature dataframe
    feature_df = df.select(feature_expr)
    logging.debug("Feature dataframe shape: %s", feature_df.shape)

    meta_data_df = df.select(
        pl.col("_batch"), pl.col("_label"), pl.col("_is_control"), pl.col("_sample_idx")
    )

    logging.debug("Meta data dataframe: shape=%s", meta_data_df.shape)
    logging.info("Finished get_data_matrices")

    return feature_df, meta_data_df


def set_feature_matrix(
    meta_data_df: pl.LazyFrame,
    feature_cols: List[str],
    new_features: np.ndarray,
) -> pl.LazyFrame:
    """
    Replace selected feature columns in a LazyFrame with new values.

    This function constructs a new LazyFrame from the provided feature
    matrix and replaces the specified feature columns in the metadata
    frame. If a boolean feature mask is provided, only the subset of
    feature columns where the mask is True will be replaced.

    Parameters
    ----------
    meta_data_df : pl.LazyFrame
        Input dataset as a Polars LazyFrame containing metadata and
        original feature columns.
    feature_cols : list of str
        Names of the feature columns to replace.
    new_features : np.ndarray
        2D NumPy array of replacement values with shape
        (n_rows, n_selected_features). The number of columns must
        match the number of feature columns being replaced.
    feature_mask : np.ndarray, optional
        Boolean mask of shape (len(feature_cols),). If provided,
        only the feature columns corresponding to True entries in
        the mask are replaced.

    Returns
    -------
    pl.LazyFrame
        A LazyFrame containing the updated feature columns alongside
        all non-feature columns from the original input.
    """
    logging.debug("Updating feature columns %s", feature_cols)
    feature_df = pl.LazyFrame(new_features, schema=feature_cols)
    return pl.concat((feature_df, meta_data_df), how="horizontal")
