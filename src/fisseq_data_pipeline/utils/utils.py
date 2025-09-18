import logging
import os
from typing import Tuple

import numpy as np
import polars as pl
import polars.selectors as cs
import sklearn.model_selection

from .config import Config

PlSelector = cs.Selector | pl.Expr

RANDOM_STATE = os.getenv("FISSEQ_PIPELINE_RAND_STATE", 42)


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


def train_test_split(
    feature_df: pl.DataFrame,
    meta_data_df: pl.DataFrame,
    test_size: float,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split feature and metadata DataFrames into stratified train and test sets.

    Parameters
    ----------
    feature_df : pl.DataFrame
        Feature matrix with shape (n_samples, n_features).
    meta_data_df : pl.DataFrame
        Metadata aligned row-wise with ``feature_df``. Must contain columns
        ``_label`` (class labels) and ``_batch`` (batch identifiers).
    test_size : float
        Proportion of the dataset to include in the test split. Should be a
        float between 0.0 and 1.0.

    Returns
    -------
    train_feature_df : pl.DataFrame
        Features for the training set.
    train_meta_data_df : pl.DataFrame
        Metadata for the training set.
    test_feature_df : pl.DataFrame
        Features for the test set.
    test_meta_data_df : pl.DataFrame
        Metadata for the test set.

    Notes
    -----
    - Each ``(_label, _batch)`` group must have at least two samples for
      stratification to succeed.
    - The split is reproducible if ``RANDOM_STATE`` is fixed.
    """
    stratify = [
        f"{label}:{batch}"
        for label, batch in zip(
            meta_data_df.get_column("_label"), meta_data_df.get_column("_batch")
        )
    ]

    train_idx, test_idx = sklearn.model_selection.train_test_split(
        np.arange(len(meta_data_df)),
        test_size=test_size,
        stratify=stratify,
        random_state=RANDOM_STATE,
    )

    train_feature_df = feature_df[train_idx, :]
    train_meta_data_df = meta_data_df[train_idx, :]
    test_feature_df = feature_df[test_idx, :]
    test_meta_data_df = meta_data_df[test_idx, :]
    logging.info(
        "Created train set containing %d samples and test set containing %d samples",
        len(train_idx),
        len(test_idx),
    )

    return train_feature_df, train_meta_data_df, test_feature_df, test_meta_data_df
