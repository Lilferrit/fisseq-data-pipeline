import logging
import os
from typing import Tuple

import polars as pl
import polars.selectors as cs

from .config import Config

PlSelector = cs.Selector | pl.Expr

RANDOM_STATE = os.getenv("FISSEQ_PIPELINE_RAND_STATE", 42)


def get_feature_selector(data_df: pl.LazyFrame, config: Config) -> PlSelector:
    """
    Build a Polars column selector for feature columns based on the config.

    This utility interprets ``config.feature_cols`` and returns a Polars
    selector expression usable in ``.select()`` or ``.with_columns()`` calls.

    Selection modes:
      - **Regex**: if ``feature_cols`` is a string, all column names matching
        the regex are selected.
      - **Explicit list**: if ``feature_cols`` is a list of strings, those
        columns are selected in the given order. Missing columns are ignored
        with a warning.

    Parameters
    ----------
    data_df : pl.LazyFrame
        Input dataset containing feature columns.
    config : Config
        Configuration object with a ``feature_cols`` attribute defining which
        columns to select (regex pattern or explicit list).

    Returns
    -------
    PlSelector
        A Polars selector expression suitable for use in ``.select()`` calls.

    Notes
    -----
    - Missing columns are ignored but logged as a warning.
    - Column order is preserved when using an explicit list.
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


def get_data_dfs(
    data_df: pl.LazyFrame,
    config: Config,
    dtype: pl.DataType = pl.Float32,
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Construct separate feature and metadata DataFrames from a Polars LazyFrame.

    This function builds a complete computation graph that:
      1. Adds a row index (``_sample_idx``) for reproducible sample tracking.
      2. Selects and casts feature columns defined in ``config.feature_cols``.
      3. Extracts metadata columns:
         - ``_batch`` from ``config.batch_col_name``
         - ``_label`` from ``config.label_col_name``
         - ``_is_control`` by evaluating ``config.control_sample_query``
      4. Materializes the LazyFrame once to produce a feature matrix
         and an eager metadata DataFrame.

    Parameters
    ----------
    data_df : pl.LazyFrame
        Input dataset containing both features and metadata.
    config : Config
        Configuration object
    dtype : pl.DataType, optional
        Data type to cast feature columns to. Defaults to ``pl.Float32``.

    Returns
    -------
    (pl.DataFrame, pl.DataFrame)
        A tuple containing:
          - **feature_df** : eager ``pl.DataFrame`` of numerical features, shape
            ``(n_samples, n_features)``, with values cast to ``dtype``.
          - **meta_data_df** : eager ``pl.DataFrame`` with columns:
                * ``_batch`` — batch labels
                * ``_label`` — class labels
                * ``_is_control`` — boolean control mask
                * ``_sample_idx`` — stable sample index

    Notes
    -----
    - The returned feature and metadata frames share identical row ordering.
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
    logging.info("Creating combined dataframe query")
    lf = base.with_columns(
        label_expr,
        batch_expr,
        control_mask_expr,
    ).select(
        feature_expr,
        pl.col("_batch"),
        pl.col("_is_control"),
        pl.col("_sample_idx"),
        pl.col("_label"),
    )

    # Get feature dataframe
    logging.info("Creating feature dataframe query")
    feature_df = lf.select(feature_expr)

    logging.info("Creating metadata frame query")
    meta_data_df = lf.select(
        pl.col("_batch"),
        pl.col("_label"),
        pl.col("_is_control"),
        pl.col("_sample_idx"),
    )

    return feature_df, meta_data_df


def train_test_split(
    feature_df: pl.LazyFrame,
    meta_data_df: pl.LazyFrame,
    test_size: float,
) -> Tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """
    Split feature and metadata DataFrames into stratified train and test sets.

    Parameters
    ----------
    feature_df : pl.LazyFrame
        Feature matrix with shape (n_samples, n_features).
    meta_data_df : pl.LazyFrame
        Metadata aligned row-wise with ``feature_df``. Must contain columns
        ``_label`` (class labels) and ``_batch`` (batch identifiers).
    test_size : float
        Proportion of the dataset to include in the test split. Should be a
        float between 0.0 and 1.0.

    Returns
    -------
    train_feature_df : pl.LazyFrame
        Features for the training set.
    train_meta_data_df : pl.LazyFrame
        Metadata for the training set.
    test_feature_df : pl.LazyFrame
        Features for the test set.
    test_meta_data_df : pl.LazyFrame
        Metadata for the test set.

    Notes
    -----
    - Each ``(_label, _batch)`` group must have at least two samples for
      stratification to succeed.
    - The split is reproducible if ``RANDOM_STATE`` is fixed.
    """
    logging.info("Creating lazy stratified train/test split query")
    lf_meta = (
        meta_data_df
        .with_row_index("row_id")
        .with_columns(
            pl.concat_str(
                [
                    pl.col("_label").cast(pl.Utf8),
                    pl.lit(":"),
                    pl.col("_batch").cast(pl.Utf8),
                ]
            ).alias("grp")
        )
    )

    lf_test_idx = (
        lf_meta.group_by("grp")
        .agg(pl.col("row_id").sample(fraction=test_size, seed=RANDOM_STATE))
        .explode("row_id")
    )

    lf_mask = (
        lf_meta.join(lf_test_idx, on="row_id", how="left")
        .with_columns(pl.col("grp_right").is_not_null().alias("_is_test"))
        .select(["row_id", "_is_test"])
    )

    logging.info("Split created, executing query")
    is_test = lf_mask.select(pl.col("_is_test")).collect().get_column("_is_test")
    is_train = ~is_test

    logging.info(
        "Split completed: %d train / %d test samples", is_train.sum(), is_test.sum()
    )

    logging.info("Copying data")
    train_feature_df = feature_df.filter(is_train)
    test_feature_df = feature_df.filter(is_test)
    train_meta_data_df = meta_data_df.filter(is_train)
    test_meta_data_df = meta_data_df.filter(is_test)

    return train_feature_df, train_meta_data_df, test_feature_df, test_meta_data_df
