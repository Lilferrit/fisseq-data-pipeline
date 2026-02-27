import datetime
import logging
import os
import pathlib
from os import PathLike
from typing import Optional, Tuple

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


def get_data_lf(
    db_lf: pl.LazyFrame,
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
    base = db_lf.with_row_index(name="_meta_sample_idx").cache()

    # Build feature selector
    feature_expr = get_feature_selector(base, config).cast(dtype=dtype)
    logging.debug("Feature selector resolved: %s", feature_expr)

    # Control mask expr
    logging.debug("Parsing control sample query: %s", config.control_sample_query)

    control_mask_expr = pl.sql_expr(config.control_sample_query).alias(
        "_meta_is_control"
    )
    batch_expr = pl.col(config.batch_col_name).alias("_meta_batch")
    label_expr = pl.col(config.label_col_name).alias("_meta_label")

    # Execute the full plan
    logging.info("Creating combined dataframe query")
    lf = base.with_columns(
        label_expr,
        batch_expr,
        control_mask_expr,
    ).select(
        feature_expr,
        pl.col("_meta_batch"),
        pl.col("_meta_is_control"),
        pl.col("_meta_sample_idx"),
        pl.col("_meta_label"),
    )

    return lf


def get_feature_cols(
    data_lf: pl.LazyFrame, as_string: bool = False
) -> list[pl.Expr] | list[str]:
    wrapper = str if as_string else pl.col
    return [wrapper(c) for c in data_lf.columns if not c.startswith("_meta")]


def get_feature_lf(data_lf: pl.LazyFrame) -> pl.LazyFrame:
    return data_lf.select(get_feature_cols(data_lf))


def setup_logging(log_dir: Optional[PathLike] = None) -> None:
    """
    Configure logging for the pipeline.

    A log file and a console stream are set up simultaneously.
    The log file is created in the specified directory (or the current
    working directory by default) with a timestamped filename.
    The log level is controlled by the environment variable
    ``FISSEQ_PIPELINE_LOG_LEVEL`` (default: ``"info"``).

    Parameters
    ----------
    log_dir : PathLike, optional
        Directory where log files will be written.
        If ``None``, the current working directory is used.
    """
    log_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    if log_dir is None:
        log_dir = pathlib.Path.cwd()
    else:
        log_dir = pathlib.Path(log_dir)

    dt_str = datetime.datetime.now().strftime("%Y%m%d:%H%M%S")
    filename = f"fisseq-data-pipeline-{dt_str}.log"
    log_path = log_dir / filename
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path, mode="w")]

    log_level = os.getenv("FISSEQ_PIPELINE_LOG_LEVEL", "info")
    log_level = log_levels.get(log_level, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] [%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
