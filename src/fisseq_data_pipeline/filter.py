import logging
import os
from typing import Tuple

import numpy as np
import polars as pl

MINIMUM_CLASS_MEMBERS = os.getenv("FISSEQ_PIPELINE_MIN_CLASS_MEMBERS", 2)


def clean_data(
    feature_df: pl.DataFrame, meta_data_df: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Clean feature and metadata tables and keep them row-aligned.

    The cleaning pipeline performs the following passes:

    1) Drop feature columns that are entirely non-finite.
    2) Drop rows that contain any remaining non-finite value.

    All feature columns must be numeric.

    Parameters
    ----------
    feature_df : pl.DataFrame
        Feature-only table, shape (n_samples, n_features).
    meta_data_df : pl.DataFrame
        Metadata aligned row-wise with ``feature_df``. Must contain ``"_label"``
        and ``"_batch"`` columns.

    Returns
    -------
    (pl.DataFrame, pl.DataFrame)
        The cleaned ``(feature_df, meta_data_df)``, with invalid rows/columns
        removed and row alignment preserved.
    """
    logging.info("Creating data cleaning query")
    lf = feature_df.lazy()

    # Pass 1: Drop columns that are entirely non-finite
    finite_summary = lf.select(
        [pl.col(c).is_finite().any().alias(c) for c in feature_df.columns]
    ).collect()
    non_finite_cols = [c for c in feature_df.columns if not finite_summary[c][0]]
    lf = lf.select(pl.exclude(non_finite_cols))
    logging.info(
        "Removing %d columns containing only non-finite values",
        len(non_finite_cols),
    )

    # Pass 2: Drop rows containing any non-finite values
    row_mask = lf.select(pl.all_horizontal(pl.all().is_finite())).collect().to_series()
    lf = lf.filter(row_mask)
    meta_data_df = meta_data_df.filter(row_mask)
    logging.info(
        "Dropping %d rows containing non-finite values",
        feature_df.height - int(row_mask.sum()),
    )

    # Execute query
    logging.info("Executing query")
    feature_df = lf.collect()

    return feature_df, meta_data_df


def drop_infrequent_pairs(
    feature_df: pl.DataFrame, meta_data_df: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Remove rows belonging to rare (``_label``, ``_batch``) groups.

    Rows are grouped by the concatenation of ``_label`` and ``_batch``.
    Any group with a sample count less than
    ``FISSEQ_PIPELINE_MIN_CLASS_MEMBERS`` (default: 2) is dropped.

    Parameters
    ----------
    feature_df : pl.DataFrame
        Feature-only table of shape (n_samples, n_features).
    meta_data_df : pl.DataFrame
        Metadata table row-aligned with ``feature_df``.
        Must include ``"_label"`` and ``"_batch"`` columns.

    Returns
    -------
    (pl.DataFrame, pl.DataFrame)
        A tuple ``(feature_df, meta_data_df)`` with rows from
        infrequent label-batch groups removed. Alignment is preserved.
    """
    n_rows = feature_df.height

    # Compute label_batch frequencies lazily
    lf = meta_data_df.lazy().with_columns(
        (pl.col("_label") + "_" + pl.col("_batch")).alias("_label_batch")
    )

    # Get per-combo counts and join back to metadata
    lf = lf.join(
        lf.group_by("_label_batch").agg(pl.count().alias("_count")),
        on="_label_batch",
        how="left",
    )

    # Boolean mask for frequent pairs
    mask_df = lf.select(
        (pl.col("_count") >= MINIMUM_CLASS_MEMBERS).alias("_keep")
    ).collect()
    mask = mask_df["_keep"]

    # Apply to both tables (row-aligned)
    feature_df = feature_df.filter(mask)
    meta_data_df = meta_data_df.filter(mask)

    logging.info(
        "Dropped %d rows containing rare (batch, label) pairs (<%d)",
        n_rows - feature_df.height,
        MINIMUM_CLASS_MEMBERS,
    )

    return feature_df, meta_data_df
