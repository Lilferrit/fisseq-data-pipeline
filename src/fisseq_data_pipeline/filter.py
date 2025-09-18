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

    The cleaning pipeline performs four passes:

    1) Drop feature columns that are entirely null.
    2) Drop rows that contain any null across the remaining feature columns.
    3) Drop feature columns with (near) zero variance.
    4) Enforce a minimum frequency for (``_label``, ``_batch``) groups.

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
    # Drop columns containing all null values
    n_rows = feature_df.height
    null_counts = feature_df.null_count().row(0, named=True)
    all_null_cols = [c for c in feature_df.columns if int(null_counts[c]) == n_rows]
    feature_df = feature_df.select(pl.exclude(all_null_cols))
    logging.info("Removed %d columns containing all null values", len(all_null_cols))

    # Drop rows containing null values
    row_mask = feature_df.select(~pl.any_horizontal(pl.all().is_null())).to_series()
    feature_df = feature_df.filter(row_mask)
    meta_data_df = meta_data_df.filter(row_mask)
    logging.debug("Dropped %d rows containing null values", n_rows - feature_df.height)

    # Drop rows that have 0 variance
    variances = feature_df.var().row(0, named=True)
    zero_var_cols = [
        c for c in feature_df.columns if float(variances[c]) < np.finfo(np.float32).eps
    ]
    feature_df = feature_df.select(pl.exclude(zero_var_cols))
    logging.info("Removed %d columns containing zero variance", len(zero_var_cols))

    # Drop rows that have infrequent (batch, label) batch pairs
    n_rows = feature_df.height
    label_batch = (
        meta_data_df.get_column("_label") + "_" + meta_data_df.get_column("_batch")
    ).alias("_label_batch")
    label_batch_counts = label_batch.value_counts().filter(
        pl.col("count") >= MINIMUM_CLASS_MEMBERS
    )
    label_batch_freq_mask = label_batch.is_in(
        label_batch_counts.get_column("_label_batch")
    )
    feature_df = feature_df.filter(label_batch_freq_mask)
    meta_data_df = meta_data_df.filter(label_batch_freq_mask)
    logging.info(
        "Dropped %d rows containing batch label pairs with frequency less than %d",
        n_rows - feature_df.height,
        MINIMUM_CLASS_MEMBERS,
    )

    return feature_df, meta_data_df
