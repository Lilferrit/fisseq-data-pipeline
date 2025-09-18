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

    1) Drop feature columns that are entirely non-finite.
    2) Drop rows that contain any remaining non-finite value across.
    3) Drop feature columns with (near) zero variance.

    All feature columns must be numeric

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
    # Drop columns containing all non-finite values
    feature_df = feature_df.fill_null(float("nan"))
    n_rows = feature_df.height
    is_all_nonfinite = feature_df.select(~pl.all().is_finite().any())
    all_nonfinite_cols = [
        c for c in feature_df.columns if is_all_nonfinite.get_column(c).item()
    ]
    feature_df = feature_df.select(pl.exclude(all_nonfinite_cols))
    logging.info(
        "Removed %d columns containing only non-finite values", len(all_nonfinite_cols)
    )

    # Drop rows containing remaining non-finite values
    row_mask = feature_df.select(pl.all_horizontal(pl.all().is_finite())).to_series()
    feature_df = feature_df.filter(row_mask)
    meta_data_df = meta_data_df.filter(row_mask)
    logging.debug(
        "Dropped %d rows containing non-finite values", n_rows - feature_df.height
    )

    # Drop rows that have 0 variance
    variances = feature_df.var().row(0, named=True)
    zero_var_cols = [
        c for c in feature_df.columns if float(variances[c]) < np.finfo(np.float32).eps
    ]
    feature_df = feature_df.select(pl.exclude(zero_var_cols))
    logging.info("Removed %d columns containing zero variance", len(zero_var_cols))

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
        infrequent label–batch groups removed. Alignment is preserved.
    """
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
