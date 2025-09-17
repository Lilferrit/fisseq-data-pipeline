import logging
from typing import Tuple

import numpy as np
import polars as pl


def clean_data(
    feature_df: pl.DataFrame, meta_data_df: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Remove unusable rows/columns from features and align metadata.

    This function computes masks for:
      - columns that are entirely NaN,
      - columns with (near) zero variance, and
      - rows containing NaNs in any non-all-NaN column.

    It then drops flagged rows/columns from the feature matrix and applies the
    same row filtering to the metadata so the two remain aligned.

    Parameters
    ----------
    feature_df : pl.DataFrame
        Feature columns only; shape (n_samples, n_features).
    meta_data_df : pl.DataFrame
        Metadata columns (e.g., _batch, _label, _is_control), aligned by rows
        with `feature_df`.

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame]
        Cleaned `(feature_df, meta_data_df)` with invalid rows/columns removed
        and row alignment preserved.
    """
    n_rows = feature_df.height
    null_counts = feature_df.null_count().row(0, named=True)
    all_null_cols = [c for c in feature_df.columns if int(null_counts[c]) == n_rows]
    feature_df = feature_df.select(pl.exclude(all_null_cols))
    logging.info("Removed %d columns containing all null values", len(all_null_cols))

    row_mask = feature_df.select(~pl.any_horizontal(pl.all().is_null())).to_series()
    feature_df = feature_df.filter(row_mask)
    meta_data_df = meta_data_df.filter(row_mask)
    logging.debug("Dropped %d rows containing null values", n_rows - feature_df.height)

    variances = feature_df.var().row(0, named=True)
    zero_var_cols = [
        c for c in feature_df.columns if float(variances[c]) < np.finfo(np.float32).eps
    ]
    feature_df = feature_df.select(pl.exclude(zero_var_cols))
    logging.info("Removed %d columns containing zero variance", len(zero_var_cols))

    return feature_df, meta_data_df
