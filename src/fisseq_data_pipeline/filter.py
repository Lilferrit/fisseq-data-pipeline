import logging
import os
from typing import Callable, Iterable, Tuple, TypeAlias

import polars as pl

MINIMUM_CLASS_MEMBERS = os.getenv("FISSEQ_PIPELINE_MIN_CLASS_MEMBERS", 2)

FilterFun: TypeAlias = Callable[
    [pl.LazyFrame, pl.LazyFrame], Tuple[pl.LazyFrame, pl.LazyFrame]
]


def drop_rows_infrequent_pairs(
    feature_df: pl.LazyFrame, meta_data_df: pl.LazyFrame
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Remove rows belonging to rare (``_label``, ``_batch``) groups.

    Rows are grouped by the concatenation of ``_label`` and ``_batch``.
    Any group with a sample count less than
    ``MINIMUM_CLASS_MEMBERS`` is dropped.

    Parameters
    ----------
    feature_df : pl.LazyFrame
        Feature-only table of shape (n_samples, n_features).
    meta_data_df : pl.LazyFrame
        Metadata table row-aligned with ``feature_df``.
        Must include ``"_label"`` and ``"_batch"`` columns.

    Returns
    -------
    (pl.LazyFrame, pl.LazyFrame)
        LazyFrames with infrequent (label, batch) groups removed.
        Row order and alignment are preserved.
    """
    logging.info("Adding query to remove infrequent batch/variant pairs")

    # Add combined key + stable row index
    meta_data_df = meta_data_df.with_row_index("_idx").with_columns(
        (pl.col("_label") + "_" + pl.col("_batch")).alias("_label_batch")
    )

    # Compute per-(label,batch) counts and join back
    counts_df = meta_data_df.group_by("_label_batch").agg(pl.count().alias("_count"))

    meta_data_df = meta_data_df.join(counts_df, on="_label_batch", how="left")

    # Attach matching index to feature_df
    feature_df = feature_df.with_row_index("_idx")

    # Join count info to feature_df via _idx (so we can filter both)
    feature_df = feature_df.join(
        meta_data_df.select("_idx", "_count"),
        on="_idx",
        how="left",
    )

    # Filter both lazily
    keep_expr = pl.col("_count") >= MINIMUM_CLASS_MEMBERS
    exclude_cols = pl.exclude(["_count", "_idx"])
    feature_df = feature_df.filter(keep_expr).select(exclude_cols)
    meta_data_df = meta_data_df.filter(keep_expr).select(exclude_cols)

    return feature_df, meta_data_df


def drop_cols_all_nonfinite(
    feature_df: pl.LazyFrame, meta_data_df: pl.LazyFrame
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Remove feature columns that contain only non-finite values.

    Any column in ``feature_df`` where all entries are NaN, +inf, or -inf
    is excluded from the output.

    Parameters
    ----------
    feature_df : pl.LazyFrame
        LazyFrame containing numerical feature columns.
    meta_data_df : pl.LazyFrame
        LazyFrame containing sample-level metadata (returned unchanged).

    Returns
    -------
    (pl.LazyFrame, pl.LazyFrame)
        Tuple of (feature_df, meta_data_df) where feature_df excludes
        all columns consisting solely of non-finite values.
    """
    # TODO: Can this also be a lazy expression?
    logging.info("Scanning for all non-finite rows")
    finite_summary = feature_df.select(
        [pl.col(c).is_finite().any().alias(c) for c in feature_df.columns]
    ).collect()

    logging.info("Adding query to remove columns containing all non-finite values")
    non_finite_cols = [c for c in feature_df.columns if not finite_summary[c][0]]
    feature_df = feature_df.select(pl.exclude(non_finite_cols))
    logging.info(
        "Removing %d columns containing only non-finite values",
        len(non_finite_cols),
    )

    return feature_df, meta_data_df


def drop_rows_any_nonfinite(
    feature_df: pl.LazyFrame, meta_data_df: pl.LazyFrame
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Remove rows that contain any non-finite feature values.

    This step filters out all samples where at least one feature value
    is NaN, +inf, or -inf. Row alignment between the feature and metadata
    tables is preserved.

    Parameters
    ----------
    feature_df : pl.LazyFrame
        LazyFrame of numerical features.
    meta_data_df : pl.LazyFrame
        Row-aligned LazyFrame of metadata.

    Returns
    -------
    (pl.LazyFrame, pl.LazyFrame)
        LazyFrames excluding all rows with any non-finite feature values.
    """
    logging.info(
        "Adding query to remove any remaining rows that contain non-finite"
        " feature values"
    )
    feature_df = feature_df.with_row_index("_idx").with_columns(
        pl.all_horizontal(pl.all().is_finite()).alias("_all_finite")
    )
    meta_data_df = meta_data_df.with_row_index("_idx").join(
        feature_df.select(pl.col("_idx"), pl.col("_all_finite")), on="_idx"
    )

    filter_expr = pl.col("_all_finite")
    exclude_expr = pl.exclude(["_all_finite", "_idx"])
    feature_df = feature_df.filter(filter_expr).select(exclude_expr)
    meta_data_df = meta_data_df.filter(filter_expr).select(exclude_expr)

    return feature_df, meta_data_df


def clean_data(
    feature_df: pl.LazyFrame,
    meta_data_df: pl.LazyFrame,
    stages: Iterable[str | FilterFun] = [
        "drop_cols_all_nonfinite",
        "drop_rows_any_nonfinite",
        "drop_rows_infrequent_pairs",
    ],
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Sequentially apply a series of filters to feature and metadata LazyFrames.

    Each stage may be specified by name or as a callable implementing
    the ``FilterFun`` signature. Stages are executed in order, and each
    must return updated (feature_df, meta_data_df) pairs.

    Parameters
    ----------
    feature_df : pl.LazyFrame
        LazyFrame containing numerical features.
    meta_data_df : pl.LazyFrame
        LazyFrame containing associated metadata.
    stages : Iterable[str | FilterFun], optional
        Ordered list of stage names or callables. The default pipeline includes:
          - ``"drop_cols_all_nonfinite"`` — remove columns that are all NaN/inf
          - ``"drop_rows_any_nonfinite"`` — drop rows with any non-finite values
          - ``"drop_rows_infrequent_pairs"`` — drop small (label,batch) groups

    Returns
    -------
    (pl.LazyFrame, pl.LazyFrame)
        The cleaned feature and metadata LazyFrames.

    Notes
    -----
    - Invalid stage names are skipped with a warning.
    """
    stage_lookup: dict[str, FilterFun] = {
        "drop_cols_all_nonfinite": drop_cols_all_nonfinite,
        "drop_rows_any_nonfinite": drop_rows_any_nonfinite,
        "drop_rows_infrequent_pairs": drop_rows_infrequent_pairs,
    }

    for stage in stages:
        if isinstance(stage, str):
            if stage not in stage_lookup:
                logging.warning("Skipping invalid filtering stage: %s", stage)
                continue
            stage = stage_lookup[stage]
        feature_df, meta_data_df = stage(feature_df, meta_data_df)

    return feature_df, meta_data_df
