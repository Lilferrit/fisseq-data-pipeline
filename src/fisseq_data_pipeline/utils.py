import glob as _glob
import logging
import pathlib
import re
from typing import Any

import polars as pl

from .config import AppConfig
from .constants import (
    META_BATCH_COL,
    META_SELECTOR,
    CONTROL_COLUMN_NAME,
    FEATURE_SELECTOR,
    IMPACT_SCORE_COL,
)

NORM_COL = "tmp_row_norm"
DOT_COL = "tmp_dot_product"


def setup_logging(cfg: AppConfig, name: str) -> None:
    """
    Configure logging for the pipeline.

    A timestamped log file and a console stream are set up simultaneously.
    The log file is named ``{name}.{cfg.output_root}.log``. Its location follows
    the same convention used for other output files:

    Parameters
    ----------
    cfg : AppConfig
        Application configuration supplying ``output_dir`` and optionally
        ``output_root``.
    name : str
        Base name for the log file, typically the calling module or command
        (e.g. ``"normalize"``).
    """
    log_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    if cfg.output_root is not None:
        name = f"{cfg.output_root}.{name}"

    name = f"{name}.log"
    log_path = pathlib.Path(cfg.output_dir) / name
    log_level = log_levels.get(cfg.log_level, logging.INFO)
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path, mode="w")]

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] [%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def get_column(lf: pl.LazyFrame, col: str) -> list[Any]:
    return lf.select(pl.col(col)).collect().get_column(col).to_list()


def classify_variant(v: str) -> str:
    """
    Classify a variant label string into a biological category.

    Parameters
    ----------
    v : str
        Variant label string (e.g. ``"A123G"``, ``"A123fs"``, ``"WT"``).

    Returns
    -------
    str
        One of: ``"Frameshift"``, ``"3nt Deletion"``, ``"Nonsense"``,
        ``"WT"``, ``"Synonymous"``, ``"Single Missense"``, or ``"Other"``.
    """
    if "fs" in v:
        return "Frameshift"
    if v.endswith("-"):
        parts = v.split("|")
        n = len(parts)
        if n == 1:
            return "3nt Deletion"
        if n == 2 and int(parts[0][1:-1]) == int(parts[1][1:-1]) - 1:
            return "3nt Deletion"
        return "Other"
    if "X" in v or "*" in v:
        return "Nonsense"
    if "WT" in v:
        return "WT"
    m = re.match(r"([A-Z])(\d+)([A-Z])", v)
    if m is None:
        return "Other"
    return "Synonymous" if m.group(1) == m.group(3) else "Single Missense"


def load_batches(glob_pattern: str) -> tuple[pl.LazyFrame, str]:
    """
    Load all Parquet files matching a glob pattern and label each with its filename stem.

    Each matching file is scanned lazily and annotated with a ``META_BATCH_COL``
    (``meta_batch``) column set to the file's stem (filename without extension).
    The frames are concatenated in sorted path order.

    Parameters
    ----------
    glob_pattern : str
        Glob pattern for input Parquet files (e.g. ``"data/batches/*.parquet"``).
        A concrete path (no wildcards) is treated as a single-file pattern.

    Returns
    -------
    lf : pl.LazyFrame
        Concatenated lazy frame with an added ``meta_batch`` column.
    output_stem : str
        Suggested output filename stem: the matched file's stem when exactly one
        file matches, otherwise ``"output"``.

    Raises
    ------
    ValueError
        If no files match ``glob_pattern``.
    """
    paths = sorted(_glob.glob(glob_pattern))
    if not paths:
        raise ValueError(f"No files matched glob pattern: {glob_pattern!r}")
    logging.info("Found %d batch file(s) matching %r", len(paths), glob_pattern)
    output_stem = pathlib.Path(paths[0]).stem if len(paths) == 1 else "output"
    lf = pl.concat(
        [
            pl.scan_parquet(p).with_columns(
                pl.lit(pathlib.Path(p).stem).alias(META_BATCH_COL)
            )
            for p in paths
        ]
    )
    return lf, output_stem


def get_aggregate_meta_data(lf: pl.LazyFrame, label_col: str) -> pl.LazyFrame:
    """
    Compute per-variant metadata statistics from a LazyFrame.

    Always produces ``meta_num_cells`` (row count per label group). If the
    frame contains a ``meta_barcode`` column, two additional columns are
    produced: ``meta_num_unique_barcodes`` (distinct barcode count per group)
    and ``meta_barcode_counts`` (per-barcode frequencies as a list of structs
    ``{meta_barcode: str, count: u32}``).

    Parameters
    ----------
    lf : pl.LazyFrame
        Input LazyFrame. Only columns matched by ``META_SELECTOR`` (i.e. those
        with a ``meta_`` prefix) are used in the aggregation.
    label_col : str
        Name of the column identifying variant labels, used as the group key.

    Returns
    -------
    pl.LazyFrame
        One row per label group with columns ``label_col``,
        ``meta_num_cells``, and (when present) ``meta_num_unique_barcodes``
        and ``meta_barcode_counts``.
    """
    label_lgb = lf.select(META_SELECTOR).group_by(label_col)
    agg_exprs = [pl.col(label_col).count().alias("meta_num_cells")]

    if "meta_barcode" in lf.collect_schema().names():
        agg_exprs.extend(
            [
                pl.col("meta_barcode").n_unique().alias("meta_num_unique_barcodes"),
                pl.col("meta_barcode").value_counts().alias("meta_barcode_counts"),
            ]
        )

    return label_lgb.agg(agg_exprs)


def compute_norm(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute the L2 norm of all feature columns for each row.

    Adds a ``NORM_COL`` (``"tmp_row_norm"``) column equal to
    ``sqrt(sum(x_i ** 2))`` over all non-``meta_*`` columns.
    ``meta_*`` columns are left unchanged.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input frame containing feature columns (any column not prefixed
        with ``meta_``).

    Returns
    -------
    pl.LazyFrame
        Same frame with an additional ``tmp_row_norm`` column.
    """
    return lf.with_columns(
        pl.sum_horizontal(FEATURE_SELECTOR.pow(2)).sqrt().alias(NORM_COL)
    )


def compute_query_dot(value_lf: pl.LazyFrame, query_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute the dot product between each row of ``value_lf`` and a single query row.

    Cross-joins ``value_lf`` with the single-row ``query_lf``, multiplies
    corresponding feature columns element-wise, and sums the products into a
    ``DOT_COL`` (``"tmp_dot_product"``) column. The suffixed query columns are
    dropped from the result; all other columns from ``value_lf`` are preserved.

    Parameters
    ----------
    value_lf : pl.LazyFrame
        Frame whose rows will each be dotted against the query.
    query_lf : pl.LazyFrame
        Exactly one-row frame whose feature columns are used as the query vector.

    Returns
    -------
    pl.LazyFrame
        ``value_lf`` with ``tmp_dot_product`` appended and the ``_query``-suffixed
        columns dropped.

    Raises
    ------
    ValueError
        If ``query_lf`` does not contain exactly one row.
    """
    query_len = query_lf.select(pl.len()).collect().item()
    if query_len != 1:
        raise ValueError(f"query must have exactly 1 row, got {query_len}")

    feature_cols = list(value_lf.select(FEATURE_SELECTOR).collect_schema().names())
    joined_lf = value_lf.join(query_lf, how="cross", suffix="_query")

    return joined_lf.with_columns(
        pl.sum_horizontal(
            pl.col(col) * pl.col(f"{col}_query") for col in feature_cols
        ).alias(DOT_COL)
    ).drop([f"{col}_query" for col in feature_cols])


def compute_impact_score(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute a cosine-distance-based impact score relative to the control median.

    The impact score measures how far each row's feature vector is from the
    median feature vector of control rows (``meta_is_control == True``):

        impact = (1 - cos_similarity) / 2

    This maps 0 (identical direction to control) → 0, orthogonal → 0.5,
    and opposite direction → 1.

    Parameters
    ----------
    lf : pl.LazyFrame
        Frame with feature columns and a boolean ``meta_is_control`` column.
        Must contain at least one control row. Feature columns that contain any
        null values are excluded from the cosine similarity calculation but are
        kept in the returned frame.

    Returns
    -------
    pl.LazyFrame
        Input frame with ``meta_impact_score`` appended and the intermediate
        ``tmp_row_norm`` and ``tmp_dot_product`` columns removed.
    """
    # Capture feature columns before any temp columns are added so that the norm
    # and dot product are computed over the original features only.
    feature_cols = lf.select(FEATURE_SELECTOR).collect_schema().names()
    # Drop any feature columns with nulls from the calculation; they remain in output.
    null_counts = lf.select(feature_cols).null_count().collect().row(0)
    feature_cols = [c for c, n in zip(feature_cols, null_counts) if n == 0]
    control_median_lf = (
        lf.filter(pl.col(CONTROL_COLUMN_NAME)).select(feature_cols).median()
    )
    control_norm = (
        control_median_lf
        .select(pl.sum_horizontal([pl.col(c).pow(2) for c in feature_cols]).sqrt())
        .collect()
        .item()
    )

    row_norm_expr = pl.sum_horizontal([pl.col(c).pow(2) for c in feature_cols]).sqrt()
    dot_expr = pl.sum_horizontal(
        pl.col(c) * pl.col(f"{c}_ctrl") for c in feature_cols
    )

    return (
        lf.join(control_median_lf, how="cross", suffix="_ctrl")
        .with_columns(row_norm_expr.alias(NORM_COL), dot_expr.alias(DOT_COL))
        .drop([f"{c}_ctrl" for c in feature_cols])
        .with_columns(
            ((1 - pl.col(DOT_COL) / (pl.col(NORM_COL) * control_norm)) / 2).alias(
                IMPACT_SCORE_COL
            )
        )
        .select(pl.exclude(DOT_COL, NORM_COL))
    )
