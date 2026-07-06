import glob as _glob
import logging
import pathlib
import re
from typing import Any

import polars as pl

from .config import AppConfig
from .constants import (
    CONTROL_COLUMN_NAME,
    FEATURE_SELECTOR,
    IMPACT_SCORE_COL,
    META_BARCODE_COL,
    META_BATCH_COL,
    META_SELECTOR,
)

NORM_COL = "tmp_row_norm"
DOT_COL = "tmp_dot_product"
COSINE_DIST_COL = "tmp_cosine_distance"


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


def load_batches(
    glob_pattern: str, *, use_parent_name: bool = False
) -> tuple[pl.LazyFrame, str]:
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
    use_parent_name : bool
        If ``True``, use each file's parent directory name as the batch label
        instead of the file stem. Useful when multiple files share the same name
        but live in different subdirectories (e.g. ``batch1/filtered_cells.parquet``
        and ``batch2/filtered_cells.parquet``).

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

    def _label(p: str) -> str:
        return pathlib.Path(p).parent.name if use_parent_name else pathlib.Path(p).stem

    lf = pl.concat(
        [
            pl.scan_parquet(p).with_columns(pl.lit(_label(p)).alias(META_BATCH_COL))
            for p in paths
        ]
    )
    return lf, output_stem


def get_aggregate_meta_data(lf: pl.LazyFrame, label_col: str) -> pl.LazyFrame:
    """
    Compute per-variant metadata statistics from a LazyFrame.

    Always produces ``meta_num_cells`` (row count per label group). For each
    of ``meta_barcode`` and ``meta_batch``, produces two additional columns:
    ``{col}_num_unique`` (distinct value count per group) and ``{col}_counts``
    (per-value frequencies as a list of structs ``{col: str, count: u32}``).
    A warning is logged for any of these columns that is absent from the input;
    the column pair is silently omitted from the result.

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
        One row per label group. Always contains ``label_col`` and
        ``meta_num_cells``. Additionally contains ``{col}_num_unique`` and
        ``{col}_counts`` for each of ``meta_barcode`` and ``meta_batch`` that
        is present in the input.
    """
    label_lgb = lf.select(META_SELECTOR).group_by(label_col)
    agg_exprs = [pl.col(label_col).count().alias("meta_num_cells")]
    cols = set(lf.collect_schema().names())

    for col in [META_BARCODE_COL, META_BATCH_COL]:
        if col not in cols:
            logging.warning(
                "Metadata column %s not present in input dataframe - "
                "skipping meta data aggregation over %s",
                col,
                col,
            )
            continue

        agg_exprs.extend(
            [
                pl.col(col).n_unique().alias(f"{col}_num_unique"),
                pl.col(col).value_counts().alias(f"{col}_counts"),
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


def compute_cosine_distance(
    lf: pl.LazyFrame, feature_cols: list[str], suffix: str
) -> pl.LazyFrame:
    """
    Compute cosine distance between each row's feature vector and a second
    feature vector already present in the same row under a suffixed name.

    Intended for use after a join (cross-join against a single reference row,
    or a self-join to form sample pairs) has placed a second copy of
    ``feature_cols`` in each row, named ``f"{col}{suffix}"``. Computes

        cosine_distance = 1 - cos_similarity

    Zero-norm vectors are treated as unit vectors so the division does not
    produce ``NaN``.

    Parameters
    ----------
    lf : pl.LazyFrame
        Frame containing both ``feature_cols`` and their ``{suffix}``-suffixed
        counterparts.
    feature_cols : list[str]
        Names of the (unsuffixed) feature columns to compare.
    suffix : str
        Suffix identifying the second feature vector's columns.

    Returns
    -------
    pl.LazyFrame
        Input frame with ``tmp_cosine_distance`` appended. The suffixed input
        columns are not dropped; the caller is responsible for that.
    """
    norm_a = pl.sum_horizontal([pl.col(c).pow(2) for c in feature_cols]).sqrt()
    norm_b = pl.sum_horizontal(
        [pl.col(f"{c}{suffix}").pow(2) for c in feature_cols]
    ).sqrt()
    dot = pl.sum_horizontal(pl.col(c) * pl.col(f"{c}{suffix}") for c in feature_cols)

    safe_norm_a = pl.when(norm_a == 0.0).then(1.0).otherwise(norm_a)
    safe_norm_b = pl.when(norm_b == 0.0).then(1.0).otherwise(norm_b)

    return lf.with_columns(
        (1 - dot / (safe_norm_a * safe_norm_b)).alias(COSINE_DIST_COL)
    )


def compute_impact_score(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute a cosine-distance-based impact score relative to the control median.

    The impact score measures how far each row's feature vector is from the
    median feature vector of control rows (``meta_is_control == True``):

        impact = cosine_distance / 2

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
        Input frame with ``meta_impact_score`` appended and intermediate
        columns removed.
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

    joined_lf = lf.join(control_median_lf, how="cross", suffix="_ctrl")
    dist_lf = compute_cosine_distance(joined_lf, feature_cols, suffix="_ctrl")

    return (
        dist_lf.drop([f"{c}_ctrl" for c in feature_cols])
        .with_columns((pl.col(COSINE_DIST_COL) / 2).alias(IMPACT_SCORE_COL))
        .drop(COSINE_DIST_COL)
    )
