"""Edit-distance and barcode-count QC filtering for raw FISSEQ cell data.

Hydra entry point ``fisseq-qc-filter`` / Nextflow process ``QC_FILTER`` (first
pipeline stage). Reads one or more raw CSV/Parquet cell files, renames columns to
canonical ``meta_*`` names, and applies sequential edit-distance, barcode-count,
and variant-barcode-count filters. Writes ``filtered_cells.parquet``,
``barcode_counts.parquet``, and ``variants_per_barcode.parquet``.
"""

import dataclasses
import logging
import pathlib
from os import PathLike
from typing import Any, Iterable

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .utils.constants import (
    META_BARCODE_COL,
    META_EDIT_DISTANCE_COL,
    META_VARIANT_TAG_COL,
)
from .utils.log import setup_logging


@dataclasses.dataclass
class QcFilterConfig(AppConfig):
    """
    Hydra structured configuration for the QC filter entry point.

    Attributes
    ----------
    cell_files : Any
        Path or list of paths to cell data files (CSV or Parquet).
    bc_threshold : int
        Minimum number of cells required for a barcode to pass QC.
        Defaults to ``10``.
    variant_bc_threshold : int
        Minimum number of unique barcodes required for a variant to pass QC.
        Defaults to ``4``.
    edit_distance_threshold : int
        Maximum edit distance allowed for a cell to pass QC. Defaults to ``1``.
    barcode_col_name : str
        Name of the barcode column in the input data. Defaults to
        ``"upBarcode"``.
    aa_changes_col_name : str
        Name of the amino-acid changes column in the input data. Defaults to
        ``"aaChanges"``.
    edit_distance_col_name : str
        Name of the edit distance column in the input data. Defaults to
        ``"editDistance"``.
    label_column : str
        Name of the output label column after renaming. Defaults to
        ``"meta_aa_changes"``.
    """

    cell_files: Any = MISSING
    bc_threshold: int = 10
    variant_bc_threshold: int = 4
    edit_distance_threshold: int = 1
    barcode_col_name: str = "upBarcode"
    aa_changes_col_name: str = "aaChanges"
    edit_distance_col_name: str = "editDistance"
    label_column: str = "meta_aa_changes"


_cs = ConfigStore.instance()
_cs.store(name="qc_filter_main", node=QcFilterConfig)


def get_barcode_counts(lf: pl.LazyFrame, cfg: DictConfig) -> pl.LazyFrame:
    """
    Count cells per barcode and flag barcodes meeting the threshold.

    Groups by ``META_BARCODE_COL``, counts occurrences, and adds a
    ``barcode_ok`` column (non-null when count >= ``cfg.bc_threshold``).
    Retains the first ``cfg.label_column`` per barcode group.

    Parameters
    ----------
    lf : pl.LazyFrame
        Cell-level lazy frame containing ``META_BARCODE_COL`` and
        ``cfg.label_column`` (as produced by :func:`filter_columns`).
    cfg : DictConfig
        Hydra config supplying ``bc_threshold`` and ``label_column``.

    Returns
    -------
    pl.LazyFrame
        Lazy frame with one row per barcode, including ``count``,
        ``cfg.label_column``, and ``barcode_ok``.
    """
    barcode_lf = (
        lf.group_by(META_BARCODE_COL)
        .agg(
            [
                pl.len().alias("count"),
                pl.col(cfg.label_column).first(),
            ]
        )
        .with_columns(
            pl.when(pl.col("count") >= cfg.bc_threshold)
            .then(pl.col("count"))
            .otherwise(None)
            .alias("barcode_ok")
        )
    )

    return barcode_lf


def get_barcodes_per_variant(cells_lf: pl.LazyFrame, cfg: DictConfig) -> pl.LazyFrame:
    """
    Count distinct barcodes per variant and flag variants meeting threshold.

    Groups by ``cfg.label_column``, counts barcodes, and adds a
    ``variant_barcode_count_ok`` column (non-null when barcode count
    >= ``cfg.variant_bc_threshold``).

    Parameters
    ----------
    cells_lf : pl.LazyFrame
        Cell-level lazy frame containing ``META_BARCODE_COL`` and
        ``cfg.label_column`` (as produced by :func:`filter_columns`).
    cfg : DictConfig
        Hydra config supplying ``variant_bc_threshold`` and ``label_column``.

    Returns
    -------
    pl.LazyFrame
        Lazy frame with one row per variant, including ``barcode_count``
        and ``variant_barcode_count_ok``.
    """
    barcodes_per_variant_lf = (
        cells_lf.group_by(cfg.label_column)
        .agg(
            [
                pl.col(META_BARCODE_COL).n_unique().alias("barcode_count"),
            ]
        )
        .with_columns(
            pl.when(pl.col("barcode_count") >= cfg.variant_bc_threshold)
            .then(pl.col("barcode_count"))
            .otherwise(None)
            .alias("variant_barcode_count_ok")
        )
    )

    return barcodes_per_variant_lf


def add_qc_queries(
    lf: pl.LazyFrame, cfg: DictConfig
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """
    Apply edit-distance, barcode-level, and variant-level QC filters.

    Filters are applied sequentially:

    1. Retain rows with ``META_EDIT_DISTANCE_COL`` <= ``cfg.edit_distance_threshold``.
    2. Remove barcodes below ``cfg.bc_threshold`` cell count.
    3. Remove variants below ``cfg.variant_bc_threshold`` barcode count.

    Parameters
    ----------
    lf : pl.LazyFrame
        Cell-level lazy frame to filter.
    cfg : DictConfig
        Hydra config supplying QC thresholds and column names.

    Returns
    -------
    tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]
        ``(filtered_lf, barcode_count_lf, variants_per_barcode_lf)`` where the
        latter two contain the intermediate QC summary frames.
    """
    logging.info("Adding edit distance QC query")
    lf = lf.filter(pl.col(META_EDIT_DISTANCE_COL) <= cfg.edit_distance_threshold)

    logging.info("Adding Barcode Level QC query")
    barcode_count_lf = get_barcode_counts(lf, cfg)
    lf = lf.join(
        barcode_count_lf.filter(pl.col("barcode_ok").is_not_null()).select(
            META_BARCODE_COL
        ),
        on=META_BARCODE_COL,
        how="inner",
    )

    logging.info("Adding Variant Level QC query")
    variants_per_barcode_lf = get_barcodes_per_variant(lf, cfg)
    lf = lf.join(
        variants_per_barcode_lf.filter(
            pl.col("variant_barcode_count_ok").is_not_null()
        ).select(cfg.label_column),
        on=cfg.label_column,
        how="inner",
    )

    return lf, barcode_count_lf, variants_per_barcode_lf


def read_file(cell_file_path: pathlib.Path) -> pl.LazyFrame:
    """
    Read a single cell file into a lazy frame.

    Supports CSV and Parquet formats (extensions ``.csv``, ``.parquet``,
    ``.parq``, ``.pq``). Adds two metadata columns: ``meta_source_file``
    (file path as a string) and ``meta_source_file_idx`` (row index within
    the source file).

    Parameters
    ----------
    cell_file_path : pathlib.Path
        Path to the cell data file.

    Returns
    -------
    pl.LazyFrame
        Lazy frame of the file contents with metadata columns.
    """
    logging.info("Scanning file %s", cell_file_path)

    if cell_file_path.suffix == ".csv":
        logging.warning(
            "Lazy evaluation is much slower for CSV files."
            " Consider converting to Parquet for better performance."
        )
        lf = pl.scan_csv(cell_file_path)
    elif cell_file_path.suffix in [".parquet", ".parq", ".pq"]:
        lf = pl.scan_parquet(cell_file_path)

    lf = lf.with_columns(
        pl.lit(str(cell_file_path)).alias("meta_source_file"),
        pl.row_index().alias("meta_source_file_idx"),
    )

    return lf


def combine_cell_files(cell_files: Iterable[PathLike]) -> pl.LazyFrame:
    """
    Read and concatenate multiple cell files into one lazy frame.

    Parameters
    ----------
    cell_files : Iterable[PathLike]
        Iterable of paths to cell data files.

    Returns
    -------
    pl.LazyFrame
        Concatenated lazy frame of all input files.
    """
    return pl.concat([read_file(pathlib.Path(cell_file)) for cell_file in cell_files])


def filter_columns(lf: pl.LazyFrame, cfg: DictConfig) -> pl.LazyFrame:
    """
    Rename input columns to canonical meta names and retain only QC-relevant columns.

    Renames ``cfg.barcode_col_name`` → ``META_BARCODE_COL`` and
    ``cfg.edit_distance_col_name`` → ``META_EDIT_DISTANCE_COL``.
    ``cfg.aa_changes_col_name`` is split on the first ``":"`` into a base
    label (``cfg.label_column``) and an optional tag (``META_VARIANT_TAG_COL``,
    ``null`` when no tag is present) — this is the single point in the
    pipeline where a raw, possibly-tagged variant label (e.g.
    ``"M1K:downsampled-half"``) resolves into the canonical
    ``meta_aa_changes`` value every downstream stage consumes. Then drops all
    columns except meta columns (``meta_`` prefix) and CellProfiler feature
    columns (starting with an uppercase letter and containing ``_``).

    Parameters
    ----------
    lf : pl.LazyFrame
        Lazy frame containing all raw input columns.
    cfg : DictConfig
        Hydra config supplying column name mappings.

    Returns
    -------
    pl.LazyFrame
        Lazy frame retaining only the necessary columns with canonical names.
    """
    aa_changes_split = pl.col(cfg.aa_changes_col_name).str.splitn(":", 2)
    lf = lf.with_columns(
        aa_changes_split.struct.field("field_0").alias(cfg.label_column),
        aa_changes_split.struct.field("field_1").alias(META_VARIANT_TAG_COL),
        pl.col(cfg.edit_distance_col_name).alias(META_EDIT_DISTANCE_COL),
        pl.col(cfg.barcode_col_name).alias(META_BARCODE_COL),
    )

    schema_names = lf.collect_schema().names()
    cell_profiler_columns = [
        col for col in schema_names if len(col) > 0 and col[0].isupper() and "_" in col
    ]
    meta_columns = [col for col in schema_names if col.startswith("meta_")]

    return lf.select(pl.col(meta_columns + cell_profiler_columns))


@hydra.main(version_base=None, config_path=None, config_name="qc_filter_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: QC filtering of cell-level data files.

    Steps
    -----
    1. Read and concatenate all ``cell_files`` via :func:`combine_cell_files`.
    2. Retain only QC-relevant columns via :func:`filter_columns`.
    3. Apply edit-distance, barcode-count, and variant-count filters via
       :func:`add_qc_queries`.
    4. Write three output Parquet files to ``output_dir``.

    Output files
    ------------
    - ``{prefix}filtered_cells.parquet``
    - ``{prefix}barcode_counts.parquet``
    - ``{prefix}variants_per_barcode.parquet``

    where ``prefix`` is ``{output_root}.`` when ``output_root`` is set,
    otherwise empty.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.qc_filter \\
            output_dir=./out \\
            'cell_files=[data/cells.parquet]' \\
            bc_threshold=10
    """
    qc_cfg: QcFilterConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(qc_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_cfg.output_dir = output_dir
    setup_logging(qc_cfg, "qc_filter")

    prefix = f"{qc_cfg.output_root}." if qc_cfg.output_root is not None else ""

    cell_files = (
        list(cfg.cell_files)
        if hasattr(cfg.cell_files, "__iter__") and not isinstance(cfg.cell_files, str)
        else [cfg.cell_files]
    )

    combined_lf = combine_cell_files(cell_files)
    combined_lf = filter_columns(combined_lf, cfg)
    combined_lf, barcode_count_lf, variants_per_barcode_lf = add_qc_queries(
        combined_lf, cfg
    )

    logging.info("Writing output files to %s", output_dir)
    for name, lf in [
        ("filtered_cells", combined_lf),
        ("barcode_counts", barcode_count_lf),
        ("variants_per_barcode", variants_per_barcode_lf),
    ]:
        logging.info("Writing %s", name)
        lf.sink_parquet(output_dir / f"{prefix}{name}.parquet")

    logging.info("Done")


if __name__ == "__main__":
    main()
