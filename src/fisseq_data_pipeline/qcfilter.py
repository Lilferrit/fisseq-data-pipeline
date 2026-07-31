"""Edit-distance and barcode-count QC filtering for raw FISSEQ cell data.

Hydra entry point (``python -m fisseq_data_pipeline.qcfilter``) / Nextflow process ``QC_FILTER`` (first
pipeline stage). Reads one or more raw CSV/Parquet cell files, renames columns to
canonical ``meta_*`` names, and applies sequential edit-distance, barcode-count,
and variant-barcode-count filters. Writes ``filtered_cells.parquet``,
``barcode_counts.parquet``, and ``variants_per_barcode.parquet``.

If ``n_variants`` is set, variants whose classified label is in
``variant_downsample_classes`` (default ``["Single Missense"]``) are
restricted to at most ``n_variants`` distinct variants before QC
thresholding runs — either the highest-cell-count variants (``"top"``
mode) or a seeded random sample (``"random"`` mode); see
:func:`select_variants`. Disabled by default.

If ``downsample_amounts`` is set (a single float/int, or a list of them),
``filtered_cells.parquet`` is additionally augmented with reproducibly-
downsampled "pseudo variant" rows for QC/calibration purposes, built from
cells whose classified label is in ``downsample_classes`` (default
``["Synonymous", "Single Missense"]``) that already survived the three QC
filters above (not the raw pre-QC population — this is what makes the
pseudo-variants a valid calibration of the post-QC analysis population).
Each amount produces its own distinctly-tagged group (``:downsample-{amount}``).
``barcode_counts.parquet`` and ``variants_per_barcode.parquet`` are computed
before this step runs, so they never include pseudo-variant rows. Disabled
by default; see :func:`add_downsampled_pseudo_variants`.
"""

import dataclasses
import logging
import pathlib
from os import PathLike
from typing import Any, Iterable, List, Optional, Union

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
from .utils.variant import classify_variant

DOWNSAMPLE_TAG = "downsample"
DOWNSAMPLE_CLASSES = ("Synonymous", "Single Missense")
VARIANT_DOWNSAMPLE_CLASSES = ("Single Missense",)
VARIANT_DOWNSAMPLE_MODES = ("top", "random")


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
    n_variants : Optional[int]
        If set, restricts variants whose classified label is in
        ``variant_downsample_classes`` to at most this many distinct
        variants (see :func:`select_variants`); every other class passes
        through untouched. Runs before QC thresholding. Defaults to
        ``None`` (disabled).
    variant_downsample_classes : List[str]
        Classes (from :func:`fisseq_data_pipeline.utils.variant.classify_variant`)
        eligible for the ``n_variants`` restriction. Defaults to
        ``["Single Missense"]``.
    variant_downsample_mode : str
        ``"top"`` keeps the ``n_variants`` variants with the highest cell
        count (ties broken alphabetically); ``"random"`` keeps a seeded
        random sample of ``n_variants`` variants. Defaults to ``"top"``.
    downsample_amounts : Any
        A single float/int, or a list of floats/ints. Each item generates
        reproducibly downsampled "pseudo variant" rows (see
        :func:`add_downsampled_pseudo_variants`) from cells whose
        classified label is in ``downsample_classes`` and that survived QC
        filtering: a float in ``(0, 1]`` keeps that fraction per variant, an
        int keeps that many cells per variant (skipping variants with fewer
        cells than that). Typed ``Any`` (like ``cell_files``) since OmegaConf
        structured configs can't express ``Union[float, int, List[...]]``.
        Defaults to ``None`` (disabled).
    downsample_classes : List[str]
        Classes eligible for ``downsample_amounts`` pseudo-variant
        generation. Defaults to ``["Synonymous", "Single Missense"]``.
    downsample_seed : int
        Seed for deterministic selection, shared by ``downsample_amounts``
        and ``variant_downsample_mode="random"``. Defaults to ``0``.
    """

    cell_files: Any = MISSING
    bc_threshold: int = 10
    variant_bc_threshold: int = 4
    edit_distance_threshold: int = 1
    barcode_col_name: str = "upBarcode"
    aa_changes_col_name: str = "aaChanges"
    edit_distance_col_name: str = "editDistance"
    label_column: str = "meta_aa_changes"
    n_variants: Optional[int] = None
    variant_downsample_classes: List[str] = dataclasses.field(
        default_factory=lambda: list(VARIANT_DOWNSAMPLE_CLASSES)
    )
    variant_downsample_mode: str = "top"
    downsample_amounts: Any = None
    downsample_classes: List[str] = dataclasses.field(
        default_factory=lambda: list(DOWNSAMPLE_CLASSES)
    )
    downsample_seed: int = 0


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


def select_variants(
    lf: pl.LazyFrame,
    cfg: DictConfig,
    variant_downsample_classes: tuple[str, ...],
    n_variants: int,
    mode: str,
    seed: int,
) -> pl.LazyFrame:
    """
    Restrict rows whose classified ``cfg.label_column`` value is in
    `variant_downsample_classes` to at most `n_variants` distinct variants.
    Rows in any other class are left untouched (not filtered at all).

    `mode` is one of:

    - ``"top"``: keep the `n_variants` variants (within
      `variant_downsample_classes`) with the highest cell count, ties broken
      alphabetically ascending on ``cfg.label_column``. Fully deterministic.
    - ``"random"``: keep a seeded-random sample of `n_variants` distinct
      variants from the eligible pool. Deterministic given `seed`.

    Runs upstream of :func:`add_qc_queries`, so ``barcode_counts``/
    ``variants_per_barcode`` reflect the post-selection population.
    """
    if mode not in VARIANT_DOWNSAMPLE_MODES:
        raise ValueError(
            f"variant_downsample_mode must be one of {VARIANT_DOWNSAMPLE_MODES}, "
            f"got {mode!r}"
        )

    logging.info(
        "Restricting classes %s to %d variant(s) (mode=%s, seed=%d)",
        variant_downsample_classes,
        n_variants,
        mode,
        seed,
    )
    classified = lf.with_columns(
        pl.col(cfg.label_column)
        .map_elements(classify_variant, return_dtype=pl.String)
        .alias("_variant_class")
    )
    non_eligible = classified.filter(
        ~pl.col("_variant_class").is_in(variant_downsample_classes)
    )
    eligible = classified.filter(
        pl.col("_variant_class").is_in(variant_downsample_classes)
    )

    counts = eligible.group_by(cfg.label_column).agg(pl.len().alias("_n_cells"))
    if mode == "top":
        selected = (
            counts.sort(["_n_cells", cfg.label_column], descending=[True, False])
            .head(n_variants)
            .select(cfg.label_column)
        )
    else:
        selected = (
            counts.with_columns(pl.col(cfg.label_column).hash(seed=seed).alias("_rand"))
            .sort("_rand")
            .head(n_variants)
            .select(cfg.label_column)
        )

    eligible_kept = eligible.join(selected, on=cfg.label_column, how="inner")
    return pl.concat([non_eligible, eligible_kept], how="vertical_relaxed").drop(
        "_variant_class"
    )


def add_downsampled_pseudo_variants(
    lf: pl.LazyFrame,
    cfg: DictConfig,
    downsample_classes: tuple[str, ...],
    downsample_amount: Union[float, int],
    seed: int,
) -> pl.LazyFrame:
    """
    For QC-surviving rows whose classified ``cfg.label_column`` value is in
    `downsample_classes`, reproducibly keep either a target fraction or a
    target count of each variant's rows (per ``cfg.label_column`` group) and
    mark them as a "pseudo variant" by appending ``:{tag}`` directly onto
    ``cfg.label_column`` (e.g. ``"A326P"`` becomes ``"A326P:downsample-0.5"``
    for a float amount of ``0.5``, or ``"A326P:downsample-500"`` for an int
    amount of ``500``). This gives pseudo-variant rows their own distinct
    label value, so downstream label-based grouping (e.g. aggregation)
    treats them as a separate calibration group rather than pooling them
    with the real variant's rows.

    `downsample_amount` is interpreted per ``cfg.label_column`` group:

    - A ``float`` in ``(0, 1]`` keeps ``floor(downsample_amount * group_size)``
      rows from every eligible group (a group may shrink to 0 pseudo rows if
      it's small, but is never skipped outright).
    - An ``int`` >= 1 keeps exactly ``downsample_amount`` rows from every
      eligible group whose size is >= ``downsample_amount`` — groups with
      fewer rows than that are skipped entirely for this amount (no
      clamping, no partial keep).

    To generate pseudo rows for multiple amounts, call this once per amount
    and concatenate the results — see :func:`main`.

    Selection is deterministic given `seed`: each row gets a seeded hash of
    its (`meta_source_file`, `meta_source_file_idx`) identity, rows are
    ranked by that hash within their `cfg.label_column` group, and the
    lowest-ranked rows up to the target are kept.

    Raises
    ------
    ValueError
        If `downsample_amount` is a float outside ``(0, 1]`` or a
        non-positive int.
    """
    if isinstance(downsample_amount, float) and not (0 < downsample_amount <= 1):
        raise ValueError(
            f"downsample_amount float must satisfy 0 < x <= 1, got {downsample_amount}"
        )
    if isinstance(downsample_amount, int) and downsample_amount <= 0:
        raise ValueError(
            f"downsample_amount int must be positive, got {downsample_amount}"
        )

    logging.info(
        "Building downsampled pseudo variants for classes %s (amount=%s, seed=%d)",
        downsample_classes,
        downsample_amount,
        seed,
    )
    eligible = lf.filter(
        pl.col(cfg.label_column)
        .map_elements(classify_variant, return_dtype=pl.String)
        .is_in(downsample_classes)
    )

    ranked = eligible.with_columns(
        pl.struct(["meta_source_file", "meta_source_file_idx"])
        .hash(seed=seed)
        .alias("_rand"),
    ).with_columns(
        pl.col("_rand").rank(method="ordinal").over(cfg.label_column).alias("_rank"),
        pl.len().over(cfg.label_column).alias("_group_size"),
    )

    if isinstance(downsample_amount, float):
        target = (pl.col("_group_size") * downsample_amount).floor()
        group_eligible = pl.lit(True)
    else:
        target = pl.lit(downsample_amount)
        group_eligible = pl.col("_group_size") >= downsample_amount

    tag = f"{DOWNSAMPLE_TAG}-{downsample_amount}"
    pseudo = (
        ranked.filter(group_eligible & (pl.col("_rank") <= target))
        .drop(["_rand", "_rank", "_group_size"])
        .with_columns((pl.col(cfg.label_column) + ":" + tag).alias(cfg.label_column))
    )
    return pseudo


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
    3. If ``n_variants`` is set, restrict ``variant_downsample_classes`` to
       at most ``n_variants`` distinct variants via :func:`select_variants`
       (``"top"`` or ``"random"`` mode); otherwise skip this step entirely.
    4. Apply edit-distance, barcode-count, and variant-count filters via
       :func:`add_qc_queries`.
    5. If ``downsample_amounts`` is set (a single float/int, or a list of
       them), add one distinctly-tagged (``downsample-{amount}``) set of
       pseudo-variant rows per amount, built only from the QC survivors of
       step 4, using ``downsample_classes`` as the eligible category set,
       via :func:`add_downsampled_pseudo_variants`; otherwise skip this step
       entirely. ``barcode_counts``/``variants_per_barcode`` reflect step 4's
       output and never include these pseudo rows.
    6. Write three output Parquet files to ``output_dir``.

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

    if qc_cfg.n_variants is not None:
        combined_lf = select_variants(
            combined_lf,
            cfg,
            variant_downsample_classes=tuple(qc_cfg.variant_downsample_classes),
            n_variants=qc_cfg.n_variants,
            mode=qc_cfg.variant_downsample_mode,
            seed=qc_cfg.downsample_seed,
        )
    else:
        logging.info("n_variants not set; skipping variant-level selection")

    combined_lf, barcode_count_lf, variants_per_barcode_lf = add_qc_queries(
        combined_lf, cfg
    )

    if qc_cfg.downsample_amounts is not None:
        downsample_amounts = (
            qc_cfg.downsample_amounts
            if isinstance(qc_cfg.downsample_amounts, list)
            else [qc_cfg.downsample_amounts]
        )
        logging.info(
            "Adding downsampled pseudo-variant rows for amounts %s (classes=%s)",
            downsample_amounts,
            qc_cfg.downsample_classes,
        )
        pseudo_lfs = [
            add_downsampled_pseudo_variants(
                combined_lf,
                cfg,
                downsample_classes=tuple(qc_cfg.downsample_classes),
                downsample_amount=amount,
                seed=qc_cfg.downsample_seed,
            )
            for amount in downsample_amounts
        ]
        if pseudo_lfs:
            combined_lf = pl.concat([combined_lf, *pseudo_lfs], how="vertical_relaxed")
    else:
        logging.info("downsample_amounts not set; skipping pseudo-variant generation")

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
