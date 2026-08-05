"""Load and merge raw input files from a YAML spec.

Hydra entry point (``python -m fisseq_data_pipeline.input``) / Nextflow process ``INPUT`` (runs once per
mandatory batch config file in ``<pipeline_dir>/configs/``). Reads a hand-authored YAML config
(``config_path``, parsed separately from the Hydra CLI config) describing one or
more input cell-score files (CSV or Parquet), and merges them into a single
``input/``-ready cell-level Parquet file.

Pipeline
--------
1. Load every file in ``input_paths`` (CSV or parquet, detected by
   extension), tagging each row with which file it came from and its row
   index within that file (``origin_file`` / ``origin_row_idx`` — these end
   up as ``meta_origin_file`` / ``meta_origin_row_idx`` in the output, via
   the same auto-prefixing step that handles the rest of the metadata
   columns). Concatenate everything into one lazy frame.
2. Select/rename columns (feature columns vs. auto-prefixed ``meta_*``
   columns, same convention as before) and write the result to a single
   output Parquet file.

Variant-class/count-based restriction (previously done here via
``top_n_missense``) now happens downstream, in
:func:`fisseq_data_pipeline.qcfilter.select_variants` (``n_variants``/
``variant_downsample_classes``/``variant_downsample_mode``), so it applies
uniformly to every batch.

Config file
-----------
``config_path`` points to a YAML file with:

    input_paths: [/path/to/file1.parquet, /path/to/file2.csv]
    feature_allowlist_file: null      # optional, default null (no allowlist)
    feature_blocklist_file: null      # optional, default null (no blocklist)
    csv_schema_scan_rows: 100         # optional, default 100

``feature_allowlist_file`` / ``feature_blocklist_file`` each point to a plain
text file with one fnmatch-style glob pattern per line (e.g.
``Cells_AreaShape_*``), matched against feature column names. If an allowlist
is given, only feature columns matching at least one of its patterns are
kept; if a blocklist is also given, matching columns are then dropped from
what remains (allowlist is applied first).

``csv_schema_scan_rows`` controls how many rows of each CSV ``input_paths``
source are scanned to infer column dtypes (forwarded to polars
``scan_csv``'s ``infer_schema_length``). ``null`` scans every row instead --
slower, but avoids mis-inferred dtypes on columns whose non-null/non-integer
values only appear after the scanned prefix. Has no effect on parquet
sources.

Usage
-----
    uv run python -m fisseq_data_pipeline.input \\
        output_dir=./out \\
        config_path=/path/to/config.yaml
"""

import dataclasses
import fnmatch
import logging
import pathlib

import hydra
import polars as pl
import polars.selectors as cs
import yaml
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .utils.log import setup_logging

logger = logging.getLogger(__name__)

barcode_col = cs.by_name("upBarcode")
aa_changes_col = cs.by_name("aaChanges")
edit_distance_col = cs.by_name("editDistance")

IDENTITY_COLUMNS = {"upBarcode", "editDistance", "aaChanges"}
KNOWN_METADATA_COLUMNS = {
    "origin_file",
    "origin_row_idx",
}


@dataclasses.dataclass
class InputStageConfig(AppConfig):
    """
    Hydra structured configuration for the input-generation entry point.

    Attributes
    ----------
    config_path : str
        Path to the YAML config describing the input files and variant
        selection behavior (see the module docstring's "Config file"
        section). Parsed separately via ``yaml.safe_load`` — intentionally
        not flattened into individual Hydra CLI fields, since
        ``input_paths`` is a list of arbitrary length. Required.
    """

    config_path: str = MISSING


_cs = ConfigStore.instance()
_cs.store(name="input_main", node=InputStageConfig)


def load_and_tag(path: str, infer_schema_length: int | None = 100) -> pl.LazyFrame:
    """
    Scan a CSV or parquet file and tag each row with its file and row index.

    Parameters
    ----------
    path : str
        Path to the CSV or parquet file to scan.
    infer_schema_length : int or None
        Number of rows to sample when inferring CSV column dtypes (forwarded
        to :func:`polars.scan_csv`'s ``infer_schema_length``). ``None`` scans
        every row. Ignored for parquet files, which carry their own schema.
    """
    suffix = pathlib.Path(path).suffix.lower()
    if suffix == ".csv":
        lf = pl.scan_csv(path, infer_schema_length=infer_schema_length)
    elif suffix == ".parquet":
        lf = pl.scan_parquet(path)
    else:
        raise ValueError(f"Unsupported input file extension '{suffix}' for {path}")

    # Workaround: unsigned integer columns have been observed to hide negative
    # values (silently reinterpreted), so force them to signed integers.
    lf = lf.with_columns(cs.unsigned_integer().cast(pl.Int64))

    return lf.with_row_index(name="origin_row_idx").with_columns(
        pl.lit(str(path)).alias("origin_file")
    )


def load_and_concat(
    paths: list[str], infer_schema_length: int | None = 100
) -> pl.LazyFrame:
    """
    Load and tag every input file, then concatenate them into one lazy frame.

    ``infer_schema_length`` is forwarded to each :func:`load_and_tag` call --
    see its docstring.
    """
    logger.info("Loading and tagging %d input file(s)", len(paths))
    for p in paths:
        logger.info("  - %s", p)
    lfs = [load_and_tag(p, infer_schema_length=infer_schema_length) for p in paths]
    # "vertical_relaxed" tolerates minor dtype mismatches across CSV/parquet
    # sources (e.g. int32 vs int64) by upcasting, rather than erroring.
    return pl.concat(lfs, how="vertical_relaxed")


def load_feature_patterns(path: str) -> list[str]:
    """Read one glob-style feature-column-name pattern per line from a text file."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def select_output_columns(
    lf: pl.LazyFrame,
    feature_allowlist: list[str] | None = None,
    feature_blocklist: list[str] | None = None,
) -> pl.LazyFrame:
    """
    Split columns into numeric feature columns vs. everything else. Barcode,
    edit distance, and aaChanges are kept unprefixed; every other non-feature
    column (including origin_file/origin_row_idx) is auto-prefixed with
    'meta_'.

    If `feature_allowlist` is given, only feature columns matching at least
    one of its fnmatch glob patterns are kept. If `feature_blocklist` is also
    given, feature columns matching any of its patterns are then dropped from
    what remains (allowlist is applied first).
    """
    schema_names = lf.collect_schema().names()

    feature_cols = [
        c
        for c in schema_names
        if c not in IDENTITY_COLUMNS
        and c not in KNOWN_METADATA_COLUMNS
        and not c.startswith("meta_")
    ]
    if feature_allowlist is not None:
        feature_cols = [c for c in feature_cols if _matches_any(c, feature_allowlist)]
    if feature_blocklist is not None:
        feature_cols = [
            c for c in feature_cols if not _matches_any(c, feature_blocklist)
        ]

    metadata_cols = [c for c in schema_names if c in KNOWN_METADATA_COLUMNS]

    return lf.select(
        cs.by_name(*feature_cols),
        barcode_col,
        edit_distance_col,
        aa_changes_col,
        cs.by_name(*metadata_cols).name.prefix("meta_"),
    )


@hydra.main(version_base=None, config_path=None, config_name="input_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: load and merge raw input files from a YAML spec.

    Steps
    -----
    1. Read ``config_path`` (a separate, hand-authored YAML file — see the
       module docstring's "Config file" section).
    2. Load and concatenate all ``input_paths`` via :func:`load_and_concat`.
    3. Select/rename output columns and write a single Parquet file.

    Output file
    -----------
    ``{prefix}output.parquet``, where ``prefix`` is ``{output_root}.`` when
    ``output_root`` is set, otherwise empty.
    """
    in_cfg: InputStageConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(in_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    in_cfg.output_dir = output_dir
    setup_logging(in_cfg, "input")

    logger.info("Reading config file %s", in_cfg.config_path)
    with open(in_cfg.config_path) as f:
        config = yaml.safe_load(f)

    input_paths = config["input_paths"]

    feature_allowlist = None
    if config.get("feature_allowlist_file"):
        logger.info(
            "Loading feature allowlist from %s", config["feature_allowlist_file"]
        )
        feature_allowlist = load_feature_patterns(config["feature_allowlist_file"])

    feature_blocklist = None
    if config.get("feature_blocklist_file"):
        logger.info(
            "Loading feature blocklist from %s", config["feature_blocklist_file"]
        )
        feature_blocklist = load_feature_patterns(config["feature_blocklist_file"])

    csv_schema_scan_rows = config.get("csv_schema_scan_rows", 100)
    data_lf = load_and_concat(input_paths, infer_schema_length=csv_schema_scan_rows)

    logger.info("Selecting output columns")
    combined_lf = select_output_columns(data_lf, feature_allowlist, feature_blocklist)

    prefix = f"{in_cfg.output_root}." if in_cfg.output_root is not None else ""
    output_path = output_dir / f"{prefix}output.parquet"
    logger.info("Writing data to %s", output_path)
    combined_lf.sink_parquet(output_path)

    logger.info("Done")


if __name__ == "__main__":
    main()
