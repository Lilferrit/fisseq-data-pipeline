"""Variant selection from a YAML spec.

Hydra entry point ``fisseq-input`` / Nextflow process ``INPUT`` (optional upstream
stage, gated by ``params.config_dir``). Reads a hand-authored YAML config
(``config_path``, parsed separately from the Hydra CLI config) describing one or
more input cell-score files (CSV or Parquet), classifies each row's variant,
and restricts to a fixed set of variant classes plus the most common missense
variants. Writes one ``input/``-ready cell-level Parquet file.

Pipeline
--------
1. Load every file in ``input_paths`` (CSV or parquet, detected by
   extension), tagging each row with which file it came from and its row
   index within that file (``origin_file`` / ``origin_row_idx`` — these end
   up as ``meta_origin_file`` / ``meta_origin_row_idx`` in the output, via
   the same auto-prefixing step that handles the rest of the metadata
   columns). Concatenate everything into one lazy frame. If ``convert_first``
   is set and ``top_n_missense`` is set (i.e. an extra pass over the data is
   going to happen below), this concatenation is instead performed once up
   front and streamed to a single merged Parquet file in ``temp_dir`` (see
   :func:`convert_and_merge_inputs`), which is then read back and used for
   every step from here on — a pure IO/perf optimization that does not
   change the output.
2. Classify each row's variant (see :func:`classify_variants`). Classification
   is based on the part of ``aaChanges`` before any ``:`` — a metadata tag
   component — so already-tagged variants classify the same as their
   untagged base.
3. If ``top_n_missense`` is set, find the ``top_n_missense`` Single Missense
   variants by cell count and filter to keep only: Synonymous, WT,
   Frameshift, and those selected missense variants. When ``top_n_missense``
   is unset/null (the default), this ranking/restriction is skipped
   entirely and all Single Missense variants are kept.
4. Select/rename columns (feature columns vs. auto-prefixed ``meta_*``
   columns, same convention as before) and write the result to a single
   output Parquet file.

Config file
-----------
``config_path`` points to a YAML file with:

    input_paths: [/path/to/file1.parquet, /path/to/file2.csv]
    top_n_missense: null              # optional, default null (keep all Single Missense variants)
    feature_allowlist_file: null      # optional, default null (no allowlist)
    feature_blocklist_file: null      # optional, default null (no blocklist)
    convert_first: false              # optional, default false (see below)
    temp_dir: null                    # optional, default $TMPDIR or the system temp dir

Set ``convert_first: true`` to merge all ``input_paths`` into a single
Parquet file up front (written to ``temp_dir``, deleted once the run
finishes) before variant classification, instead of re-scanning/re-
concatenating the original files on every downstream pass. This only
happens when ``top_n_missense`` is also set — that's what causes the extra
pass it optimizes for; otherwise the pipeline is already single-pass and
this step is skipped even if ``convert_first`` is true. ``temp_dir``
defaults to ``$TMPDIR`` if set, otherwise the system temp directory (see
:func:`tempfile.gettempdir`); it is never read/used when ``convert_first``
is false.

``feature_allowlist_file`` / ``feature_blocklist_file`` each point to a plain
text file with one fnmatch-style glob pattern per line (e.g.
``Cells_AreaShape_*``), matched against feature column names. If an allowlist
is given, only feature columns matching at least one of its patterns are
kept; if a blocklist is also given, matching columns are then dropped from
what remains (allowlist is applied first).

Usage
-----
    uv run fisseq-input \\
        output_dir=./out \\
        config_path=/path/to/config.yaml
"""

import dataclasses
import fnmatch
import logging
import os
import pathlib
import tempfile

import hydra
import polars as pl
import polars.selectors as cs
import yaml
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .utils.log import setup_logging
from .utils.variant import classify_variant

logger = logging.getLogger(__name__)

barcode_col = cs.by_name("upBarcode")
aa_changes_col = cs.by_name("aaChanges")
edit_distance_col = cs.by_name("editDistance")

DEFAULT_TOP_N_MISSENSE = None
DEFAULT_CONVERT_FIRST = False
DEFAULT_TEMP_DIR = None

CONVERTED_INPUT_FILENAME = "converted_input.parquet"

IDENTITY_COLUMNS = {"upBarcode", "editDistance", "aaChanges"}
KNOWN_METADATA_COLUMNS = {
    "origin_file",
    "origin_row_idx",
    "variant_base",
    "variant_class",
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


def load_and_tag(path: str) -> pl.LazyFrame:
    """Scan a CSV or parquet file and tag each row with its file and row index."""
    suffix = pathlib.Path(path).suffix.lower()
    if suffix == ".csv":
        lf = pl.scan_csv(path)
    elif suffix == ".parquet":
        lf = pl.scan_parquet(path)
    else:
        raise ValueError(f"Unsupported input file extension '{suffix}' for {path}")

    return lf.with_row_index(name="origin_row_idx").with_columns(
        pl.lit(str(path)).alias("origin_file")
    )


def load_and_concat(paths: list[str]) -> pl.LazyFrame:
    """Load and tag every input file, then concatenate them into one lazy frame."""
    logger.info("Loading and tagging %d input file(s)", len(paths))
    for p in paths:
        logger.info("  - %s", p)
    lfs = [load_and_tag(p) for p in paths]
    # "vertical_relaxed" tolerates minor dtype mismatches across CSV/parquet
    # sources (e.g. int32 vs int64) by upcasting, rather than erroring.
    return pl.concat(lfs, how="vertical_relaxed")


def convert_and_merge_inputs(input_paths: list[str], temp_dir: str) -> pathlib.Path:
    """Merge all `input_paths` into a single combined Parquet file in `temp_dir`."""
    converted_path = pathlib.Path(temp_dir) / CONVERTED_INPUT_FILENAME
    logger.info(
        "Converting and merging %d input file(s) into %s",
        len(input_paths),
        converted_path,
    )
    load_and_concat(input_paths).sink_parquet(converted_path)
    logger.info("Merge complete")
    return converted_path


def classify_variants(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add `variant_base` (aaChanges with any ":"-delimited tag stripped) and
    `variant_class` (from `classify_variant`) columns.
    """
    logger.info("Classifying variants")
    return lf.with_columns(
        pl.col("aaChanges").str.split(":").list.first().alias("variant_base")
    ).with_columns(
        pl.col("variant_base")
        .map_elements(classify_variant, return_dtype=pl.String)
        .alias("variant_class")
    )


def select_top_missense(lf: pl.LazyFrame, top_n_missense: int) -> list[str]:
    """Return the `top_n_missense` Single Missense variants, ranked by cell count."""
    logger.info("Counting cells per Single Missense variant")
    counts = (
        lf.filter(pl.col("variant_class") == "Single Missense")
        .group_by("variant_base")
        .agg(pl.len().alias("n_cells"))
        .collect()
    )
    # sort on (n_cells desc, variant_base asc) so the top-N cutoff is
    # deterministic even when counts tie
    top = (
        counts.sort(["n_cells", "variant_base"], descending=[True, False])
        .head(top_n_missense)
        .get_column("variant_base")
        .to_list()
    )
    logger.info(
        "Selected %d Single Missense variant(s) out of %d observed",
        len(top),
        counts.shape[0],
    )
    return top


def filter_variants(lf: pl.LazyFrame, top_missense: list[str] | None) -> pl.LazyFrame:
    """
    Keep Synonymous, WT, Frameshift, and Single Missense variants.

    If `top_missense` is a list, only Single Missense variants in that list
    are kept. If `top_missense` is `None`, all Single Missense variants are
    kept (no top-N restriction).
    """
    keep_classes = ["Synonymous", "WT", "Frameshift"]
    if top_missense is None:
        logger.info(
            "top_n_missense not set; keeping classes %s plus all Single Missense variants",
            keep_classes,
        )
        return lf.filter(
            pl.col("variant_class").is_in(keep_classes + ["Single Missense"])
        )

    logger.info(
        "Filtering to classes %s plus %d selected Single Missense variant(s)",
        keep_classes,
        len(top_missense),
    )
    return lf.filter(
        pl.col("variant_class").is_in(keep_classes)
        | pl.col("variant_base").is_in(top_missense)
    )


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
    column (including variant_base/variant_class/origin_file/origin_row_idx)
    is auto-prefixed with 'meta_'.

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
    Hydra entry point: variant selection from a YAML spec.

    Steps
    -----
    1. Read ``config_path`` (a separate, hand-authored YAML file — see the
       module docstring's "Config file" section).
    2. Load and concatenate all ``input_paths`` via :func:`load_and_concat`.
       If ``convert_first`` is set and ``top_n_missense`` is also set, this
       is instead done once via :func:`convert_and_merge_inputs`, and the
       merged file is used for every step below (the merged temp file is
       removed once the run finishes, even on failure).
    3. Classify variants and filter to the fixed classes plus the top
       missense variants (:func:`classify_variants`, :func:`select_top_missense`,
       :func:`filter_variants`).
    4. Select/rename output columns and write a single Parquet file.

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
    top_n_missense = config.get("top_n_missense", DEFAULT_TOP_N_MISSENSE)
    convert_first = config.get("convert_first", DEFAULT_CONVERT_FIRST)

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

    # Merging up front avoids re-scanning/re-concatenating input_paths on the
    # extra pass that top_n_missense triggers (top-missense counts). Skipped
    # when it's not set, since the pipeline is then already a single pass
    # over the data.
    needs_extra_passes = top_n_missense is not None
    converted_path = None
    if convert_first and needs_extra_passes:
        temp_dir = config.get("temp_dir", DEFAULT_TEMP_DIR)
        if temp_dir is None:
            temp_dir = os.environ.get("TMPDIR") or tempfile.gettempdir()
        converted_path = convert_and_merge_inputs(input_paths, temp_dir)
        data_lf = pl.scan_parquet(converted_path)
    else:
        data_lf = load_and_concat(input_paths)

    try:
        data_lf = classify_variants(data_lf)
        if top_n_missense is not None:
            top_missense = select_top_missense(data_lf, top_n_missense)
        else:
            logger.info(
                "top_n_missense not set; skipping Single Missense variant ranking"
            )
            top_missense = None
        filtered_lf = filter_variants(data_lf, top_missense)

        logger.info("Selecting output columns")
        combined_lf = select_output_columns(
            filtered_lf, feature_allowlist, feature_blocklist
        )

        prefix = f"{in_cfg.output_root}." if in_cfg.output_root is not None else ""
        output_path = output_dir / f"{prefix}output.parquet"
        logger.info("Writing data to %s", output_path)
        combined_lf.sink_parquet(output_path)
    finally:
        if converted_path is not None:
            converted_path.unlink(missing_ok=True)

    logger.info("Done")


if __name__ == "__main__":
    main()
