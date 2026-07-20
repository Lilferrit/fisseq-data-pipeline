"""Variant selection and optional downsampled-pseudo-variant generation from a YAML spec.

Hydra entry point ``fisseq-input`` / Nextflow process ``INPUT`` (optional upstream
stage, gated by ``params.config_dir``). Reads a hand-authored YAML config
(``config_path``, parsed separately from the Hydra CLI config) describing one or
more input cell-score files (CSV or Parquet), classifies each row's variant,
restricts to a fixed set of variant classes plus the most common missense
variants, and optionally augments the result with reproducibly-downsampled
"pseudo variant" rows for QC/calibration purposes. Writes one ``input/``-ready
cell-level Parquet file.

Pipeline
--------
1. Load every file in ``input_paths`` (CSV or parquet, detected by
   extension), tagging each row with which file it came from and its row
   index within that file (``origin_file`` / ``origin_row_idx`` — these end
   up as ``meta_origin_file`` / ``meta_origin_row_idx`` in the output, via
   the same auto-prefixing step that handles the rest of the metadata
   columns). Concatenate everything into one lazy frame.
2. Classify each row's variant (see :func:`classify_variants`). Classification
   is based on the part of ``aaChanges`` before any ``:`` — a metadata tag
   component — so already-tagged variants (including the pseudo variants
   this script itself creates) classify the same as their untagged base.
3. Find the ``top_n_missense`` (default 50) Single Missense variants by cell
   count, then filter to keep only: Synonymous, WT, Frameshift, and those
   selected missense variants.
4. If ``downsample_fraction`` is set (disabled by default — see Config file
   below), create reproducibly downsampled "pseudo variants" for the kept
   Synonymous and Single Missense rows: for each variant, keep a
   deterministic, seeded ~``downsample_fraction`` of its rows, and tag the
   copy's ``aaChanges`` with ``:downsampled-half`` (e.g. ``V123A`` ->
   ``V123A:downsampled-half``). These are added alongside the originals, not
   in place of them. When ``downsample_fraction`` is unset/null, this step
   is skipped entirely — no pseudo-variant rows, no hashing/ranking work.
5. Select/rename columns (feature columns vs. auto-prefixed ``meta_*``
   columns, same convention as before) and write the result to a single
   output Parquet file.

Config file
-----------
``config_path`` points to a YAML file with:

    input_paths: [/path/to/file1.parquet, /path/to/file2.csv]
    top_n_missense: 50                # optional, default 50
    downsample_fraction: null         # optional, default null (disabled)
    downsample_seed: 0                # optional, default 0

Set ``downsample_fraction`` to a value in (0, 1] (e.g. ``0.5``) to enable
downsampled pseudo-variant generation.

Usage
-----
    uv run fisseq-input \\
        output_dir=./out \\
        config_path=/path/to/config.yaml
"""

import dataclasses
import logging
import pathlib

import hydra
import polars as pl
import polars.selectors as cs
import yaml
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .utils.constants import FEATURE_SELECTOR
from .utils.log import setup_logging
from .utils.variant import classify_variant

logger = logging.getLogger(__name__)

barcode_col = cs.by_name("upBarcode")
aa_changes_col = cs.by_name("aaChanges")
edit_distance_col = cs.by_name("editDistance")

DOWNSAMPLE_TAG = "downsampled-half"
DOWNSAMPLE_CLASSES = ("Synonymous", "Single Missense")
DEFAULT_TOP_N_MISSENSE = 50
DEFAULT_DOWNSAMPLE_FRACTION = None
DEFAULT_DOWNSAMPLE_SEED = 0


@dataclasses.dataclass
class InputStageConfig(AppConfig):
    """
    Hydra structured configuration for the input-generation entry point.

    Attributes
    ----------
    config_path : str
        Path to the YAML config describing the input files and (optional)
        downsampling behavior (see the module docstring's "Config file"
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


def filter_variants(lf: pl.LazyFrame, top_missense: list[str]) -> pl.LazyFrame:
    """Keep Synonymous, WT, Frameshift, and the selected Single Missense variants."""
    keep_classes = ["Synonymous", "WT", "Frameshift"]
    logger.info(
        "Filtering to classes %s plus %d selected Single Missense variant(s)",
        keep_classes,
        len(top_missense),
    )
    return lf.filter(
        pl.col("variant_class").is_in(keep_classes)
        | pl.col("variant_base").is_in(top_missense)
    )


def add_downsampled_pseudo_variants(
    lf: pl.LazyFrame,
    downsample_classes: tuple[str, ...],
    downsample_fraction: float,
    seed: int,
) -> pl.LazyFrame:
    """
    For rows whose `variant_class` is in `downsample_classes`, reproducibly
    keep ~`downsample_fraction` of each variant's rows and re-tag them as a
    "pseudo variant": `aaChanges` becomes `<aaChanges>:downsampled-half`.

    Selection is deterministic given `seed`: each row gets a seeded hash of
    its (origin_file, origin_row_idx) identity, rows are ranked by that hash
    within their variant group, and the lowest floor(fraction * group_size)
    ranks are kept. This avoids collecting the full eligible subset up front
    and reproduces exactly across runs for a fixed seed.
    """
    logger.info(
        "Building downsampled pseudo variants for classes %s (fraction=%.3f, seed=%d)",
        downsample_classes,
        downsample_fraction,
        seed,
    )
    eligible = lf.filter(pl.col("variant_class").is_in(downsample_classes))

    ranked = eligible.with_columns(
        pl.struct(["origin_file", "origin_row_idx"]).hash(seed=seed).alias("_rand"),
    ).with_columns(
        pl.col("_rand").rank(method="ordinal").over("variant_base").alias("_rank"),
        pl.len().over("variant_base").alias("_group_size"),
    )

    pseudo = (
        ranked.filter(
            pl.col("_rank") <= (pl.col("_group_size") * downsample_fraction).floor()
        )
        .drop(["_rand", "_rank", "_group_size"])
        .with_columns((pl.col("aaChanges") + f":{DOWNSAMPLE_TAG}").alias("aaChanges"))
    )
    return pseudo


def select_output_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Split columns into numeric feature columns vs. everything else. Barcode,
    edit distance, and aaChanges are kept unprefixed; every other non-feature
    column (including variant_base/variant_class/origin_file/origin_row_idx)
    is auto-prefixed with 'meta_'.
    """
    return lf.select(
        FEATURE_SELECTOR - (barcode_col | edit_distance_col | aa_changes_col),
        barcode_col,
        edit_distance_col,
        aa_changes_col,
        cs.exclude(FEATURE_SELECTOR).name.prefix("meta_"),
    )


@hydra.main(version_base=None, config_path=None, config_name="input_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: variant selection + optional downsampled pseudo-variant generation.

    Steps
    -----
    1. Read ``config_path`` (a separate, hand-authored YAML file — see the
       module docstring's "Config file" section).
    2. Load and concatenate all ``input_paths`` via :func:`load_and_concat`.
    3. Classify variants and filter to the fixed classes plus the top
       missense variants (:func:`classify_variants`, :func:`select_top_missense`,
       :func:`filter_variants`).
    4. If ``downsample_fraction`` is set, add downsampled pseudo-variant rows
       via :func:`add_downsampled_pseudo_variants`; otherwise skip this step
       entirely.
    5. Select/rename output columns and write a single Parquet file.

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
    downsample_fraction = config.get("downsample_fraction", DEFAULT_DOWNSAMPLE_FRACTION)
    downsample_seed = config.get("downsample_seed", DEFAULT_DOWNSAMPLE_SEED)

    data_lf = load_and_concat(input_paths)
    data_lf = classify_variants(data_lf)
    top_missense = select_top_missense(data_lf, top_n_missense)
    filtered_lf = filter_variants(data_lf, top_missense)

    if downsample_fraction is not None:
        pseudo_lf = add_downsampled_pseudo_variants(
            filtered_lf,
            downsample_classes=DOWNSAMPLE_CLASSES,
            downsample_fraction=downsample_fraction,
            seed=downsample_seed,
        )
        logger.info("Concatenating original and pseudo-variant rows")
        combined_lf = pl.concat([filtered_lf, pseudo_lf], how="vertical_relaxed")
    else:
        logger.info("downsample_fraction not set; skipping pseudo-variant generation")
        combined_lf = filtered_lf

    logger.info("Selecting output columns")
    combined_lf = select_output_columns(combined_lf)

    prefix = f"{in_cfg.output_root}." if in_cfg.output_root is not None else ""
    output_path = output_dir / f"{prefix}output.parquet"
    logger.info("Writing data to %s", output_path)
    combined_lf.sink_parquet(output_path)

    logger.info("Done")


if __name__ == "__main__":
    main()
