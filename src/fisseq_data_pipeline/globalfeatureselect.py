"""Global (cross-batch, per group) feature-selection stage.

Hydra entry point backing the Nextflow process ``GLOBAL_FEATURE_SELECT``. Runs
once per active ``global_group``, reusing the already-computed BATCHWISE
feature-selection artifacts (:mod:`.aggregatefeaturetype`,
:mod:`.combineblocklists`) for that group's member batches instead of
recomputing anything from raw cells:

1. Combine each member batch's own combined blocklist
   (``feature_select_batchwise/<batch>/blocklist.parquet``) using an
   agreement threshold (see :func:`combine_batch_blocklists`).
2. For each member batch, join its per-feature-type aggregate files
   (``feature_select_batchwise/<batch>/aggregates/<feature_type>.parquet``)
   and normalize the joined table to its own synonymous baseline (see
   :func:`normalize_batch_aggregate`) — this is both the batch-correction
   step and the normalization step.
3. Concatenate every member batch's normalized table and take the
   per-feature median, grouped by ``label_column`` (see
   :func:`median_across_batches`), since the same variant can appear in more
   than one batch.
4. Drop columns blocked by step 1's blocklist and run
   :func:`fisseq_data_pipeline.featureselect.pyc_feature_select` (see
   :func:`select_global_aggregate`).

Writes exactly two outputs: the selected aggregate table and the combined
blocklist.
"""

import dataclasses
import glob
import logging
import pathlib
from typing import List, Optional

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .aggregate import variant_classification
from .config import AppConfig
from .featureselect import pyc_feature_select
from .normalize import Normalizer
from .utils.constants import FEATURE_SELECTOR
from .utils.featuretypes import join_feature_type_files
from .utils.log import setup_logging

_cs = ConfigStore.instance()


def normalize_batch_aggregate(
    pipeline_dir: str, batch_stem: str, label_column: str
) -> pl.LazyFrame:
    """
    Join one batch's per-feature-type aggregates and normalize to its own
    synonymous baseline.

    Parameters
    ----------
    pipeline_dir : str
        Absolute path to the pipeline's root output directory.
    batch_stem : str
        The batch's identifier (matches ``feature_select_batchwise/<batch_stem>``).
    label_column : str
        Name of the column identifying variant labels.

    Returns
    -------
    pl.LazyFrame
        The batch's combined per-variant aggregate table, z-score normalized
        to its own synonymous (control) rows.

    Raises
    ------
    ValueError
        If no per-feature-type aggregate files are found for this batch.
    """
    glob_pattern = (
        f"{pipeline_dir}/feature_select_batchwise/{batch_stem}/aggregates/*.parquet"
    )
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        raise ValueError(
            f"No per-feature-type aggregate files matched glob pattern: {glob_pattern!r}"
        )
    agg_df = join_feature_type_files(paths, label_column)
    lf = variant_classification(agg_df.lazy(), label_column)
    normalizer = Normalizer.from_lazyframe(lf, fit_only_on_control=True)
    return normalizer.apply(lf)


def median_across_batches(
    batch_lfs: List[pl.LazyFrame], label_column: str
) -> pl.DataFrame:
    """
    Concatenate every member batch's normalized aggregate table and take the
    per-feature median, grouped by ``label_column``.

    A variant appearing in multiple batches is collapsed to a single row;
    a variant appearing in only one batch passes through unchanged (the
    median of one value is that value).

    Parameters
    ----------
    batch_lfs : list[pl.LazyFrame]
        Each member batch's normalized aggregate table (see
        :func:`normalize_batch_aggregate`).
    label_column : str
        Name of the column identifying variant labels, used as the group key.

    Returns
    -------
    pl.DataFrame
        One row per variant, with every feature column set to its
        cross-batch median.
    """
    lf = pl.concat(batch_lfs)
    feature_cols = lf.select(FEATURE_SELECTOR).columns
    return (
        lf.group_by(label_column)
        .agg([pl.col(c).median() for c in feature_cols])
        .collect()
    )


def combine_batch_blocklists(
    paths: List[str], min_batches_ok: Optional[int]
) -> pl.DataFrame:
    """
    Combine member batches' blocklists into one global blocklist, using an
    agreement threshold across batches.

    For each feature, ``n_batches`` counts how many of the given batch
    blocklists report on it and ``n_ok`` counts how many mark it
    ``feature_ok=True``. If ``min_batches_ok`` is ``None``, a feature is
    globally ok iff it is ok in every batch that reports on it
    (``n_ok == n_batches``); otherwise iff ``n_ok >= min_batches_ok``.

    Parameters
    ----------
    paths : list[str]
        Paths to member batches' combined blocklist parquet files (each with
        ``feature``/``feature_ok`` columns). Must be non-empty.
    min_batches_ok : int or None
        Minimum number of batches that must mark a feature ok for it to be
        globally ok. ``None`` requires unanimity across reporting batches.

    Returns
    -------
    pl.DataFrame
        Columns ``feature``, ``n_batches``, ``n_ok``, ``feature_ok``.

    Raises
    ------
    ValueError
        If ``paths`` is empty.
    """
    if not paths:
        raise ValueError("No blocklist files provided")
    combined = pl.concat(
        [pl.read_parquet(p).select("feature", "feature_ok") for p in paths]
    )
    result = combined.group_by("feature").agg(
        pl.col("feature").count().alias("n_batches"),
        pl.col("feature_ok").sum().alias("n_ok"),
    )
    if min_batches_ok is None:
        ok_expr = pl.col("n_ok") == pl.col("n_batches")
    else:
        ok_expr = pl.col("n_ok") >= min_batches_ok
    return result.with_columns(ok_expr.alias("feature_ok"))


def select_global_aggregate(agg_df: pl.DataFrame, bl_df: pl.DataFrame) -> pl.DataFrame:
    """
    Drop globally-blocked feature columns and run pycytominer feature
    selection.

    Parameters
    ----------
    agg_df : pl.DataFrame
        The cross-batch median aggregate table (see
        :func:`median_across_batches`).
    bl_df : pl.DataFrame
        The combined global blocklist (see :func:`combine_batch_blocklists`).

    Returns
    -------
    pl.DataFrame
        ``agg_df`` with blocked columns dropped and pycytominer feature
        selection applied.
    """
    block_list = set(bl_df.filter(~pl.col("feature_ok"))["feature"].to_list())
    agg_df = agg_df.drop([c for c in block_list if c in agg_df.columns])
    return pyc_feature_select(agg_df)


@dataclasses.dataclass
class GlobalFeatureSelectConfig(AppConfig):
    """
    Hydra structured configuration for the global feature-selection entry
    point.

    Attributes
    ----------
    pipeline_dir : str
        Absolute path to the pipeline's root output directory. Required.
    batch_stems : list[str]
        The active group's member batch stems (only those with
        ``run_feature_selection`` enabled, i.e. the ones that actually have
        ``feature_select_batchwise/<batch>/...`` on disk). Required,
        non-empty.
    label_column : str
        Name of the column identifying variant labels. Defaults to
        ``meta_aa_changes``.
    min_batches_ok : int or None
        Minimum number of member batches that must mark a feature ok for it
        to be globally ok. Defaults to ``None`` (unanimity across batches
        that report on it) — see :func:`combine_batch_blocklists`.
    """

    pipeline_dir: str = MISSING
    batch_stems: List[str] = MISSING
    label_column: str = "meta_aa_changes"
    min_batches_ok: Optional[int] = None


_cs.store(name="global_feature_select_main", node=GlobalFeatureSelectConfig)


@hydra.main(
    version_base=None, config_path=None, config_name="global_feature_select_main"
)
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: global (cross-batch, per group) feature selection.

    Output files
    ------------
    - ``{output_dir}/aggregate.parquet`` — the selected, cross-batch median
      aggregate table.
    - ``{output_dir}/blocklist.parquet`` — the combined global blocklist.

    Raises
    ------
    ValueError
        If ``batch_stems`` is empty, or if any member batch is missing its
        BATCHWISE aggregate or blocklist files.
    """
    gfs_cfg: GlobalFeatureSelectConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(gfs_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gfs_cfg.output_dir = output_dir
    setup_logging(gfs_cfg, "global_feature_select")

    if not gfs_cfg.batch_stems:
        raise ValueError("batch_stems must be a non-empty list")

    logging.info(
        "Normalizing per-batch aggregates for %d batch(es)", len(gfs_cfg.batch_stems)
    )
    batch_lfs = [
        normalize_batch_aggregate(gfs_cfg.pipeline_dir, stem, gfs_cfg.label_column)
        for stem in gfs_cfg.batch_stems
    ]

    logging.info("Computing cross-batch median aggregate")
    agg_df = median_across_batches(batch_lfs, gfs_cfg.label_column)

    logging.info("Combining per-batch blocklists")
    bl_paths = [
        f"{gfs_cfg.pipeline_dir}/feature_select_batchwise/{stem}/blocklist.parquet"
        for stem in gfs_cfg.batch_stems
    ]
    bl_df = combine_batch_blocklists(bl_paths, gfs_cfg.min_batches_ok)

    logging.info("Running pycytominer feature selection")
    selected_df = select_global_aggregate(agg_df, bl_df)

    agg_path = output_dir / "aggregate.parquet"
    bl_path = output_dir / "blocklist.parquet"
    logging.info("Writing aggregate to %s", agg_path)
    selected_df.write_parquet(agg_path)
    logging.info("Writing blocklist to %s", bl_path)
    bl_df.write_parquet(bl_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
