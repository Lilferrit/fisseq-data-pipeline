"""Global (cross-batch, per channel) feature-selection stage.

Hydra entry point backing the Nextflow process ``GLOBAL_FEATURE_SELECT``. Runs
once per active ``global_channel``, reusing the already-computed BATCHWISE
feature-selection artifacts (:mod:`.aggregatefeaturetype`,
:mod:`.combineblocklists`) for that channel's member batches instead of
recomputing anything from raw cells:

1. Combine each member batch's own combined blocklist
   (``feature_select_batchwise/<batch>/blocklist.parquet``) using an
   agreement threshold (see :func:`combine_batch_blocklists`). This runs
   first, before any per-batch aggregate is touched, so the resulting
   globally-blocked feature set can be dropped from each batch up front.
2. For each member batch, join its per-feature-type aggregate files
   (``feature_select_batchwise/<batch>/aggregates/<feature_type>.parquet``,
   filtered to the currently-configured ``feature_select_types`` — a batch's
   ``aggregates/`` directory can otherwise carry stale files for feature
   types no longer configured, left behind by an earlier run's larger
   ``feature_select_types`` since ``publishDir mode: 'copy'`` never deletes
   them), drop step 1's globally-blocked columns, and normalize the joined
   table to its own synonymous baseline (see
   :func:`normalize_batch_aggregate`) — this is both the batch-correction
   step and the normalization step.
3. Concatenate every member batch's normalized table and take the
   per-feature median, grouped by ``label_column`` (see
   :func:`median_across_batches`), since the same variant can appear in more
   than one batch. Dropping the blocklist in step 2 is not sufficient on its
   own to make every batch's table concat-safe: each batch's
   ``aggregates/`` directory reflects whichever ``AGGREGATE_FEATURE_TYPE``
   Nextflow tasks actually succeeded for that batch (that process runs
   ``errorStrategy 'ignore'``, so a failed per-feature-type task silently
   drops its output file rather than aborting the run) and may also contain
   stale files from an earlier pipeline version. So batches can legitimately
   disagree on which feature columns exist at all (e.g. one has ``_MAD``
   where another has ``_mean``/``_std`` for the same feature, or is missing
   ``_AUROC``/``_KS`` entirely). :func:`median_across_batches` handles this
   defensively: it intersects each batch's feature columns down to the set
   common to all of them (logging a warning listing what got dropped per
   batch) and keeps only ``label_column`` among metadata columns, before
   concatenating.
4. Drop columns blocked by step 1's blocklist (typically a no-op by this
   point, since step 2 already dropped them per batch; kept as
   defense-in-depth for direct callers) and run
   :func:`fisseq_data_pipeline.featureselect.pyc_feature_select` (see
   :func:`select_global_aggregate`).
5. Optionally (``compute_impact_score``, default ``True``) re-derive
   ``meta_is_control`` via :func:`.aggregate.variant_classification` — lost
   in step 3's metadata collapse — and compute each variant's cosine-distance
   impact score against the control median (see
   :func:`.utils.vectors.compute_impact_score`), the same measure the
   BATCHWISE ``FINALIZE_FEATURE_SELECT`` stage computes.

Writes exactly two outputs: the selected aggregate table and the combined
blocklist.
"""

import dataclasses
import glob
import logging
import pathlib
from typing import Iterable, List, Optional

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .aggregate import variant_classification
from .config import AppConfig
from .featureselect import pyc_feature_select
from .normalize import Normalizer
from .utils.constants import FEATURE_SELECTOR
from .utils.dimreduction import compute_pca, compute_umap
from .utils.featuretypes import join_feature_type_files
from .utils.log import setup_logging
from .utils.vectors import compute_impact_score

_cs = ConfigStore.instance()


def normalize_batch_aggregate(
    pipeline_dir: str,
    batch_stem: str,
    label_column: str,
    feature_select_types: Iterable[str],
    blocked_features: Optional[Iterable[str]] = None,
) -> pl.LazyFrame:
    """
    Join one batch's per-feature-type aggregates, drop globally-blocked
    feature columns, and normalize to its own synonymous baseline.

    Parameters
    ----------
    pipeline_dir : str
        Absolute path to the pipeline's root output directory.
    batch_stem : str
        The batch's identifier (matches ``feature_select_batchwise/<batch_stem>``).
    label_column : str
        Name of the column identifying variant labels.
    feature_select_types : Iterable[str]
        The currently-configured feature types (e.g. ``mean``, ``median``,
        ``MAD``, ``std``, ``KS``, ``QQ``, ``AUROC`` — see
        :data:`fisseq_data_pipeline.aggregate._AGGREGATORS`). Only aggregate
        files whose stem (``<feature_type>.parquet``) is in this set are
        joined; any other file present in the batch's ``aggregates/``
        directory (e.g. a stale file for a feature type no longer
        configured, left behind by an earlier run's larger
        ``feature_select_types`` since ``AGGREGATE_FEATURE_TYPE``'s
        ``publishDir mode: 'copy'`` never deletes removed outputs) is
        ignored. A configured type missing its file for this batch (e.g. its
        ``AGGREGATE_FEATURE_TYPE`` task failed under ``errorStrategy
        'ignore'``) is likewise silently skipped — see
        :func:`median_across_batches` for how cross-batch column mismatches
        are handled downstream.
    blocked_features : Iterable[str] or None
        Feature column names to drop before normalization (e.g. the
        globally-blocked set from :func:`combine_batch_blocklists`). Names
        not present in this batch's joined aggregate are silently ignored.
        Defaults to ``None`` (no columns dropped).

    Returns
    -------
    pl.LazyFrame
        The batch's combined per-variant aggregate table, with
        ``blocked_features`` dropped, z-score normalized to its own
        synonymous (control) rows.

    Raises
    ------
    ValueError
        If no per-feature-type aggregate files matching
        ``feature_select_types`` are found for this batch.
    """
    glob_pattern = (
        f"{pipeline_dir}/feature_select_batchwise/{batch_stem}/aggregates/*.parquet"
    )
    allowed_types = set(feature_select_types)
    paths = sorted(
        p for p in glob.glob(glob_pattern) if pathlib.Path(p).stem in allowed_types
    )
    if not paths:
        raise ValueError(
            f"No per-feature-type aggregate files matching feature_select_types="
            f"{sorted(allowed_types)!r} found for batch {batch_stem!r} "
            f"(glob pattern: {glob_pattern!r})"
        )
    agg_df = join_feature_type_files(paths, label_column)
    if blocked_features:
        agg_df = agg_df.drop([c for c in blocked_features if c in agg_df.columns])
    lf = variant_classification(agg_df.lazy(), label_column)
    normalizer = Normalizer.from_lazyframe(lf, fit_only_on_control=True)
    return normalizer.apply(lf)


def median_across_batches(
    batch_lfs: List[pl.LazyFrame],
    label_column: str,
    batch_labels: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Align every member batch's normalized aggregate table to a common
    schema, concatenate, and take the per-feature median grouped by
    ``label_column``.

    Member batches are not guaranteed to share an identical schema (see the
    module docstring) — before concatenating, each batch frame is reduced
    to ``label_column`` plus the intersection of feature columns (matched by
    ``FEATURE_SELECTOR``) present in *every* batch frame. Any feature column
    not common to all batches is dropped (logged as a warning per batch,
    since silently losing features cross-batch is worth being loud about),
    and every metadata column other than ``label_column`` is dropped
    unconditionally rather than reconciled.

    A variant appearing in multiple batches is collapsed to a single row;
    a variant appearing in only one batch passes through unchanged (the
    median of one value is that value).

    Parameters
    ----------
    batch_lfs : list[pl.LazyFrame]
        Each member batch's normalized aggregate table (see
        :func:`normalize_batch_aggregate`). Must be non-empty.
    label_column : str
        Name of the column identifying variant labels, used as the group key.
    batch_labels : list[str] or None
        Optional per-batch identifiers (e.g. batch stems), used only to
        name batches in the dropped-column warning. Defaults to ``None``
        (batches are identified by position instead).

    Returns
    -------
    pl.DataFrame
        One row per variant, with every common feature column set to its
        cross-batch median.

    Raises
    ------
    ValueError
        If ``batch_lfs`` is empty, or if no feature column is common to
        every batch.
    """
    if not batch_lfs:
        raise ValueError("batch_lfs must be non-empty")
    if batch_labels is None:
        batch_labels = [str(i) for i in range(len(batch_lfs))]

    per_batch_feature_cols = [
        set(lf.select(FEATURE_SELECTOR).collect_schema().names()) - {label_column}
        for lf in batch_lfs
    ]
    common_feature_cols = sorted(set.intersection(*per_batch_feature_cols))
    if not common_feature_cols:
        raise ValueError(
            "No feature column is common to every batch; nothing to median "
            "across batches"
        )

    aligned_lfs = []
    for label, lf, batch_cols in zip(batch_labels, batch_lfs, per_batch_feature_cols):
        dropped = sorted(batch_cols - set(common_feature_cols))
        if dropped:
            logging.warning(
                "Batch %s: dropping %d feature column(s) not common to every "
                "batch's aggregate before cross-batch concat: %s",
                label,
                len(dropped),
                dropped,
            )
        aligned_lfs.append(lf.select([label_column, *common_feature_cols]))

    lf = pl.concat(aligned_lfs)
    return (
        lf.group_by(label_column)
        .agg([pl.col(c).median() for c in common_feature_cols])
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
    feature_select_types : list[str]
        The currently-configured feature types (e.g. ``mean``, ``median``,
        ``MAD``, ``std``, ``KS``, ``QQ``, ``AUROC`` — mirrors the
        pipeline-wide ``feature_select_types`` Nextflow param). Used to
        filter each member batch's ``aggregates/`` directory down to files
        for these types, so stale files for feature types no longer
        configured (left behind by an earlier run) are ignored — see
        :func:`normalize_batch_aggregate`. Required.
    label_column : str
        Name of the column identifying variant labels. Defaults to
        ``meta_aa_changes``.
    min_batches_ok : int or None
        Minimum number of member batches that must mark a feature ok for it
        to be globally ok. Defaults to ``None`` (unanimity across batches
        that report on it) — see :func:`combine_batch_blocklists`.
    compute_impact_score : bool
        If ``True``, compute per-variant impact score (cosine distance vs
        synonymous baseline) after feature selection. Defaults to ``True``.
    run_pca : bool
        If ``True``, compute PCA on the final selected/normalized feature
        matrix (see :func:`.utils.dimreduction.compute_pca`), appending
        ``meta_pc_1..meta_pc_{pca_n_components}`` and writing a separate
        PCA-components output file. Defaults to ``False``.
    pca_n_components : int
        Number of principal components to compute and retain. Arbitrary
        default -- tune to the dataset's actual post-selection feature
        count. Must be ``<= min(n_rows, n_retained_features)`` after
        all-null feature columns are dropped, or the run fails. Defaults to
        ``10``.
    run_umap : bool
        If ``True``, compute UMAP on the final selected/normalized feature
        matrix (see :func:`.utils.dimreduction.compute_umap`), appending
        ``meta_umap_1..meta_umap_{umap_n_components}``. Defaults to
        ``False``.
    umap_n_components : int
        Dimensionality of the UMAP embedding. Defaults to ``2``.
    umap_n_neighbors : int
        ``umap.UMAP``'s local neighborhood size. Defaults to ``10``.
    umap_metric : str
        ``umap.UMAP``'s distance metric. Defaults to ``"cosine"``.
    umap_min_dist : float
        ``umap.UMAP``'s minimum embedded distance between points. Defaults
        to ``0.1``.
    umap_random_state : int or None
        Seed for UMAP's fit. ``None`` disables seeding, enabling faster
        nondeterministic multithreaded fitting. Defaults to ``42``.
    """

    pipeline_dir: str = MISSING
    batch_stems: List[str] = MISSING
    feature_select_types: List[str] = MISSING
    label_column: str = "meta_aa_changes"
    min_batches_ok: Optional[int] = None
    compute_impact_score: bool = True
    run_pca: bool = False
    pca_n_components: int = 10
    run_umap: bool = False
    umap_n_components: int = 2
    umap_n_neighbors: int = 10
    umap_metric: str = "cosine"
    umap_min_dist: float = 0.1
    umap_random_state: Optional[int] = 42


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
      aggregate table, with a ``meta_is_control`` column and (when
      ``compute_impact_score`` is ``True``) a ``meta_impact_score`` column,
      plus (when ``run_pca``/``run_umap`` is ``True``)
      ``meta_pc_*``/``meta_umap_*`` embedding columns computed independently
      on the same selected/normalized feature matrix (see
      :func:`.utils.dimreduction.compute_pca`/``compute_umap``).
    - ``{output_dir}/blocklist.parquet`` — the combined global blocklist.
    - ``{output_dir}/pca_components.parquet`` — only when ``run_pca`` is
      ``True``: one row per principal component (see
      :func:`.utils.dimreduction.compute_pca`).

    Raises
    ------
    ValueError
        If ``batch_stems`` is empty, if any member batch is missing its
        BATCHWISE aggregate or blocklist files, or if no feature column is
        common to every member batch's aggregate (see
        :func:`median_across_batches`).
    """
    gfs_cfg: GlobalFeatureSelectConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(gfs_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gfs_cfg.output_dir = output_dir
    setup_logging(gfs_cfg, "global_feature_select")

    if not gfs_cfg.batch_stems:
        raise ValueError("batch_stems must be a non-empty list")

    logging.info("Combining per-batch blocklists")
    bl_paths = [
        f"{gfs_cfg.pipeline_dir}/feature_select_batchwise/{stem}/blocklist.parquet"
        for stem in gfs_cfg.batch_stems
    ]
    bl_df = combine_batch_blocklists(bl_paths, gfs_cfg.min_batches_ok)
    blocked_features = set(bl_df.filter(~pl.col("feature_ok"))["feature"].to_list())

    logging.info(
        "Normalizing per-batch aggregates for %d batch(es)", len(gfs_cfg.batch_stems)
    )
    batch_lfs = [
        normalize_batch_aggregate(
            gfs_cfg.pipeline_dir,
            stem,
            gfs_cfg.label_column,
            gfs_cfg.feature_select_types,
            blocked_features,
        )
        for stem in gfs_cfg.batch_stems
    ]

    logging.info("Computing cross-batch median aggregate")
    agg_df = median_across_batches(
        batch_lfs, gfs_cfg.label_column, batch_labels=gfs_cfg.batch_stems
    )

    logging.info("Running pycytominer feature selection")
    selected_df = select_global_aggregate(agg_df, bl_df)

    if gfs_cfg.compute_impact_score:
        logging.info(
            "Classifying variants and marking synonymous as impact-score reference"
        )
        selected_lf = variant_classification(selected_df.lazy(), gfs_cfg.label_column)
        logging.info("Computing impact scores")
        selected_df = compute_impact_score(selected_lf).collect()

    # PCA and UMAP are computed independently, both on the same
    # selected/normalized feature matrix -- UMAP does not run on PCA's
    # output. selected_df is already eager here, so no collect()/lazy() is
    # needed (unlike featureselect.py's LazyFrame equivalent).
    pca_components_df = None
    if gfs_cfg.run_pca:
        logging.info("Computing PCA (%d components)", gfs_cfg.pca_n_components)
        pca_scores_df, pca_components_df = compute_pca(
            selected_df, gfs_cfg.label_column, gfs_cfg.pca_n_components
        )
        selected_df = selected_df.join(pca_scores_df, on=gfs_cfg.label_column)

    if gfs_cfg.run_umap:
        logging.info("Computing UMAP (%d components)", gfs_cfg.umap_n_components)
        umap_scores_df = compute_umap(
            selected_df,
            gfs_cfg.label_column,
            gfs_cfg.umap_n_components,
            gfs_cfg.umap_n_neighbors,
            gfs_cfg.umap_metric,
            gfs_cfg.umap_min_dist,
            gfs_cfg.umap_random_state,
        )
        selected_df = selected_df.join(umap_scores_df, on=gfs_cfg.label_column)

    agg_path = output_dir / "aggregate.parquet"
    bl_path = output_dir / "blocklist.parquet"
    logging.info("Writing aggregate to %s", agg_path)
    selected_df.write_parquet(agg_path)
    logging.info("Writing blocklist to %s", bl_path)
    bl_df.write_parquet(bl_path)

    if gfs_cfg.run_pca:
        pca_components_path = output_dir / "pca_components.parquet"
        logging.info("Writing PCA components to %s", pca_components_path)
        pca_components_df.write_parquet(pca_components_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
