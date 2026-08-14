"""Per-feature-type bootstrap blocklist generation.

Hydra entry point backing the Nextflow process ``BLOCKLIST``: derives a
per-feature blocklist from a Fisher-z-averaged correlation estimate (with a
paired precision/quality gate) across bootstrap replicates (outputs of
:func:`fisseq_data_pipeline.correlatefeatures.main`), part of the bootstrap
feature-selection pipeline.
"""

import dataclasses
import glob
import logging
import pathlib

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .utils.log import setup_logging

_cs = ConfigStore.instance()

# Clip |r| to at most 1 - _R_CLIP_EPS before ``arctanh`` so r == +/-1 (e.g. two
# perfectly correlated halves) maps to a large-but-finite Fisher z rather than
# +/-inf. Distinct from utils.constants.EPS (a different tolerance used for
# near-zero-variance checks elsewhere) -- not reused here on purpose.
_R_CLIP_EPS: float = 1e-6


@dataclasses.dataclass
class BlocklistConfig(AppConfig):
    """
    Hydra structured configuration for the per-feature-type blocklist
    generation entry point.

    Attributes
    ----------
    correlation_files : str
        Glob pattern matching all bootstrap-replicate correlation parquet
        files for one feature type (outputs of
        :func:`fisseq_data_pipeline.correlatefeatures.main`). Required.
    minimum_correlation : float
        Magnitude gate: minimum Fisher-z-averaged Pearson *r* estimate
        (``r_est``) across bootstrap replicates required for a feature to
        pass. A feature must also clear the quality gate (``max_se_z``) to be
        marked ``feature_ok`` -- this threshold alone is not sufficient.
        Defaults to ``0.5``.
    max_se_z : float
        Quality/precision gate: maximum acceptable standard error of the mean
        Fisher-z estimate (``se_z``) across bootstrap replicates. The default
        ``0.0884`` is calibrated for the pipeline's default
        ``bootstrap_reps=10`` (``se_z = 0.2 / t_crit(df=9, 0.975) ~= 0.0884``),
        targeting a ~95% CI half-width of ~0.15 in *r* near ``r=0.5``. If
        ``bootstrap_reps`` is ever changed from its default, rescale this
        value by roughly ``sqrt(10 / new_bootstrap_reps)``, or re-derive it
        from scratch via the retroactive replicate-resampling check. A
        feature with fewer than 2 usable (non-null) replicates has
        ``se_z = null`` and automatically fails this gate -- a single
        replicate can't support a precision claim. Defaults to ``0.0884``.
    """

    correlation_files: str = MISSING
    minimum_correlation: float = 0.5
    max_se_z: float = 0.0884


_cs.store(name="blocklist_main", node=BlocklistConfig)


@hydra.main(version_base=None, config_path=None, config_name="blocklist_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: compute a per-feature-type blocklist from N bootstrap
    correlation tables.

    This is the one intentional synchronization point across bootstrap
    replicates in the feature-selection pipeline. Globs
    ``correlation_files``, concatenates all bootstrap-replicate correlation
    tables for one feature type, and for each feature Fisher-z-transforms
    every replicate's ``r`` (``z = arctanh(clip(r, -1+eps, 1-eps))``),
    averages in z-space, and back-transforms to get a point estimate
    (``r_est = tanh(mean(z))``) plus its standard error
    (``se_z = std(z, ddof=1) / sqrt(n_replicates)``). A feature is marked
    ``feature_ok`` only if it clears both an independent magnitude gate
    (``r_est >= minimum_correlation``) and a quality/precision gate
    (``se_z`` is defined and ``<= max_se_z``); a feature with fewer than 2
    usable replicates has ``se_z = null`` and fails the quality gate
    automatically.

    Output file
    -----------
    - ``{output_dir}/blocklist.parquet`` with columns ``feature``,
      ``r_est``, ``se_z``, ``n_replicates``, ``feature_ok``.

    Raises
    ------
    ValueError
        If ``correlation_files`` matches no files.
    """
    bl_cfg: BlocklistConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(bl_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bl_cfg.output_dir = output_dir
    setup_logging(bl_cfg, "blocklist")

    paths = sorted(glob.glob(bl_cfg.correlation_files))
    if not paths:
        raise ValueError(f"No files matched glob pattern: {bl_cfg.correlation_files!r}")
    logging.info("Found %d bootstrap correlation file(s)", len(paths))
    corr_df = pl.concat([pl.read_parquet(p) for p in paths])

    blocklist_df = (
        corr_df.with_columns(
            pl.col("r")
            .clip(-1 + _R_CLIP_EPS, 1 - _R_CLIP_EPS)
            .arctanh()
            .alias("_z")
        )
        .group_by("feature")
        .agg(
            pl.col("_z").count().alias("n_replicates"),
            pl.col("_z").mean().alias("_z_mean"),
            pl.col("_z").std(ddof=1).alias("_std_z"),
        )
        .with_columns(pl.col("_z_mean").tanh().alias("r_est"))
        .with_columns(
            pl.when(pl.col("n_replicates") <= 1)
            .then(None)
            .otherwise(pl.col("_std_z") / pl.col("n_replicates").cast(pl.Float64).sqrt())
            .alias("se_z")
        )
        .with_columns(
            (
                (pl.col("r_est") >= bl_cfg.minimum_correlation)
                & pl.col("se_z").is_not_null()
                & (pl.col("se_z") <= bl_cfg.max_se_z)
            ).alias("feature_ok")
        )
        .select("feature", "r_est", "se_z", "n_replicates", "feature_ok")
    )

    out_path = output_dir / "blocklist.parquet"
    logging.info("Writing blocklist to %s", out_path)
    blocklist_df.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
