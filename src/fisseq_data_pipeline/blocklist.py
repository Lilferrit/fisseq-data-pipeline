"""Per-feature-type bootstrap blocklist generation.

Hydra entry point backing the Nextflow process ``BLOCKLIST``: derives a
per-feature blocklist from a Fisher-z-averaged correlation estimate, with an
optional lower-confidence-bound precision adjustment (``se_multiplier``),
across bootstrap replicates (outputs of
:func:`fisseq_data_pipeline.correlatefeatures.main`), part of the bootstrap
feature-selection pipeline.
"""

import dataclasses
import glob
import logging
import pathlib
from typing import Optional

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
        Magnitude gate: minimum adjusted correlation estimate
        (``adjusted_r`` -- see ``se_multiplier``) required for a feature to
        pass. Defaults to ``0.5``.
    se_multiplier : float or None
        Precision/confidence adjustment applied in Fisher-z space before the
        magnitude gate: a feature's Fisher-z-averaged correlation estimate
        is penalized by ``se_multiplier`` standard errors before being
        compared to ``minimum_correlation``, giving an approximate
        lower-confidence-bound criterion (larger ``se_multiplier`` ->
        stricter). ``None`` disables the adjustment entirely and gates on
        the raw ``r_est`` point estimate -- NOT equivalent to the old
        two-gate ``max_se_z`` strategy's pass/fail semantics; it drops the
        precision requirement rather than replacing it with an
        unconditional pass. Defaults to ``1.0``.

        IMPORTANT design note: the subtraction happens in Fisher-z space
        (``adjusted_z = z_mean - se_multiplier * se_z``, then
        ``adjusted_r = tanh(adjusted_z)``), NOT directly on ``r_est``
        (``adjusted_r = r_est - se_multiplier * se_z``). This is
        deliberate: ``se_z`` is the standard error of the mean Fisher-z
        estimate, whose sampling distribution is approximately symmetric
        and unbounded, so a z-space shift is well-defined; ``r`` is bounded
        to ``[-1, 1]``, so subtracting a z-space-derived quantity directly
        from ``r_est`` mixes units and can behave oddly near ``r = +/-1``.
        If the r-space alternative is wanted instead, it is a one-line
        change in ``main()`` (see the comment there).

        A feature with fewer than 2 usable (non-null) replicates has
        ``se_z = null``, so ``adjusted_r`` (when ``se_multiplier`` is not
        ``None``) is also ``null`` and the feature automatically fails --
        a single replicate can't support a precision claim.
    """

    correlation_files: str = MISSING
    minimum_correlation: float = 0.5
    se_multiplier: Optional[float] = 1.0


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
    (``se_z = std(z, ddof=1) / sqrt(n_replicates)``). The precision-adjusted
    ``adjusted_r`` (see ``BlocklistConfig.se_multiplier``) is then compared
    to ``minimum_correlation`` to decide ``feature_ok``; a feature with
    fewer than 2 usable replicates has ``se_z = null`` and (when
    ``se_multiplier`` is set) fails automatically.

    Output file
    -----------
    - ``{output_dir}/blocklist.parquet`` with columns ``feature``,
      ``r_est``, ``se_z``, ``n_replicates``, ``adjusted_r``, ``feature_ok``.

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
    )

    # adjusted_r: se_multiplier=None -> raw r_est passthrough (no precision
    # penalty). Otherwise, the lower-confidence-bound adjustment is applied
    # in FISHER-Z SPACE -- see BlocklistConfig.se_multiplier's docstring for
    # the z-space-vs-r-space design rationale. One-line r-space alternative
    # instead:
    #     blocklist_df = blocklist_df.with_columns(
    #         (pl.col("r_est") - bl_cfg.se_multiplier * pl.col("se_z")).alias("adjusted_r")
    #     )
    if bl_cfg.se_multiplier is None:
        blocklist_df = blocklist_df.with_columns(pl.col("r_est").alias("adjusted_r"))
    else:
        blocklist_df = blocklist_df.with_columns(
            (pl.col("_z_mean") - bl_cfg.se_multiplier * pl.col("se_z"))
            .tanh()
            .alias("adjusted_r")
        )

    blocklist_df = blocklist_df.with_columns(
        (
            pl.col("adjusted_r").is_not_null()
            & (pl.col("adjusted_r") >= bl_cfg.minimum_correlation)
        ).alias("feature_ok")
    ).select("feature", "r_est", "se_z", "n_replicates", "adjusted_r", "feature_ok")

    out_path = output_dir / "blocklist.parquet"
    logging.info("Writing blocklist to %s", out_path)
    blocklist_df.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
