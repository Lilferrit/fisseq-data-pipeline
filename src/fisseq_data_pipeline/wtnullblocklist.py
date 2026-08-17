"""WT-null bootstrap blocklist generation.

Hydra entry point backing the Nextflow process ``WT_NULL_BLOCKLIST``: derives
a per-feature-type blocklist from N WT-null bootstrap replicates (outputs of
:func:`fisseq_data_pipeline.wtnullaggregate.main`) by averaging each
feature's null-noise value across bootstraps and applying an upper Tukey
fence over the batch's per-feature null-mean distribution. Part of the
bootstrap feature-selection pipeline; replaces the old Fisher-z-correlation
``blocklist.py``.
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


@dataclasses.dataclass
class WtNullBlocklistConfig(AppConfig):
    """
    Hydra structured configuration for the WT-null blocklist generation
    entry point.

    Attributes
    ----------
    wt_null_files : str
        Glob pattern matching all bootstrap-replicate WT-null parquet files
        for one feature type (outputs of
        :func:`fisseq_data_pipeline.wtnullaggregate.main`). Required.
    tukey_multiplier : float
        IQR multiplier for the upper reproducibility fence: a feature is
        blocked if its null mean exceeds ``Q1(null means) +
        tukey_multiplier * IQR(null means)``, computed over every feature's
        finite null mean for this feature type. Defaults to ``1.5``.

        NOTE: this fence is anchored on **Q1**, not the conventional Q3
        anchor of a textbook upper Tukey fence (``Q3 + 1.5*IQR``) -- a
        deliberate, confirmed choice, making the default a stricter cutoff
        than a standard outlier fence. Since Q1 <= Q3 always, this fence
        never sits above the standard one for the same multiplier.
    """

    wt_null_files: str = MISSING
    tukey_multiplier: float = 1.5


_cs.store(name="wt_null_blocklist_main", node=WtNullBlocklistConfig)


@hydra.main(version_base=None, config_path=None, config_name="wt_null_blocklist_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: compute a per-feature-type WT-null blocklist from N
    bootstrap replicates.

    Globs ``wt_null_files``, concatenates all bootstrap-replicate long
    ``(feature, value)`` tables for one feature type, and for each feature
    averages ``value`` across bootstraps (``null_mean``), skipping
    non-finite bootstrap values (e.g. a degenerate/constant WT
    sub-distribution). A feature with zero finite replicate values has
    ``null_mean = null`` and ``n_bootstraps = 0``, and is unconditionally
    blocked (``feature_ok = False``) -- its own reproducibility can't be
    established, and it is excluded from the Tukey-fence quantile
    computation below so it can't skew the fence for every other feature.

    The fence is ``Q1(null_mean) + tukey_multiplier * IQR(null_mean)``,
    computed over every feature's finite ``null_mean`` (see
    ``WtNullBlocklistConfig.tukey_multiplier`` for why this anchors on Q1,
    not the conventional Q3). A feature passes (``feature_ok = True``) iff
    its own ``null_mean`` is finite and does not exceed the fence.

    Output file
    -----------
    - ``{output_dir}/blocklist.parquet`` with columns ``feature``,
      ``feature_ok``, ``null_mean``, ``threshold``, ``n_bootstraps``.

    Raises
    ------
    ValueError
        If ``wt_null_files`` matches no files.
    """
    bl_cfg: WtNullBlocklistConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(bl_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bl_cfg.output_dir = output_dir
    setup_logging(bl_cfg, "wt_null_blocklist")

    paths = sorted(glob.glob(bl_cfg.wt_null_files))
    if not paths:
        raise ValueError(f"No files matched glob pattern: {bl_cfg.wt_null_files!r}")
    logging.info("Found %d WT-null bootstrap replicate file(s)", len(paths))
    long_df = pl.concat([pl.read_parquet(p) for p in paths])

    per_feature = long_df.group_by("feature").agg(
        pl.col("value").filter(pl.col("value").is_finite()).mean().alias("null_mean"),
        pl.col("value").is_finite().sum().cast(pl.Int64).alias("n_bootstraps"),
    )

    finite_null_means = per_feature["null_mean"].drop_nulls()
    if finite_null_means.len() == 0:
        logging.warning(
            "No feature has a finite null_mean across all bootstraps -- every "
            "feature will be blocked."
        )
        fence = None
    else:
        q1 = finite_null_means.quantile(0.25, interpolation="linear")
        q3 = finite_null_means.quantile(0.75, interpolation="linear")
        fence = q1 + bl_cfg.tukey_multiplier * (q3 - q1)
        logging.info(
            "Tukey fence: Q1=%.6g, Q3=%.6g, multiplier=%.3g -> threshold=%.6g",
            q1,
            q3,
            bl_cfg.tukey_multiplier,
            fence,
        )

    if fence is None:
        per_feature = per_feature.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("threshold"),
            pl.lit(False).alias("feature_ok"),
        )
    else:
        per_feature = per_feature.with_columns(
            pl.lit(fence, dtype=pl.Float64).alias("threshold"),
            (pl.col("null_mean").is_not_null() & (pl.col("null_mean") <= fence)).alias(
                "feature_ok"
            ),
        )

    blocklist_df = per_feature.select(
        "feature", "feature_ok", "null_mean", "threshold", "n_bootstraps"
    )

    out_path = output_dir / "blocklist.parquet"
    logging.info("Writing blocklist to %s", out_path)
    blocklist_df.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
