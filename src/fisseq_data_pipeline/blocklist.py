"""Per-feature-type bootstrap blocklist generation.

Hydra entry point backing the Nextflow process ``BLOCKLIST``: derives a
per-feature blocklist from the median correlation across bootstrap replicates
(outputs of :func:`fisseq_data_pipeline.correlatefeatures.main`), part of the
bootstrap feature-selection pipeline.
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
        Minimum median Pearson *r* (across bootstrap replicates) required for
        a feature to pass. Defaults to ``0.5``.
    """

    correlation_files: str = MISSING
    minimum_correlation: float = 0.5


_cs.store(name="blocklist_main", node=BlocklistConfig)


@hydra.main(version_base=None, config_path=None, config_name="blocklist_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: compute a per-feature-type blocklist from N bootstrap
    correlation tables.

    This is the one intentional synchronization point across bootstrap
    replicates in the feature-selection pipeline. Globs
    ``correlation_files``, concatenates all bootstrap-replicate correlation
    tables for one feature type, computes each feature's median ``r`` across
    replicates, and marks ``feature_ok = median_r >= minimum_correlation``.

    Output file
    -----------
    - ``{output_dir}/blocklist.parquet`` with columns ``feature``,
      ``median_r``, ``feature_ok``.

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
        corr_df.group_by("feature")
        .agg(pl.col("r").median().alias("median_r"))
        .with_columns(
            (pl.col("median_r") >= bl_cfg.minimum_correlation).alias("feature_ok")
        )
    )

    out_path = output_dir / "blocklist.parquet"
    logging.info("Writing blocklist to %s", out_path)
    blocklist_df.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
