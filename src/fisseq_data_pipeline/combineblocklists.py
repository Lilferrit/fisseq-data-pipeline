"""Combine per-feature-type blocklists.

Hydra entry point backing the Nextflow process ``COMBINE_BLOCKLISTS``:
concatenates all per-feature-type blocklists (outputs of
:func:`fisseq_data_pipeline.blocklist.main`) into one combined blocklist, part of
the bootstrap feature-selection pipeline.
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
class CombineBlocklistsConfig(AppConfig):
    """
    Hydra structured configuration for the blocklist-combination entry point.

    Attributes
    ----------
    blocklist_files : str
        Glob pattern matching all per-feature-type blocklist parquet files
        (outputs of :func:`fisseq_data_pipeline.blocklist.main`). Required.
    """

    blocklist_files: str = MISSING


_cs.store(name="combine_blocklists_main", node=CombineBlocklistsConfig)


@hydra.main(version_base=None, config_path=None, config_name="combine_blocklists_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: concatenate all per-feature-type blocklists into one
    combined blocklist.

    Globs ``blocklist_files`` and concatenates them with no deduplication —
    each feature type's blocklist covers a disjoint set of feature columns
    (stat-suffixed column names like ``f1_mean`` and ``f1_KS`` never
    collide across feature types), so a plain concat is correct.

    Output file
    -----------
    - ``{output_dir}/blocklist.parquet``

    Raises
    ------
    ValueError
        If ``blocklist_files`` matches no files.
    """
    cb_cfg: CombineBlocklistsConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(cb_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cb_cfg.output_dir = output_dir
    setup_logging(cb_cfg, "combine_blocklists")

    paths = sorted(glob.glob(cb_cfg.blocklist_files))
    if not paths:
        raise ValueError(f"No files matched glob pattern: {cb_cfg.blocklist_files!r}")
    logging.info("Found %d per-feature-type blocklist file(s)", len(paths))
    combined = pl.concat([pl.read_parquet(p) for p in paths])

    out_path = output_dir / "blocklist.parquet"
    logging.info("Writing combined blocklist to %s", out_path)
    combined.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
