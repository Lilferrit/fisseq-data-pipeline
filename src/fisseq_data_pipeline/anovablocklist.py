"""Feature block-list derived from per-feature ANOVA p-values.

Hydra entry point ``fisseq-anova-blocklist``, backing the Nextflow process
``ANOVA_BLOCKLIST``. Consumes the output of :mod:`.anova` as-is (columns
``feature``, ``f_value``, ``p_value``) and marks each feature ``feature_ok``
based on whether its ANOVA p-value indicates a statistically significant
batch effect: a feature is blocked (``feature_ok = False``) when
``p_value < pvalue_threshold`` (a significant batch effect was detected),
matching the ``block_list_file`` convention already used by
:class:`fisseq_data_pipeline.aggregate.AggregateConfig` and
:class:`fisseq_data_pipeline.features.FinalizeFeatureSelectConfig`.
"""

import dataclasses
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
class AnovaBlocklistConfig(AppConfig):
    """
    Hydra structured configuration for the ANOVA block-list entry point.

    Attributes
    ----------
    anova_file : str
        Path to the ANOVA results parquet file (output of
        :func:`fisseq_data_pipeline.anova.main`), with columns ``feature``,
        ``f_value``, and ``p_value``. Required.
    pvalue_threshold : float
        A feature is blocked (``feature_ok = False``) when its ANOVA
        ``p_value`` is strictly less than this threshold, i.e. when a
        statistically significant batch effect was detected. Defaults to
        ``0.05``.
    """

    anova_file: str = MISSING
    pvalue_threshold: float = 0.05


_cs.store(name="anova_blocklist_main", node=AnovaBlocklistConfig)


@hydra.main(version_base=None, config_path=None, config_name="anova_blocklist_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: derive a feature block-list from ANOVA p-values.

    Reads ``anova_file`` and marks ``feature_ok = p_value >= pvalue_threshold``
    for every feature -- i.e. a feature is blocked when its p-value indicates
    a statistically significant batch effect. Emits one row per input
    feature (not just the blocked ones), matching the convention used by
    :func:`fisseq_data_pipeline.features.blocklist_main`.

    Output file
    -----------
    - ``{output_dir}/anova_blocklist.parquet`` with columns ``feature``,
      ``p_value``, ``feature_ok``.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.anovablocklist \\
            output_dir=./out \\
            anova_file=./out/anova.parquet \\
            pvalue_threshold=0.05
    """
    bl_cfg: AnovaBlocklistConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(bl_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bl_cfg.output_dir = output_dir
    setup_logging(bl_cfg, "anova_blocklist")

    logging.info("Loading ANOVA results from %s", bl_cfg.anova_file)
    anova_df = pl.read_parquet(bl_cfg.anova_file)

    blocklist_df = anova_df.select("feature", "p_value").with_columns(
        (pl.col("p_value") >= bl_cfg.pvalue_threshold).alias("feature_ok")
    )
    logging.info(
        "Blocked %d of %d feature(s) at pvalue_threshold=%s",
        int((~blocklist_df["feature_ok"]).sum()),
        len(blocklist_df),
        bl_cfg.pvalue_threshold,
    )

    out_path = output_dir / "anova_blocklist.parquet"
    logging.info("Writing block-list to %s", out_path)
    blocklist_df.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
