"""Passthrough blocklist generation for non-WT-null feature types.

Hydra entry point backing the Nextflow process ``PASSTHROUGH_BLOCKLIST``: for
a feature type not in ``params.feature_select_wt_null_types`` (by default,
the summary-statistic aggregators ``mean``, ``median``, ``MAD``, ``std``,
which have no reference distribution to compare against and so have no
WT-null concept to apply), emits a trivial blocklist marking every feature
in that feature type's full aggregate ok, with no reproducibility
computation. Those features are still subject to pycytominer's variance/
correlation thresholds later, in ``FINALIZE_FEATURE_SELECT``.

A real CLI entry point (rather than an inline Nextflow shell one-liner) for
auditability and skill parity with :mod:`.wtnullblocklist`, its counterpart
for WT-null-eligible feature types. Both write the same blocklist schema —
see :mod:`.wtnullblocklist`'s module docstring.
"""

import dataclasses
import logging
import pathlib

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .utils.constants import FEATURE_SELECTOR
from .utils.log import setup_logging

_cs = ConfigStore.instance()


@dataclasses.dataclass
class PassthroughBlocklistConfig(AppConfig):
    """
    Hydra structured configuration for the passthrough blocklist generation
    entry point.

    Attributes
    ----------
    aggregate_file : str
        Path to this feature type's full aggregate parquet (output of
        :func:`fisseq_data_pipeline.aggregatefeaturetype.main`, i.e.
        ``AGGREGATE_FEATURE_TYPE``'s output). Required.
    """

    aggregate_file: str = MISSING


_cs.store(name="passthrough_blocklist_main", node=PassthroughBlocklistConfig)


@hydra.main(
    version_base=None, config_path=None, config_name="passthrough_blocklist_main"
)
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: emit a trivial all-ok blocklist for one feature type.

    Scans ``aggregate_file``'s schema (without loading any data) and
    enumerates its feature columns via the same ``FEATURE_SELECTOR`` used
    throughout the pipeline to distinguish feature columns from ``meta_*``
    columns (which also excludes the label column, e.g.
    ``meta_aa_changes``). Every feature is marked ``feature_ok = True``;
    ``null_mean``/``threshold``/``n_bootstraps`` are ``null`` (no
    reproducibility computation was performed) -- the same schema
    :mod:`.wtnullblocklist` writes, so :mod:`.combineblocklists`'s plain
    concat keeps working unmodified.

    Output file
    -----------
    - ``{output_dir}/blocklist.parquet`` with columns ``feature``,
      ``feature_ok``, ``null_mean``, ``threshold``, ``n_bootstraps``.
    """
    pt_cfg: PassthroughBlocklistConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(pt_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_cfg.output_dir = output_dir
    setup_logging(pt_cfg, "passthrough_blocklist")

    logging.info("Scanning schema of %s", pt_cfg.aggregate_file)
    features = (
        pl.scan_parquet(pt_cfg.aggregate_file)
        .select(FEATURE_SELECTOR)
        .collect_schema()
        .names()
    )
    logging.info(
        "Marking %d feature(s) ok (passthrough, no WT-null check)", len(features)
    )

    blocklist_df = pl.DataFrame(
        {"feature": pl.Series(features, dtype=pl.String)}
    ).with_columns(
        pl.lit(True).alias("feature_ok"),
        pl.lit(None, dtype=pl.Float64).alias("null_mean"),
        pl.lit(None, dtype=pl.Float64).alias("threshold"),
        pl.lit(None, dtype=pl.Int64).alias("n_bootstraps"),
    )

    out_path = output_dir / "blocklist.parquet"
    logging.info("Writing blocklist to %s", out_path)
    blocklist_df.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
