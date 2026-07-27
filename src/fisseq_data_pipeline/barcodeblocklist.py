"""Barcode block-list derived from per-barcode CHECK_BARCODES p-values.

Hydra entry point ``fisseq-barcode-blocklist``, backing the Nextflow process
``BARCODE_BLOCKLIST`` (per batch, same cadence as ``CHECK_BARCODES`` -- unlike
``ANOVA_BLOCKLIST``, which runs once globally). Consumes the output of
:mod:`.checkbarcodes` (columns ``variant``, ``barcode``, ``group_mean``,
``comparison_barcode``, ``comparison_group_mean``, ``mean_diff``, ``p_adj``,
``reject`` -- one row per compared barcode pair within a variant) and marks
each barcode ``barcode_ok`` based on the median of its ``p_adj`` values
across every comparison it took part in: a barcode is blocked
(``barcode_ok = False``) when that median is strictly less than
``pvalue_threshold`` (its cells score anomalously relative to the variant's
other barcodes), matching the ``feature_ok`` convention of
:mod:`.anovablocklist`.

Each pair's two barcodes are recorded as ``barcode`` (alphabetically first)
and ``comparison_barcode`` (alphabetically second), so a given barcode's
p_adj values are scattered across both columns depending on what it happens
to be paired against. Both columns are unioned before aggregating so no
barcode's data is dropped just because it never sorted first.
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
class BarcodeBlocklistConfig(AppConfig):
    """
    Hydra structured configuration for the barcode block-list entry point.

    Attributes
    ----------
    check_barcodes_file : str
        Path to a single batch's CHECK_BARCODES results parquet (output of
        :func:`fisseq_data_pipeline.checkbarcodes.main`), with columns
        ``barcode``, ``comparison_barcode``, and ``p_adj``. Required.
    pvalue_threshold : float
        A barcode is blocked (``barcode_ok = False``) when the median of its
        ``p_adj`` values is strictly less than this threshold. Defaults to
        ``0.05``.
    """

    check_barcodes_file: str = MISSING
    pvalue_threshold: float = 0.05


_cs.store(name="barcode_blocklist_main", node=BarcodeBlocklistConfig)

_BLOCKLIST_SCHEMA = {
    "barcode": pl.Utf8,
    "p_adj": pl.Float64,
    "barcode_ok": pl.Boolean,
}


def compute_barcode_blocklist(
    results_df: pl.DataFrame, pvalue_threshold: float
) -> pl.DataFrame:
    """
    Aggregate CHECK_BARCODES' pairwise results into a per-barcode block-list.

    Parameters
    ----------
    results_df : pl.DataFrame
        CHECK_BARCODES' ``results.parquet`` for a single batch -- one row
        per compared barcode pair, with columns ``barcode``,
        ``comparison_barcode``, and ``p_adj``.
    pvalue_threshold : float
        A barcode is blocked (``barcode_ok = False``) when the median of its
        ``p_adj`` values is strictly less than this threshold.

    Returns
    -------
    pl.DataFrame
        One row per distinct barcode, with columns ``barcode``, ``p_adj``
        (median across all comparisons that barcode took part in), and
        ``barcode_ok``. Empty (but correctly typed) if ``results_df`` is
        empty.
    """
    if results_df.height == 0:
        return pl.DataFrame(schema=_BLOCKLIST_SCHEMA)

    # Union both sides of each pair -- a barcode's p_adj values are split
    # across `barcode` and `comparison_barcode` depending on which side of
    # the alphabetical dedup it landed on for a given pair.
    long_df = pl.concat(
        [
            results_df.select(pl.col("barcode"), pl.col("p_adj")),
            results_df.select(
                pl.col("comparison_barcode").alias("barcode"), pl.col("p_adj")
            ),
        ]
    )

    return (
        long_df.group_by("barcode")
        .agg(pl.col("p_adj").median().alias("p_adj"))
        .with_columns((pl.col("p_adj") >= pvalue_threshold).alias("barcode_ok"))
        .select(list(_BLOCKLIST_SCHEMA.keys()))
    )


@hydra.main(version_base=None, config_path=None, config_name="barcode_blocklist_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: derive a barcode block-list from CHECK_BARCODES p-values.

    Reads ``check_barcodes_file``, computes each barcode's median ``p_adj``
    via :func:`compute_barcode_blocklist`, and marks
    ``barcode_ok = median_p_adj >= pvalue_threshold``.

    Output file
    -----------
    - ``{output_dir}/barcode_blocklist.parquet`` with columns ``barcode``,
      ``p_adj``, ``barcode_ok``.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.barcodeblocklist \\
            output_dir=./out \\
            check_barcodes_file=./out/results.parquet \\
            pvalue_threshold=0.05
    """
    bl_cfg: BarcodeBlocklistConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(bl_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bl_cfg.output_dir = output_dir
    setup_logging(bl_cfg, "barcode_blocklist")

    logging.info("Loading CHECK_BARCODES results from %s", bl_cfg.check_barcodes_file)
    results_df = pl.read_parquet(bl_cfg.check_barcodes_file)

    blocklist_df = compute_barcode_blocklist(results_df, bl_cfg.pvalue_threshold)
    logging.info(
        "Blocked %d of %d barcode(s) at pvalue_threshold=%s",
        int((~blocklist_df["barcode_ok"]).sum()),
        len(blocklist_df),
        bl_cfg.pvalue_threshold,
    )

    out_path = output_dir / "barcode_blocklist.parquet"
    logging.info("Writing block-list to %s", out_path)
    blocklist_df.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
