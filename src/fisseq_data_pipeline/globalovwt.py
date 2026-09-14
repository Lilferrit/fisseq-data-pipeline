"""GLOBAL_OVWT: cross-experiment correction and aggregation of OvWT scores.

Two steps, not one. Per experiment, z-score ``auroc_pooled`` and
``auroc_median_barcode`` against that experiment's own synonymous variants;
*then* take the cross-experiment median of the z-scored values -- not a direct
median of raw AUROC.

Raw AUROC is not comparable across experiments: differing cell counts, feature
quality, and batch effects all shift where a genuinely-neutral variant's
classifier score sits. Re-centering each experiment against its own synonymous
population first puts every experiment on a common scale before pooling.

This replaces OVWT_GLOBAL, which fit a single classifier on cells pooled across
batches. Aggregating per-experiment scores instead keeps each experiment's own
baseline intact rather than blending them into one.

Both halves reuse machinery that already exists in this package, unmodified:
:func:`~fisseq_data_pipeline.aggregate.variant_classification` flags synonymous,
untagged labels as controls, and
:class:`~fisseq_data_pipeline.normalize.Normalizer` fit with
``fit_only_on_control=True`` does the z-scoring. ``Normalizer.apply`` needs no
changes either -- it operates on ``FEATURE_SELECTOR`` (exclude ``meta_*``),
which already matches exactly ``auroc_pooled``/``auroc_median_barcode`` and
excludes ``meta_n_barcodes``/``meta_n_cells``, provided ``label_column`` itself
carries the conventional ``meta_`` prefix.
"""

import dataclasses
import logging
import pathlib
from typing import List

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .aggregate import variant_classification
from .config import AppConfig
from .normalize import Normalizer
from .utils.log import setup_logging
from .utils.nextflow_staging import reconstruct_staged_paths

_cs = ConfigStore.instance()


def global_variant_distinguishability(
    batch_score_dfs: List[pl.DataFrame], label_column: str
) -> pl.DataFrame:
    """
    Per-experiment synonymous z-score of both AUROC columns, then
    cross-experiment median.

    The normalizer is fit fresh per experiment, on that experiment's own
    OVWT_BATCHWISE ``results.parquet`` -- one row per variant, not one row per
    cell.

    Parameters
    ----------
    batch_score_dfs : list[pl.DataFrame]
        Each experiment's OVWT_BATCHWISE ``results.parquet``: ``label_column``,
        ``auroc_pooled``, ``auroc_median_barcode``, plus ``meta_n_barcodes`` /
        ``meta_n_cells``. Must be non-empty.
    label_column : str
        Name of the column identifying variant labels.

    Returns
    -------
    pl.DataFrame
        One row per variant, with ``meta_median_auroc_pooled`` and
        ``meta_median_auroc_median_barcode`` (cross-experiment medians of the
        per-experiment z-scored values), plus ``meta_num_experiments`` -- how
        many experiments' z-scored value for that variant was non-null and
        therefore contributed to the median.

    Raises
    ------
    ValueError
        If ``batch_score_dfs`` is empty.
    """
    if not batch_score_dfs:
        raise ValueError("batch_score_dfs must be non-empty")

    zscored_dfs = []
    for df in batch_score_dfs:
        classified = variant_classification(df.lazy(), label_column)
        normalizer = Normalizer.from_lazyframe(classified, fit_only_on_control=True)
        zscored_dfs.append(normalizer.apply(classified).collect())

    return (
        pl.concat(
            [
                df.select(label_column, "auroc_pooled", "auroc_median_barcode")
                for df in zscored_dfs
            ]
        )
        .group_by(label_column)
        .agg(
            pl.col("auroc_pooled").median().alias("meta_median_auroc_pooled"),
            pl.col("auroc_median_barcode")
            .median()
            .alias("meta_median_auroc_median_barcode"),
            pl.col("auroc_pooled").count().alias("meta_num_experiments"),
        )
    )


@dataclasses.dataclass
class GlobalOvwtConfig(AppConfig):
    """
    Hydra structured configuration for GLOBAL_OVWT.

    Extends :class:`~fisseq_data_pipeline.config.app.AppConfig` (``output_dir``,
    ``output_root``, ``log_level``, ``random_seed``). Nothing here is
    stochastic -- both the z-scoring and the median are deterministic -- but
    every stage config inherits ``random_seed`` regardless, so the Nextflow
    module can pass it uniformly.

    Attributes
    ----------
    batch_stems : List[str]
        This channel's experiment identifiers, one per contributing
        OVWT_BATCHWISE ``results.parquet``. Required, non-empty. Same order and
        length as the staged results files (see :func:`main`).
    label_column : str
        Name of the variant label column. Defaults to ``"meta_aa_changes"``.
    """

    batch_stems: List[str] = MISSING
    label_column: str = "meta_aa_changes"


_cs.store(name="global_ovwt_main", node=GlobalOvwtConfig)


@hydra.main(version_base=None, config_path=None, config_name="global_ovwt_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: per-experiment synonymous z-score, then cross-experiment
    median.

    Reads one ``results.parquet`` per entry in ``batch_stems``, staged by the
    calling Nextflow process as ``res_input_1.parquet``, ``res_input_2.parquet``,
    ... in the same order (see ``modules/local/global_ovwt.nf``, and
    :func:`~fisseq_data_pipeline.utils.nextflow_staging.reconstruct_staged_paths`
    for the single-file naming quirk).

    Output file
    -----------
    - ``{output_dir}/{prefix}global_scores.parquet``

    where ``prefix`` is ``{output_root}.`` when ``output_root`` is set,
    otherwise empty.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.globalovwt \\
            output_dir=./out \\
            'batch_stems=[plate1,plate2]'
    """
    gd_cfg: GlobalOvwtConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(gd_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gd_cfg.output_dir = str(output_dir)
    setup_logging(gd_cfg, "globalovwt")

    if not gd_cfg.batch_stems:
        raise ValueError("batch_stems must be a non-empty list")

    prefix = f"{gd_cfg.output_root}." if gd_cfg.output_root is not None else ""

    results_paths = reconstruct_staged_paths(len(gd_cfg.batch_stems), "res_input")
    logging.info(
        "Reading %d per-experiment results file(s): %s",
        len(results_paths),
        list(zip(gd_cfg.batch_stems, results_paths)),
    )
    batch_score_dfs = [pl.read_parquet(p) for p in results_paths]

    global_scores = global_variant_distinguishability(
        batch_score_dfs, gd_cfg.label_column
    )

    out_path = output_dir / f"{prefix}global_scores.parquet"
    logging.info("Writing %s", out_path)
    global_scores.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
