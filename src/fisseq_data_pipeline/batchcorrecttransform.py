"""Apply a fitted two-pass centroid batch correction to a single batch.

Hydra entry point backing the Nextflow process ``BATCH_CORRECT_TRANSFORM``.
Loads a :class:`fisseq_data_pipeline.batchcorrect.BatchCorrector` fitted by
:mod:`.batchcorrect` and rescales one batch's cells to its variant's centroid
and finally to the wildtype centroid.
"""

import dataclasses
import logging
import pathlib

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .batchcorrect import BatchCorrector
from .config import LabeledInputConfig
from .utils.log import setup_logging


@dataclasses.dataclass
class BatchCorrectTransformConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the batch-correction transform entry point.

    Attributes
    ----------
    stats_file : str
        Path to the per-(variant, batch) statistics Parquet file written by
        :func:`fisseq_data_pipeline.batchcorrect.main`.
    centroids_file : str
        Path to the per-variant centroids Parquet file written by
        :func:`fisseq_data_pipeline.batchcorrect.main`.
    batch : str
        Label identifying which batch ``input_file`` belongs to. Passed
        explicitly rather than inferred from the filename, since batch files
        may share an identical name (e.g. every batch's QC-filtered output is
        named ``filtered_cells.parquet``).
    wt_label : str
        Value of ``label_column`` identifying wild-type rows. Defaults to
        ``"WT"``.
    """

    stats_file: str = MISSING
    centroids_file: str = MISSING
    batch: str = MISSING
    wt_label: str = "WT"


_cs = ConfigStore.instance()
_cs.store(name="batch_correct_transform_main", node=BatchCorrectTransformConfig)


@hydra.main(
    version_base=None, config_path=None, config_name="batch_correct_transform_main"
)
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: apply a fitted BatchCorrector to a single batch file.

    Reads the input file at ``input_file``, loads a
    :class:`fisseq_data_pipeline.batchcorrect.BatchCorrector` from
    ``stats_file``/``centroids_file``, applies it, and writes the corrected
    batch to its own output file.

    Output path
    -----------
    - If ``output_root`` is set: ``{output_root}.{stem}.{ext}``
    - Otherwise: ``{output_dir}/{filename}`` (same name as the input file)

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.batchcorrecttransform \\
            output_dir=./out \\
            input_file=data/batch1.parquet \\
            batch=batch1 \\
            stats_file=./fit/stats_vb.parquet \\
            centroids_file=./fit/centroids.parquet
    """
    trans_cfg: BatchCorrectTransformConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(trans_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trans_cfg.output_dir = output_dir
    setup_logging(trans_cfg, "batch_correct_transform")

    input_path = pathlib.Path(trans_cfg.input_file)
    logging.info("Loading input from %s", input_path)
    lf = pl.scan_parquet(input_path)

    logging.info("Loading batch corrector")
    corrector = BatchCorrector.load(
        trans_cfg.stats_file,
        trans_cfg.centroids_file,
        label_col=trans_cfg.label_column,
        wt_label=trans_cfg.wt_label,
    )

    logging.info("Applying batch correction for batch %r", trans_cfg.batch)
    lf = corrector.transform(lf, batch=trans_cfg.batch)

    stem = input_path.stem
    ext = input_path.suffix.lstrip(".")
    if trans_cfg.output_root is not None:
        out_path = pathlib.Path(f"{trans_cfg.output_root}.{stem}.{ext}")
    else:
        out_path = output_dir / input_path.name

    logging.info("Writing output to %s", out_path)
    lf.sink_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
