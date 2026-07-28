"""Bootstrap pseudo-replicate split generation.

Hydra entry point backing the Nextflow process ``GENERATE_SPLIT``: splits cells
into stratified pseudo-replicate halves, used by the bootstrap feature-selection
pipeline (per-feature-type aggregation itself lives in
:mod:`.aggregatefeaturetype`, run as ``AGGREGATE_HALF``).
"""

import dataclasses
import logging
import pathlib

import hydra
import polars as pl
import sklearn.model_selection
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import LabeledInputConfig
from .utils.batches import load_batches
from .utils.log import setup_logging
from .utils.metadata import get_column
from .utils.splits import TMP_IDX_COL, add_row_index

_cs = ConfigStore.instance()


@dataclasses.dataclass
class GenerateSplitConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the pseudo-replicate split-generation
    entry point.

    Attributes
    ----------
    random_state : int
        Seed for the stratified 50/50
        :func:`sklearn.model_selection.train_test_split` by ``label_column``.
        In the Nextflow pipeline this is set directly to the bootstrap-loop
        index (``1..params.feature_select_bootstrap_reps``), so each bootstrap replicate gets a
        distinct, reproducible split. Required.
    """

    random_state: int = MISSING


_cs.store(name="generate_split_main", node=GenerateSplitConfig)


@hydra.main(version_base=None, config_path=None, config_name="generate_split_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: generate one pseudo-replicate 50/50 split.

    Loads ``input_file``, adds a row-index column
    (:func:`.utils.splits.add_row_index`), and performs a stratified (by
    ``label_column``) 50/50 :func:`sklearn.model_selection.train_test_split`
    at seed ``random_state``. Each half's row indices are written as a
    single-column (``TMP_IDX_COL``) parquet file, consumable by
    :func:`fisseq_data_pipeline.aggregatefeaturetype.main`'s ``index_file``
    option.

    Output files
    ------------
    - ``{output_dir}/half1.parquet``
    - ``{output_dir}/half2.parquet``

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.generatesplit \\
            output_dir=./out \\
            input_file=data/normalized.parquet \\
            random_state=3
    """
    split_cfg: GenerateSplitConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(split_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_cfg.output_dir = output_dir
    setup_logging(split_cfg, "generate_split")

    logging.info("Loading input from %s", split_cfg.input_file)
    lf, _ = load_batches(split_cfg.input_file)
    lf = add_row_index(lf)

    idx = get_column(lf, TMP_IDX_COL)
    labels = get_column(lf, split_cfg.label_column)

    half1_idx, half2_idx, _, _ = sklearn.model_selection.train_test_split(
        idx, labels, stratify=labels, random_state=split_cfg.random_state, test_size=0.5
    )

    logging.info(
        "Writing half1.parquet (%d rows) and half2.parquet (%d rows)",
        len(half1_idx),
        len(half2_idx),
    )
    pl.DataFrame({TMP_IDX_COL: half1_idx}).write_parquet(output_dir / "half1.parquet")
    pl.DataFrame({TMP_IDX_COL: half2_idx}).write_parquet(output_dir / "half2.parquet")

    logging.info("Done")


if __name__ == "__main__":
    main()
