import dataclasses
import logging
import pathlib
import pickle
import traceback

import hydra
import polars as pl
import xgboost as xgb
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import LabeledInputConfig
from .utils.batches import load_batches
from .utils.constants import FEATURE_SELECTOR, META_SELECTOR
from .utils.log import setup_logging
from .utils.xgbparams import get_dmatrix


@dataclasses.dataclass
class OvwtCellScoresConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the one-vs-wildtype cell-scores entry point.

    Extends :class:`.config.LabeledInputConfig` with parameters controlling
    model loading and batch processing.

    Attributes
    ----------
    models_path : str
        Path to the pickled ``dict[str, xgb.Booster]`` produced by the ovwt
        training step. Required.
    wt_label : str
        Label string identifying wildtype cells. Defaults to ``"WT"``.
    batch_size : int
        Number of rows to process per batch when iterating over the input
        LazyFrame. Defaults to ``10_000``.
    """

    models_path: str = MISSING
    wt_label: str = "WT"
    batch_size: int = 10_000


_cs = ConfigStore.instance()
_cs.store(name="ovwtcellscores_main", node=OvwtCellScoresConfig)

_SPLIT_INDEX_COLS = {"row_idx", "origin_file"}


def load_input(input_file: str) -> pl.LazyFrame:
    """
    Load feature data from either a full feature parquet or a split index file.

    Detects the file type by reading its schema. A split index file has exactly
    two columns — ``row_idx`` (int) and ``origin_file`` (str) — produced by the
    OvWT training step. Any other parquet is treated as a full feature file and
    loaded via :func:`.utils.load_batches`.

    Parameters
    ----------
    input_file : str
        Path to either a full feature parquet or a split index parquet.

    Returns
    -------
    pl.LazyFrame
        Lazy frame of feature rows, with meta columns intact.
    """
    probe = pl.read_parquet(input_file, n_rows=0)
    if _SPLIT_INDEX_COLS.issubset(probe.columns):
        index_df = pl.read_parquet(input_file)
        frames = []
        for (origin_file,), group in index_df.group_by("origin_file"):
            row_idx_set = set(group["row_idx"].to_list())
            frames.append(
                pl.scan_parquet(origin_file)
                .with_row_index("__row_idx__")
                .filter(pl.col("__row_idx__").is_in(row_idx_set))
                .drop("__row_idx__")
            )
        return pl.concat(frames)
    return load_batches(input_file)[0]


def get_cell_scores(data_lf: pl.LazyFrame, cfg: DictConfig) -> pl.DataFrame:
    """
    Score every cell in ``data_lf`` against each trained one-vs-wildtype model.

    For each batch of rows, builds an XGBoost DMatrix from the feature columns
    and runs ``model.predict`` for every variant model. Meta columns (those
    matching ``^meta_.*$``) are passed through unchanged. Results across all
    batches are concatenated before returning.

    Parameters
    ----------
    data_lf : pl.LazyFrame
        Input lazy frame containing feature columns, meta columns, and the
        label column specified by ``cfg.label_column``.
    cfg : DictConfig
        Hydra config supplying ``models_path``, ``label_column``, ``wt_label``,
        and ``batch_size``.

    Returns
    -------
    pl.DataFrame
        DataFrame with one column per variant key in the loaded models dict,
        plus all meta columns from the input. Row order matches the input.
    """
    logging.info("Loading models from %s", cfg.models_path)
    with open(cfg.models_path, "rb") as f:
        models: dict[str, xgb.Booster] = pickle.load(f)
    logging.info("Loaded %d model(s): %s", len(models), list(models.keys()))

    result_dfs = []
    for batch_idx, curr_df in enumerate(
        data_lf.collect_batches(chunk_size=cfg.batch_size)
    ):
        logging.info("Scoring batch %d (%d rows)", batch_idx, len(curr_df))
        curr_dict = {}
        dmatrix = get_dmatrix(
            curr_df.select(FEATURE_SELECTOR, cfg.label_column),
            cfg.label_column,
            cfg.wt_label,
        )

        for variant, model in models.items():
            curr_dict[variant] = model.predict(dmatrix)

        curr_scores_df = pl.from_dict(curr_dict)
        curr_meta_df = curr_df.select(META_SELECTOR)
        curr_combined_df = pl.concat((curr_scores_df, curr_meta_df), how="horizontal")
        result_dfs.append(curr_combined_df)

    result = pl.concat(result_dfs)
    logging.info(
        "Scored %d total rows across %d batch(es)", len(result), len(result_dfs)
    )
    return result


@hydra.main(version_base=None, config_path=None, config_name="ovwtcellscores_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: score cells against trained one-vs-wildtype models.

    Steps
    -----
    1. Load the feature data at ``cfg.input_file`` via :func:`load_input`, which
       accepts either a full feature parquet or a split index parquet produced
       by the OvWT training step.
    2. Call :func:`get_cell_scores` to predict per-variant scores for every cell.
    3. Write the result to ``{output_dir}/cell_scores.parquet``.

    Output files
    ------------
    - ``{output_dir}/cell_scores.parquet``

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.ovwtcellscores \\
            output_dir=./out \\
            input_file=data/features.parquet \\
            models_path=out/models.pkl
    """
    ovwtcellscores_cfg: OvwtCellScoresConfig = OmegaConf.to_object(cfg)
    output_dir = pathlib.Path(ovwtcellscores_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ovwtcellscores_cfg.output_dir = output_dir
    setup_logging(ovwtcellscores_cfg, "ovwtcellscores")

    logging.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    logging.info("Loading input from %s", cfg.input_file)
    data_lf = load_input(cfg.input_file)

    try:
        scores_df = get_cell_scores(data_lf, cfg)
    except Exception:
        logging.error("Failed to score cells:\n%s", traceback.format_exc())
        raise

    out_path = output_dir / "cell_scores.parquet"
    scores_df.write_parquet(out_path)
    logging.info("Cell scores written to %s", out_path)
    logging.info("Done")


if __name__ == "__main__":
    main()
