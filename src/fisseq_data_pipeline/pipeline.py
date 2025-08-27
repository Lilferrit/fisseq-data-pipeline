import datetime
import logging
import pathlib
import pickle
import shutil
from os import PathLike
from typing import Optional

import fire
import numpy as np
import polars as pl
import sklearn.model_selection

from .harmonize import fit_harmonizer, harmonize
from .normalize import fit_normalizer, normalize
from .utils import Config, get_rows_by_idx
from .utils.config import DEFAULT_CFG_PATH


def setup_logging(
    log_dir: Optional[PathLike] = None,
    to_file: bool = True,
) -> None:
    """Configure logging for the topdown-convert pipeline.

    If logging has already been configured, this function does nothing.

    Parameters
    ----------
    log_dir : PathLike, optional
        Directory where log files will be written. If None, uses the
        current working directory. Ignored if `to_file` is False.
    filename : str, optional
        Custom log filename. If None, defaults to
        ``top-down-convert-YYYYDDMM.log``.
    """
    if log_dir is None:
        log_dir = pathlib.Path.cwd()
    else:
        log_dir = pathlib.Path(log_dir)

    dt_str = datetime.datetime.now().strftime("%Y%m%d:%H%M%S")
    filename = f"fisseq-data-pipeline-{dt_str}.log"
    log_path = log_dir / filename
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path, mode="w")]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def validate(
    input_data_path: PathLike,
    config: Optional[Config | PathLike],
    output_dir: Optional[PathLike] = None,
    n_folds: int = 5,
) -> None:
    data_df = pl.scan_parquet(input_data_path)
    output_dir = pathlib.Path.cwd() if output_dir is None else pathlib.Path(output_dir)
    config = Config(config)

    skf = sklearn.model_selection.StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=42
    )

    # Mixing pandas and polars is a necessary evil here
    strata = (
        data_df.select(
            pl.col(config.batch_col_name),
            pl.col(config.label_col_name),
        )
        .collect()
        .to_pandas()
        .astype(str)
        .agg("_".join, axis=1)
    )

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(np.empty(len(strata)), strata), 1
    ):
        train_df = get_rows_by_idx(data_df, train_idx)
        test_df = get_rows_by_idx(data_df, test_idx)

        normalizer = fit_normalizer(train_df, config)
        normalized_data = normalize(test_df, config, normalizer)
        harmonizer = fit_harmonizer(normalized_data, config)
        harmonized_data = harmonize(normalized_data, config, harmonizer)

        test_df.sink_parquet(output_dir / f"unmodified.fold_{fold:05}.parquet")
        normalized_data.sink_parquet(output_dir / f"normalized.fold_{fold:05}.parquet")
        harmonized_data.sink_parquet(output_dir / f"harmonized.fold_{fold:05}.parquet")
        with open(output_dir / f"normalizer.fold_{fold:05}.pkl", "wb") as f:
            pickle.dump(normalizer, f)
        with open(output_dir / f"harmonizer.fold_{fold:05}.pkl", "wb") as f:
            pickle.dump(harmonizer, f)


def run(
    input_data_path: PathLike,
    config: Optional[Config | PathLike],
    output_dir: Optional[PathLike] = None,
) -> None:
    data_df = pl.scan_parquet(input_data_path)
    output_dir = pathlib.Path.cwd() if output_dir is None else pathlib.Path(output_dir)
    config = Config(config)

    normalizer = fit_normalizer(data_df, config)
    normalized_data = normalize(data_df, config, normalizer)
    harmonizer = fit_harmonizer(normalized_data, config)
    harmonized_data = harmonize(normalized_data, config, harmonizer)

    normalized_data.sink_parquet(output_dir / f"normalized.parquet")
    harmonized_data.sink_parquet(output_dir / f"harmonized.parquet")
    with open(output_dir / f"normalizer.pkl", "wb") as f:
        pickle.dump(normalizer, f)
    with open(output_dir / f"harmonizer.pkl", "wb") as f:
        pickle.dump(harmonizer, f)


def configure(output_path: Optional[PathLike] = None) -> None:
    if output_path is None:
        output_path = pathlib.Path.cwd() / "config.yaml"

    shutil.copy(DEFAULT_CFG_PATH, output_path)


def main() -> None:
    fire.Fire({"validate": validate, "run": run, "configure": configure})


if __name__ == "__main__":
    main()
