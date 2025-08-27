import datetime
import logging
import os
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


def setup_logging(log_dir: Optional[PathLike] = None) -> None:
    """
    Configure logging for the pipeline.

    A log file and a console stream are set up simultaneously.
    The log file is created in the specified directory (or the current
    working directory by default) with a timestamped filename.
    The log level is controlled by the environment variable
    ``FISSEQ_PIPELINE_LOG_LEVEL`` (default: ``"info"``).

    Parameters
    ----------
    log_dir : PathLike, optional
        Directory where log files will be written.
        If ``None``, the current working directory is used.
    """
    log_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    if log_dir is None:
        log_dir = pathlib.Path.cwd()
    else:
        log_dir = pathlib.Path(log_dir)

    dt_str = datetime.datetime.now().strftime("%Y%m%d:%H%M%S")
    filename = f"fisseq-data-pipeline-{dt_str}.log"
    log_path = log_dir / filename
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path, mode="w")]

    log_level = os.getenv("FISSEQ_PIPELINE_LOG_LEVEL", "info")
    log_level = log_levels.get(log_level, logging.INFO)

    logging.basicConfig(
        level=log_level,
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
    """
    Perform cross-validated normalization and harmonization on input data.

    The dataset is stratified by batch and label columns, split into
    ``n_folds`` folds, and each fold is processed to generate unmodified,
    normalized, and harmonized versions of the test set. Corresponding
    normalizer and harmonizer models are serialized to disk.

    Parameters
    ----------
    input_data_path : PathLike
        Path to the input Parquet file containing the dataset.
    config : Config or PathLike, optional
        Configuration object or path to a config file.
        Must specify at least ``batch_col_name`` and ``label_col_name``.
    output_dir : PathLike, optional
        Directory in which to write output files.
        Defaults to the current working directory.
    n_folds : int, default=5
        Number of stratified cross-validation folds to perform.

    Returns
    -------
    None
        Writes Parquet files and pickled model objects for each fold to
        ``output_dir``.
    """
    setup_logging(output_dir)
    logging.info("Starting validation with input path: %s", input_data_path)

    data_df = pl.scan_parquet(input_data_path)
    output_dir = pathlib.Path.cwd() if output_dir is None else pathlib.Path(output_dir)
    logging.info("Output directory set to: %s", output_dir)

    config = Config(config)
    skf = sklearn.model_selection.StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=42
    )
    logging.info("Initialized StratifiedKFold with %d folds", n_folds)

    # build strata
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
    logging.info("Strata built with %d samples", len(strata))

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(np.empty(len(strata)), strata), 1
    ):
        logging.info(
            "Processing fold %d: train size=%d, test size=%d",
            fold,
            len(train_idx),
            len(test_idx),
        )

        train_df = get_rows_by_idx(data_df, train_idx)
        test_df = get_rows_by_idx(data_df, test_idx)

        logging.debug("Fitting normalizer for fold %d", fold)
        normalizer = fit_normalizer(train_df, config)

        logging.debug("Normalizing test data for fold %d", fold)
        normalized_data = normalize(test_df, config, normalizer)

        logging.debug("Fitting harmonizer for fold %d", fold)
        harmonizer = fit_harmonizer(normalized_data, config)

        logging.debug("Harmonizing test data for fold %d", fold)
        harmonized_data = harmonize(normalized_data, config, harmonizer)

        # write outputs
        logging.info("Writing output files for fold %d", fold)
        test_df.sink_parquet(output_dir / f"unmodified.fold_{fold:05}.parquet")
        normalized_data.sink_parquet(output_dir / f"normalized.fold_{fold:05}.parquet")
        harmonized_data.sink_parquet(output_dir / f"harmonized.fold_{fold:05}.parquet")
        with open(output_dir / f"normalizer.fold_{fold:05}.pkl", "wb") as f:
            pickle.dump(normalizer, f)
        with open(output_dir / f"harmonizer.fold_{fold:05}.pkl", "wb") as f:
            pickle.dump(harmonizer, f)

    logging.info("Validation finished successfully with %d folds", n_folds)


def run(
    input_data_path: PathLike,
    config: Optional[Config | PathLike],
    output_dir: Optional[PathLike] = None,
) -> None:
    """
    Perform normalization and harmonization on the full dataset.

    The dataset is normalized using a fitted StandardScaler and harmonized
    using neuroHarmonize. Both transformed datasets and the fitted models
    are saved to disk.

    Parameters
    ----------
    input_data_path : PathLike
        Path to the input Parquet file containing the dataset.
    config : Config or PathLike, optional
        Configuration object or path to a config file.
        Must specify at least ``batch_col_name`` and ``feature_cols``.
    output_dir : PathLike, optional
        Directory in which to write output files.
        Defaults to the current working directory.

    Returns
    -------
    None
        Writes normalized and harmonized Parquet files, along with pickled
        normalizer and harmonizer models, to ``output_dir``.
    """
    setup_logging(output_dir)
    logging.info("Starting run with input path: %s", input_data_path)

    data_df = pl.scan_parquet(input_data_path)
    output_dir = pathlib.Path.cwd() if output_dir is None else pathlib.Path(output_dir)
    logging.info("Output directory set to: %s", output_dir)

    config = Config(config)
    logging.debug("Fitting normalizer on full dataset")
    normalizer = fit_normalizer(data_df, config)

    logging.debug("Normalizing dataset")
    normalized_data = normalize(data_df, config, normalizer)

    logging.debug("Fitting harmonizer on normalized dataset")
    harmonizer = fit_harmonizer(normalized_data, config)

    logging.debug("Harmonizing dataset")
    harmonized_data = harmonize(normalized_data, config, harmonizer)

    logging.info("Writing output files")
    normalized_data.sink_parquet(output_dir / "normalized.parquet")
    harmonized_data.sink_parquet(output_dir / "harmonized.parquet")
    with open(output_dir / "normalizer.pkl", "wb") as f:
        pickle.dump(normalizer, f)
    with open(output_dir / "harmonizer.pkl", "wb") as f:
        pickle.dump(harmonizer, f)

    logging.info("Run finished successfully")


def configure(output_path: Optional[PathLike] = None) -> None:
    """
    Create a copy of the default configuration file at the specified location.

    Parameters
    ----------
    output_path : PathLike, optional
        Path where the configuration file should be written.
        Defaults to ``config.yaml`` in the current working directory.

    Returns
    -------
    None
        Writes the default configuration YAML file to ``output_path``.
    """
    if output_path is None:
        output_path = pathlib.Path.cwd() / "config.yaml"

    shutil.copy(DEFAULT_CFG_PATH, output_path)


def main() -> None:
    """CLI Entry"""
    try:
        fire.Fire({"validate": validate, "run": run, "configure": configure})
    except:
        logging.exception("Run failed due to the following exception:")
        raise


if __name__ == "__main__":
    main()
