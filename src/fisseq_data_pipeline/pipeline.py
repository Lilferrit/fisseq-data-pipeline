import datetime
import logging
import os
import pathlib
import pickle
import shutil
from os import PathLike
from typing import Optional

import fire
import polars as pl

from .filter import clean_data, drop_infrequent_pairs
from .harmonize import fit_harmonizer, harmonize
from .normalize import fit_normalizer, normalize
from .utils import Config, get_data_dfs, train_test_split
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
    config: Optional[Config | PathLike] = None,
    output_dir: Optional[PathLike] = None,
    test_size: float = 0.2,
    write_train_results: bool = True,
) -> None:
    """
    Run a stratified train/test validation with normalization and harmonization.

    Pipeline:
      1) Load dataset, derive feature/metadata frames, and clean invalid
         rows/columns.
      2) Build a stratification vector from ``_batch`` and ``_label`` and
         perform a single stratified train/test split.
      3) Fit a normalizer on the training split; transform train and test.
      4) Fit a harmonizer (e.g., ComBat) on normalized training data; apply
         to normalized test (and optionally train).
      5) Write unmodified/normalized/harmonized Parquet outputs and save
         fitted models.

    Parameters
    ----------
    input_data_path : PathLike
        Path to a Parquet file to scan and process.
    config : Config or PathLike, optional
        Configuration object or path. Must define feature columns and the
        names of ``_batch``, ``_label``, and ``_is_control`` metadata fields.
    output_dir : PathLike, optional
        Directory for outputs. Defaults to the current working directory.
    test_size : float, default=0.2
        Fraction of samples assigned to the test split.
    write_train_results : bool, default=True
        If True, also write the train split's unmodified/normalized/harmonized
        outputs.

     Returns
    -------
    None

    Side Effects
    ------------
    Writes the following files into ``output_dir``:
      - ``meta_data.test.parquet``
      - ``features.test.parquet``
      - ``normalized.test.parquet``
      - ``harmonized.test.parquet``
      - ``normalizer.pkl`
      - ``harmonizer.pkl``
      - If ``write_train_results=True``:
        - ``meta_data.train.parquet``
        - ``features.train.parquet``
        - ``normalized.train.parquet``
        - ``harmonized.train.parquet``
    """
    setup_logging(output_dir)
    logging.info("Starting validation with input path: %s", input_data_path)

    data_df = pl.scan_parquet(input_data_path)
    output_dir = pathlib.Path.cwd() if output_dir is None else pathlib.Path(output_dir)
    logging.info("Output directory set to: %s", output_dir)

    logging.info("Collecting data matrices")
    config = Config(config)
    feature_df, meta_data_df = get_data_dfs(data_df, config)
    feature_df, meta_data_df = clean_data(feature_df, meta_data_df)
    feature_df, meta_data_df = drop_infrequent_pairs(feature_df, meta_data_df)
    train_feature_df, train_meta_df, test_feature_df, test_meta_df = train_test_split(
        feature_df, meta_data_df, test_size=test_size
    )

    logging.info("Fitting normalizer on train data")
    normalizer = fit_normalizer(
        train_feature_df,
        meta_data_df=train_meta_df,
        fit_only_on_control=True,
    )

    logging.info("Running normalizer on train/test data")
    train_normalized_df = normalize(train_feature_df, normalizer)
    test_normalized_df = normalize(test_feature_df, normalizer)

    logging.info("Fitting harmonizer on train data")
    harmonizer = fit_harmonizer(
        train_normalized_df, train_meta_df, fit_only_on_control=True
    )

    logging.info("Harmonizing test data")
    test_harmonized_df = harmonize(test_normalized_df, test_meta_df, harmonizer)

    # write outputs
    logging.info("Writing test outputs to %s", output_dir)
    test_meta_df.write_parquet(output_dir / "meta_data.test.parquet")
    test_feature_df.write_parquet(output_dir / "features.test.parquet")
    test_normalized_df.write_parquet(output_dir / "normalized.test.parquet")
    test_harmonized_df.write_parquet(output_dir / "harmonized.test.parquet")

    logging.info("Writing fitted parameters to %s", output_dir)
    with open(output_dir / f"normalizer.pkl", "wb") as f:
        pickle.dump(normalizer, f)
    with open(output_dir / f"harmonizer.pkl", "wb") as f:
        pickle.dump(harmonizer, f)

    if write_train_results:
        logging.info("Harmonizing train data")
        train_harmonized_df = harmonize(train_normalized_df, train_meta_df, harmonizer)
        train_meta_df.write_parquet(output_dir / "meta_data.train.parquet")
        train_feature_df.write_parquet(output_dir / "features.train.parquet")
        train_normalized_df.write_parquet(output_dir / "normalized.train.parquet")
        train_harmonized_df.write_parquet(output_dir / "harmonized.train.parquet")


def run(*args, **kwargs) -> None:
    # TODO: implement run
    raise NotImplementedError()


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
