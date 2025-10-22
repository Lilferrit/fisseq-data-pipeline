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
from .combat import fit_harmonizer, harmonize
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
    Train pipeline parameters and run on a stratified train/test split.

    Validation Pipeline steps
    --------------
    1. Load dataset, derive feature/metadata frames, and clean invalid
       rows/columns.
    2. Build a stratification vector from ``_batch`` and ``_label`` and
       perform a single stratified train/test split.
    3. Fit a normalizer on the training split; transform train and test.
    4. Fit ComBat harmonizer on normalized training data; apply to the
       normalized test (and optionally train).
    5. Write unmodified, normalized, and harmonized Parquet outputs, and
       save fitted models.

    Parameters
    ----------
    input_data_path : PathLike
        Path to a Parquet file to scan and process.
    config : Config or PathLike, optional
        Configuration object or path. Must define feature columns and the
        names of ``_batch``, ``_label``, and ``_is_control`` fields.
    output_dir : PathLike, optional
        Output directory. Defaults to the current working directory.
    test_size : float, default=0.2
        Fraction of samples assigned to the test split.
    write_train_results : bool, default=True
        If True, also write the train split's unmodified/normalized/
        harmonized outputs.

    Outputs
    -------
    Written to ``output_dir``:

    - ``meta_data.test.parquet``
    - ``features.test.parquet``
    - ``normalized.test.parquet``
    - ``harmonized.test.parquet``
    - ``normalizer.pkl``
    - ``harmonizer.pkl``

    If ``write_train_results=True``:

    - ``meta_data.train.parquet``
    - ``features.train.parquet``
    - ``normalized.train.parquet``
    - ``harmonized.train.parquet``

    CLI
    ---
    Exposed via Fire at the ``fisseq-data-pipeline`` entry point, e.g.::

    ```bash
    fisseq-data-pipeline validate
        --input_data_path data.parquet
        --config config.yaml
        --output_dir out
        --test_size 0.2
        --write_train_results true
    ```
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

    logging.info("Writing feature matrices")
    test_meta_df.write_parquet(output_dir / "meta_data.test.parquet")
    test_feature_df.write_parquet(output_dir / "features.test.parquet")

    logging.info("Fitting normalizer on train data")
    normalizer = fit_normalizer(
        train_feature_df,
        meta_data_df=train_meta_df,
        fit_only_on_control=True,
    )

    logging.info("Running normalizer on train/test data")
    train_normalized_df = normalize(
        train_feature_df, normalizer, meta_data_df=train_meta_df
    )
    test_normalized_df = normalize(
        test_feature_df, normalizer, meta_data_df=test_meta_df
    )

    logging.info("Writing normalizer outputs")
    test_normalized_df.write_parquet(output_dir / "normalized.test.parquet")
    with open(output_dir / f"normalizer.pkl", "wb") as f:
        pickle.dump(normalizer, f)

    logging.info("Fitting harmonizer on train data")
    harmonizer = fit_harmonizer(
        train_normalized_df, train_meta_df, fit_only_on_control=True
    )

    logging.info("Harmonizing test data")
    test_harmonized_df = harmonize(test_normalized_df, test_meta_df, harmonizer)

    logging.info("Writing harmonization outputs")
    test_harmonized_df.write_parquet(output_dir / "harmonized.test.parquet")
    with open(output_dir / f"harmonizer.pkl", "wb") as f:
        pickle.dump(harmonizer, f)

    if write_train_results:
        logging.info("Writing train output up to the normalization stage")
        train_meta_df.write_parquet(output_dir / "meta_data.train.parquet")
        train_feature_df.write_parquet(output_dir / "features.train.parquet")
        train_normalized_df.write_parquet(output_dir / "normalized.train.parquet")

        logging.info("Harmonizing train data")
        train_harmonized_df = harmonize(train_normalized_df, train_meta_df, harmonizer)

        logging.info("Writing harmonized train data")
        train_harmonized_df.write_parquet(output_dir / "harmonized.train.parquet")


def run(*args, **kwargs) -> None:
    """
    Run the production pipeline on a full dataset.

    This function is a placeholder for a single-pass production run
    (no train/test split). It is not implemented yet.

    Raises
    ------
    NotImplementedError
        Always raised. The function body is not implemented.

    CLI
    ---
    Registered subcommand (placeholder)::

    ```bash
    fisseq-data-pipeline run
    ```
    """
    # TODO: implement run
    raise NotImplementedError()


def configure(output_path: Optional[PathLike] = None) -> None:
    """
    Write a copy of the default configuration to ``output_path``.

    Parameters
    ----------
    output_path : PathLike, optional
        Target path for the configuration file. If ``None``, writes
        ``config.yaml`` to the current working directory.

    Returns
    -------
    None

    CLI
    ---
    Exposed via Fire at the ``fisseq-data-pipeline`` entry point

    ```bash
    # Write config.yaml to CWD
    fisseq-data-pipeline configure

    # Write to a custom location
    fisseq-data-pipeline configure --output_path path/to/config.yaml
    ```
    """
    if output_path is None:
        output_path = pathlib.Path.cwd() / "config.yaml"

    shutil.copy(DEFAULT_CFG_PATH, output_path)


def main() -> None:
    """
    CLI entry that registers Fire subcommands.

    Subcommands
    -----------
    - ``validate``  : Train/validate on a stratified split and write outputs.
    - ``run``       : Production, single-pass run (not yet implemented).
    - ``configure`` : Write a default configuration file.

    CLI
    ---
    Invoked as the ``fisseq-data-pipeline`` console script. For example::

        fisseq-data-pipeline validate --input_data_path data.parquet
    """
    try:
        fire.Fire({"validate": validate, "run": run, "configure": configure})
    except:
        logging.exception("Run failed due to the following exception:")
        raise


if __name__ == "__main__":
    main()
