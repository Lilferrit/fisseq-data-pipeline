import datetime
import logging
import os
import pathlib
import shutil
from os import PathLike
from typing import Optional

import fire
import polars as pl

from .filter import clean_data
from .normalize import fit_normalizer, normalize
from .utils import Config, get_data_dfs, train_test_split
from .utils.config import DEFAULT_CFG_PATH


def get_db(input_data_path: PathLike, eager: bool) -> pl.LazyFrame:
    """
    Load or scan a Parquet database as a Polars LazyFrame.

    Depending on the ``eager`` flag, this function either reads the entire
    Parquet file into memory eagerly or sets up a lazy scan that defers I/O
    until query execution. The result is always returned as a
    :class:`polars.LazyFrame` for downstream pipeline compatibility.

    Parameters
    ----------
    input_data_path : PathLike
        Path to the Parquet file containing the dataset to be processed.
    eager : bool
        If True, fully load the dataset into memory using
        :func:`polars.read_parquet` and immediately convert it to a lazy
        frame. This avoids repeated on-disk scans and can improve runtime on
        systems with sufficient memory.
        If False, construct a lazy, on-demand scan using
        :func:`polars.scan_parquet`, which minimizes memory usage but may be
        slower due to disk I/O during execution.

    Returns
    -------
    polars.LazyFrame
        A lazy Polars representation of the dataset, ready for further
        transformations.

    Notes
    -----
    - Eager loading is advantageous for repeated access or complex queries
      over a moderate-sized dataset that fits comfortably in RAM.
    - Lazy scanning is preferred for very large datasets or when operating
      in a memory-constrained environment.

    Examples
    --------
    >>> lf = get_db("features.parquet", eager=False)
    >>> lf = get_db("features.parquet", eager=True)
    """
    if eager:
        logging.info("Eagerly loading database: %s", input_data_path)
        data_df = pl.read_parquet(input_data_path).lazy()
    else:
        logging.info("Scanning database: %s", input_data_path)
        data_df = pl.scan_parquet(input_data_path)

    logging.info("Finished %s database.", "loading" if eager else "scanning")
    return data_df


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
    eager_db_loading: bool = False,
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
    eager_db_loading : bool, default=False
        If True, fully load the input Parquet file into memory eagerly
        using :func:`polars.read_parquet`. This avoids repeated on-disk
        scans and can significantly speed up processing on systems with
        sufficient RAM. If False (default), the dataset is accessed lazily
        using :func:`polars.scan_parquet`, which minimizes memory usage but
        may incur slower disk I/O during computation.

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

    data_df = get_db(input_data_path, eager_db_loading)
    output_dir = pathlib.Path.cwd() if output_dir is None else pathlib.Path(output_dir)
    logging.info("Output directory set to: %s", output_dir)

    logging.info("Collecting data matrices")
    config = Config(config)
    feature_df, meta_data_df = get_data_dfs(data_df, config)
    feature_df, meta_data_df = clean_data(feature_df, meta_data_df)
    train_feature_df, train_meta_df, test_feature_df, test_meta_df = train_test_split(
        feature_df, meta_data_df, test_size=test_size
    )

    logging.info("Writing feature matrices")
    test_meta_df.sink_parquet(output_dir / "meta_data.test.parquet")
    test_feature_df.sink_parquet(output_dir / "features.test.parquet")

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
    test_normalized_df.sink_parquet(output_dir / "normalized.test.parquet")
    normalizer.save(output_dir / "normalizer.pkl")

    if write_train_results:
        logging.info("Writing train output up to the normalization stage")
        train_meta_df.sink_parquet(output_dir / "meta_data.train.parquet")
        train_feature_df.sink_parquet(output_dir / "features.train.parquet")
        train_normalized_df.sink_parquet(output_dir / "normalized.train.parquet")


def run(
    input_data_path: PathLike,
    config: Optional[Config | PathLike] = None,
    output_dir: Optional[PathLike] = None,
    eager_db_loading: bool = False,
) -> None:
    """
    Run the full batch-correction pipeline.

    This function performs a one-pass processing workflow that reads a full
    dataset, cleans invalid rows and columns, fits normalization statistics
    (optionally batch-wise and on control samples), applies normalization, and
    writes the resulting cleaned and normalized outputs to disk.

    Production Pipeline steps
    -------------------------
    1. Load and scan the input Parquet dataset into a Polars LazyFrame.
    2. Derive feature and metadata frames using configuration-specified
       column selections.
    3. Clean the dataset by removing:
         - Columns that contain only non-finite (NaN/inf) values.
         - Rows containing any non-finite feature values.
    4. Fit a batch-wise normalizer (computed from control samples only).
    5. Apply normalization to the full cleaned dataset.
    6. Write the cleaned, normalized, and fitted model artifacts to disk.

    Parameters
    ----------
    input_data_path : PathLike
        Path to the input Parquet file containing the full dataset to process.
    config : Config or PathLike, optional
        Path to configuration
    output_dir : PathLike, optional
        Directory to which cleaned and normalized outputs will be written.
        Defaults to the current working directory if not specified.
    eager_db_loading : bool, default=False
        If True, fully load the input Parquet file into memory eagerly
        using :func:`polars.read_parquet`. This avoids repeated on-disk
        scans and can significantly speed up processing on systems with
        sufficient RAM. If False (default), the dataset is accessed lazily
        using :func:`polars.scan_parquet`, which minimizes memory usage but
        may incur slower disk I/O during computation.

    Outputs
    -------
    Written to ``output_dir``:

    - ``meta_data.parquet`` — cleaned metadata table.
    - ``features.parquet`` — cleaned feature matrix.
    - ``normalized.parquet`` — z-score normalized feature matrix.
    - ``normalizer.pkl`` — serialized :class:`Normalizer` object containing
      per-feature mean and standard deviation statistics.

    CLI
    ---
    Exposed via Fire at the ``fisseq-data-pipeline`` entry point, e.g.::

    ```bash
    fisseq-data-pipeline run
        --input_data_path data.parquet
        --config config.yaml
        --output_dir out
    ```
    """
    setup_logging(output_dir)
    logging.info("Starting validation with input path: %s", input_data_path)

    data_df = get_db(input_data_path, eager_db_loading)
    output_dir = pathlib.Path.cwd() if output_dir is None else pathlib.Path(output_dir)
    logging.info("Output directory set to: %s", output_dir)

    logging.info("Collecting data matrices")
    config = Config(config)
    feature_df, meta_data_df = get_data_dfs(data_df, config)

    logging.info("Cleaning data")
    feature_df, meta_data_df = clean_data(
        feature_df,
        meta_data_df,
        stages=[
            "drop_cols_all_nonfinite",
            "drop_rows_any_nonfinite",
        ],
    )

    logging.info("Saving cleaned data")
    feature_df.sink_parquet(output_dir / "meta_data.parquet")
    meta_data_df.sink_parquet(output_dir / "features.parquet")

    logging.info("Fitting normalizer")
    normalizer = fit_normalizer(
        feature_df,
        meta_data_df=meta_data_df,
        fit_batch_wise=True,
        fit_only_on_control=True,
    )

    logging.info("Saving normalizer")
    normalizer.save(output_dir / "normalizer.pkl")

    logging.info("Normalizing data")
    normalized_df = normalize(feature_df, normalizer, meta_data_df=meta_data_df)

    logging.info("Saving normalized data")
    normalized_df.sink_parquet(output_dir / "normalized.parquet")


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
