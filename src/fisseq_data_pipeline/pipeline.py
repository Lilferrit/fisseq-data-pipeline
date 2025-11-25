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
from .utils import Config, get_data_lf
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

    db_lf = get_db(input_data_path, eager_db_loading)
    output_dir = pathlib.Path.cwd() if output_dir is None else pathlib.Path(output_dir)
    logging.info("Output directory set to: %s", output_dir)

    logging.info("Collecting data matrices")
    config = Config(config)
    data_lf = get_data_lf(db_lf, config)

    logging.info("Cleaning data")
    data_lf = clean_data(data_lf)

    logging.info("Saving cleaned data")
    data_lf.sink_parquet(output_dir / "data-cleaned.parquet")

    logging.info("Fitting normalizer")
    normalizer = fit_normalizer(
        data_lf=data_lf,
        fit_batch_wise=True,
        fit_only_on_control=True,
    )

    logging.info("Saving normalizer")
    normalizer.save(output_dir / "normalizer.pkl")

    logging.info("Normalizing data")
    normalized_lf = normalize(data_lf, normalizer)

    logging.info("Saving normalized data")
    normalized_lf.sink_parquet(output_dir / "normalized.parquet")


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
        fire.Fire({"run": run, "configure": configure})
    except:
        logging.exception("Run failed due to the following exception:")
        raise


if __name__ == "__main__":
    main()
