import datetime
import logging
import os
import pathlib
import pickle
import shutil
from os import PathLike
from typing import List, Optional, Tuple

import fire
import numpy as np
import polars as pl
import sklearn.model_selection

from .filter import get_clean_masks
from .harmonize import fit_harmonizer, harmonize
from .normalize import fit_normalizer, normalize
from .utils import Config, get_data_dfs, set_feature_matrix
from .utils.config import DEFAULT_CFG_PATH

RANDOM_STATE = os.getenv("FISSEQ_PIPELINE_RAND_STATE", 42)


def _setup_logging(log_dir: Optional[PathLike] = None) -> None:
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


def _clean(
    feature_df: pl.DataFrame, meta_data_df: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Remove unusable rows/columns from features and align metadata.

    This function computes masks for:
      - columns that are entirely NaN,
      - columns with (near) zero variance, and
      - rows containing NaNs in any non-all-NaN column.

    It then drops flagged rows/columns from the feature matrix and applies the
    same row filtering to the metadata so the two remain aligned.

    Parameters
    ----------
    feature_df : pl.DataFrame
        Feature columns only; shape (n_samples, n_features).
    meta_data_df : pl.DataFrame
        Metadata columns (e.g., _batch, _label, _is_control), aligned by rows
        with `feature_df`.

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame]
        Cleaned `(feature_df, meta_data_df)` with invalid rows/columns removed
        and row alignment preserved.
    """
    logging.info("Cleaning data")
    feature_matrix = feature_df.to_numpy()
    col_all_nan, col_zero_var, row_contains_nan = get_clean_masks(feature_matrix)
    row_mask = ~row_contains_nan
    col_mask = ~col_all_nan & ~col_zero_var

    # Columns to keep (preserve order)
    keep_cols = [c for c, keep in zip(feature_df.columns, col_mask) if keep]

    # Apply masks (rows via filter; columns via select)
    feature_df = feature_df.filter(pl.Series(row_mask)).select(keep_cols)
    meta_data_df = meta_data_df.filter(pl.Series(row_mask))

    return feature_df, meta_data_df


def _get_train_test(
    feature_df: pl.DataFrame,
    meta_data_df: pl.DataFrame,
    test_size: float,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Stratify by batch+label and split features/metadata into train/test sets.

    A 1D stratification vector is built by concatenating ``_batch`` and
    ``_label``; indices are split with sklearn's `train_test_split`, then the
    corresponding rows are selected from both the feature and metadata frames.

    Parameters
    ----------
    feature_df : pl.DataFrame
        Cleaned feature frame; shape (n_samples, n_features).
    meta_data_df : pl.DataFrame
        Cleaned metadata frame aligned to `feature_df`.
    test_size : float
        Proportion of samples to assign to the test split.

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]
        train_feature_df, train_meta_data_df, test_feature_df, test_meta_data_df
    """
    stratify = meta_data_df.select(
        pl.concat_str(
            (pl.col("_batch").cast(pl.Utf8), pl.col("_label").cast(pl.Utf8)),
            separator="_",
        ).alias("_stratify")
    )["_stratify"].to_list()

    train_idx, test_idx = sklearn.model_selection.train_test_split(
        np.arange(len(meta_data_df)),
        test_size=test_size,
        stratify=stratify,
        random_state=RANDOM_STATE,
    )

    train_feature_df = feature_df[train_idx, :]
    train_meta_data_df = meta_data_df[train_idx, :]
    test_feature_df = feature_df[test_idx, :]
    test_meta_data_df = meta_data_df[test_idx, :]

    return train_feature_df, train_meta_data_df, test_feature_df, test_meta_data_df


def _write_output(
    unmodified_matrix: np.ndarray,
    normalized_matrix: np.ndarray,
    harmonized_matrix: np.ndarray,
    meta_data: pl.LazyFrame,
    feature_cols: List[str],
    output_dir: pathlib.Path,
    split_name: str,
) -> None:
    """
    Write unmodified, normalized, and harmonized splits to Parquet files.

    Uses `set_feature_matrix` to pair a feature matrix with the provided
    metadata (as a LazyFrame) and writes three Parquet files named with the
    given split suffix.

    Parameters
    ----------
    unmodified_matrix : np.ndarray
        Raw features for the split; shape (n_split, n_features_kept).
    normalized_matrix : np.ndarray
        Normalized features for the split; same shape as `unmodified_matrix`.
    harmonized_matrix : np.ndarray
        Harmonized features for the split; same shape as above.
    meta_data : pl.LazyFrame
        Metadata rows corresponding to the split.
    feature_cols : list[str]
        Column names matching the feature matrices' column order.
    output_dir : pathlib.Path
        Destination directory; must exist prior to writing.
    split_name : str
        Suffix used in filenames (e.g., ``"train"`` or ``"test"``).

    Returns
    -------
    None
        Writes:
          - ``unmodified.<split_name>.parquet``
          - ``normalized.<split_name>.parquet``
          - ``harmonized.<split_name>.parquet``
    """
    set_feature_matrix(
        meta_data,
        feature_cols,
        unmodified_matrix,
    ).sink_parquet(output_dir / f"unmodified.{split_name}.parquet")

    set_feature_matrix(
        meta_data,
        feature_cols,
        normalized_matrix,
    ).sink_parquet(output_dir / f"normalized.{split_name}.parquet")

    set_feature_matrix(
        meta_data,
        feature_cols,
        harmonized_matrix,
    ).sink_parquet(output_dir / f"harmonized.{split_name}.parquet")


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
        Writes Parquet files and model artifacts to ``output_dir``:
          - ``unmodified.test.parquet``
          - ``normalized.test.parquet``
          - ``harmonized.test.parquet``
          - (optionally) the corresponding ``*.train.parquet`` files
          - ``normalizer.test.pkl``
          - ``harmonizer.test.pkl``
    """
    _setup_logging(output_dir)
    logging.info("Starting validation with input path: %s", input_data_path)

    data_df = pl.scan_parquet(input_data_path)
    output_dir = pathlib.Path.cwd() if output_dir is None else pathlib.Path(output_dir)
    logging.info("Output directory set to: %s", output_dir)

    logging.info("Collecting data matrices")
    config = Config(config)
    feature_df, meta_data_df = get_data_dfs(data_df, config)
    feature_df, meta_data_df = _clean(feature_df, meta_data_df)
    train_feature_df, train_meta_data, test_feature_df, test_meta_data = (
        _get_train_test(feature_df, meta_data_df, test_size)
    )

    logging.info("Fitting normalizer on train data")
    train_matrix = train_feature_df.to_numpy()
    test_matrix = test_feature_df.to_numpy()
    normalizer = fit_normalizer(
        train_matrix, config, is_control=train_meta_data["_is_control"].to_numpy()
    )

    logging.info("Running normalizer on train/test data")
    normalized_train = normalize(train_matrix, normalizer)
    normalized_test = normalize(test_matrix, normalizer)

    logging.info("Fitting harmonizer on train data")
    harmonizer = fit_harmonizer(
        normalized_train,
        train_meta_data["_batch"].to_numpy(),
        config,
        is_control=train_meta_data["_is_control"].to_numpy(),
    )

    logging.info("Harmonizing test data")
    harmonized_test = harmonize(
        normalized_test, test_meta_data["_batch"].to_numpy(), config, harmonizer
    )

    harmonized_train = None
    if write_train_results:
        harmonized_train = harmonize(
            normalized_train, test_meta_data["_batch"].to_numpy(), config, harmonizer
        )

    # write outputs
    logging.info("Writing outputs to %s", output_dir)
    test_meta_data = test_meta_data.lazy()
    feature_cols = feature_df.columns

    _write_output(
        unmodified_matrix=test_matrix,
        normalized_matrix=normalized_test,
        harmonized_matrix=harmonized_test,
        meta_data=test_meta_data,
        feature_cols=feature_cols,
        output_dir=output_dir,
        split_name="test",
    )

    if write_train_results:
        _write_output(
            unmodified_matrix=train_matrix,
            normalized_matrix=normalized_train,
            harmonized_matrix=harmonized_train,
            meta_data=train_meta_data,
            feature_cols=feature_cols,
            output_dir=output_dir,
            split_name="train",
        )

    with open(output_dir / f"normalizer.test.pkl", "wb") as f:
        pickle.dump(normalizer, f)
    with open(output_dir / f"harmonizer.test.pkl", "wb") as f:
        pickle.dump(harmonizer, f)

    logging.info("Validation finished successfully")


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
