import logging
import pathlib
import pickle
from os import PathLike

import fire
import fisseq_data_pipeline.normalize
import polars as pl


def just_normalize(
    data_path: PathLike,
    output_path: PathLike,
    fit_only_on_control: bool = True,
    fit_batch_wise: bool = True,
    normalize_train: bool = True,
) -> None:
    """
    Fit and apply a feature normalization model to FISSEQ training data.

    Parameters
    ----------
    data_path : PathLike
        Path to a directory containing the input Parquet files:
        - ``features.train.parquet``
        - ``features.test.parquet``
        - ``meta_data.train.parquet``
        - ``meta_data.test.parquet``

    output_path : PathLike
        Directory where the normalized Parquet files and the pickled
        normalizer object will be written.

    fit_only_on_control : bool, optional (default=True)
        If True, fit the normalizer using only control samples
        (as indicated in the metadata). Otherwise, fit on all training samples.

    fit_batch_wise : bool, optional (default=True)
        If True, fit normalization parameters separately for each batch.
        If False, compute global normalization statistics across all batches.

    normalize_train : bool, optional (default=True)
        If True, normalize both the test and training data. If False, only
        normalize the test data and save the fitted normalizer.

    Outputs
    -------
    Writes the following files to ``output_path``:
        - ``normalized.test.parquet`` : normalized test feature matrix
        - ``normalized.train.parquet`` : normalized train feature matrix
        - ``normalizer.pkl`` : pickled fitted normalizer object
    """
    logging.info("Loading data from %s", data_path)
    data_path = pathlib.Path(data_path)
    test_data = pl.read_parquet(data_path / "features.test.parquet")
    test_meta = pl.read_parquet(data_path / "meta_data.test.parquet")
    train_data = pl.read_parquet(data_path / "features.train.parquet")
    train_meta = pl.read_parquet(data_path / "meta_data.train.parquet")

    logging.info("Fitting normalizer")
    normalizer = fisseq_data_pipeline.normalize.fit_normalizer(
        train_data,
        meta_data_df=train_meta,
        fit_only_on_control=fit_only_on_control,
        fit_batch_wise=fit_batch_wise,
    )

    logging.info("Normalizing test data")
    test_normalize = fisseq_data_pipeline.normalize.normalize(
        test_data, normalizer, meta_data_df=test_meta
    )

    logging.info("Writing normalizer and normalized test data to %s", output_path)
    output_path = pathlib.Path(output_path)
    test_normalize.write_parquet(output_path / "normalized.test.parquet")
    with open(output_path / "normalizer.pkl", "wb") as f:
        pickle.dump(normalizer, f)

    if not normalize_train:
        return

    logging.info("Normalizing train data")
    train_normalize = fisseq_data_pipeline.normalize.normalize(
        train_data, normalizer, meta_data_df=train_meta
    )

    logging.info("Writing normalized train data to %s", str(output_path))
    train_normalize.write_parquet(output_path / "normalized.train.parquet")


def main() -> None:
    """Configure logging and CLI"""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] [%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fire.Fire(just_normalize)


if __name__ == "__main__":
    main()
