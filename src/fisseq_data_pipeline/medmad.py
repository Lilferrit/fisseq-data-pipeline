import dataclasses
import logging
import pickle
from os import PathLike

import numpy as np
import polars as pl
from sklearn.preprocessing import Normalizer

from .utils import get_feature_cols


@dataclasses.dataclass
class MedMadNormalizer:
    """
    Container for per-feature robust normalization statistics (median / MAD).

    This object stores the statistics needed to apply *median/MAD normalization*
    (a robust alternative to mean/std standardization). Optionally, statistics
    can be computed *per batch* (using the ``_meta_batch`` column), enabling
    batch-wise centering/scaling.

    The normalized value for a feature ``c`` in batch ``b`` is:

        x_norm = (x - median_b[c]) / mad_b[c]

    where MAD is the median absolute deviation:

        mad_b[c] = median_b( |x - median_b[c]| )

    Attributes
    ----------
    medians : pl.DataFrame
        Per-batch per-feature medians. Shape is (n_batches, n_features + 1),
        where the extra column is ``_meta_batch``. If batch-wise fitting is
        disabled (``is_batch_wise=False``), statistics are computed globally
        and stored as a single pseudo-batch with ``_meta_batch == 0``.

    mads : pl.DataFrame
        Per-batch per-feature MADs. Same shape/keys as ``medians``.
        Features with MAD close to zero are dropped at fit time (see Notes).

    is_batch_wise : bool
        Whether the normalizer was fit batch-wise (using ``_meta_batch``) or
        globally (all samples treated as one batch).
    """

    medians: pl.DataFrame
    mads: pl.DataFrame
    is_batch_wise: bool

    def save(self, save_path: PathLike) -> None:
        """
        Save this fitted normalizer to disk via pickle.

        Parameters
        ----------
        save_path : PathLike
            Destination file path for the serialized object (commonly
            ``normalizer.pkl``).

        Notes
        -----
        - Reload with ``pickle.load(open(path, "rb"))``.
        - Pickles are Python-version dependent; do not rely on portability
          across major Python versions or untrusted environments.
        """
        with open(save_path, "wb") as f:
            pickle.dump(self, f)

    medians: pl.DataFrame
    mads: pl.DataFrame
    is_batch_wise: bool

    def save(self, save_path: PathLike) -> None:
        """
        Serialize and save the Normalizer object to disk using pickle.

        This method stores the fitted normalization statistics — per-feature
        means, standard deviations, and batch-wise configuration flag — as a
        single binary file that can later be reloaded to reproduce the same
        normalization behavior.

        Parameters
        ----------
        save_path : PathLike
            Destination file path for the serialized Normalizer object. The file
            is written in binary format (typically named ``normalizer.pkl``).

        Notes
        -----
        - The file can be reloaded using ``pickle.load(open(path, "rb"))``.
        - Only the Normalizer object and its attributes are serialized; any
        external references (e.g., LazyFrames) are not included.
        - The resulting file is Python-version dependent and not guaranteed
        to be portable across major interpreter versions.

        Examples
        --------
        >>> normalizer = fit_normalizer(feature_df, meta_data_df)
        >>> normalizer.save("output/normalizer.pkl")

        To reload later:
        >>> with open("output/normalizer.pkl", "rb") as f:
        ...     normalizer = pickle.load(f)
        """
        with open(save_path, "wb") as f:
            pickle.dump(self, f)


def fit_normalizer(
    data_lf: pl.LazyFrame,
    fit_only_on_control: bool = False,
    fit_batch_wise: bool = True,
) -> MedMadNormalizer:
    """
    Fit a median/MAD normalizer from a Polars LazyFrame.

    This computes per-feature medians and MADs either:
    - **Batch-wise** (grouped by ``_meta_batch``), or
    - **Globally** (all rows treated as one batch) if ``fit_batch_wise=False``.

    Optionally, the fit can be restricted to control samples via the boolean
    metadata column ``_meta_is_control``.

    Parameters
    ----------
    data_lf : pl.LazyFrame
        Input LazyFrame containing:
        - Feature columns (as identified by ``get_feature_cols``), and
        - Metadata columns:
          - ``_meta_batch`` (int or categorical-like), required if
            ``fit_batch_wise=True``.
          - ``_meta_is_control`` (bool), required if
            ``fit_only_on_control=True``.

    fit_only_on_control : bool, default=False
        If True, compute statistics using only rows where
        ``_meta_is_control`` is True. Normalization can still be applied to
        all rows later.

    fit_batch_wise : bool, default=True
        If True, compute statistics per batch using ``_meta_batch``.
        If False, ignore any existing batch values and fit a single global
        normalizer (implemented by setting ``_meta_batch`` to 0 for all rows).

    Returns
    -------
    MedMadNormalizer
        A fitted normalizer containing per-batch medians and MADs.

    Notes
    -----
    - Features with MAD <= ``np.finfo(np.float32).eps`` in *any* batch are
      dropped from both the stored medians and MADs. This prevents division by
      ~0 at transform time.
    - This function executes a collect to materialize MADs for the zero-MAD
      scan, and collects the medians before returning.
    """
    if fit_only_on_control:
        logging.info("Adding query to filter for control samples")
        data_lf = data_lf.filter(pl.col("_meta_is_control"))

    if not fit_batch_wise:
        data_lf = data_lf.with_columns(pl.lit(0).alias("_meta_batch"))

    logging.info("Adding queries to compute medians and MADs")
    feature_cols = get_feature_cols(data_lf, as_string=True)
    medians_lf = (
        data_lf.select(feature_cols + ["_meta_batch"]).group_by("_meta_batch").median()
    )
    data_lf = (
        data_lf.join(medians_lf, on="_meta_batch", how="left", suffix="_median")
        .with_columns([pl.col(c) - pl.col(f"{c}_median") for c in feature_cols])
        .with_columns([pl.col(c).abs().alias(f"{c}_mad") for c in feature_cols])
    )
    mads_lf = (
        data_lf.select([f"{c}_mad" for c in feature_cols] + ["_meta_batch"])
        .group_by("_meta_batch")
        .median()
        .rename({f"{c}_mad": c for c in feature_cols})
    )

    logging.info("Scanning for zero-MAD features to drop")
    mads_lf = mads_lf.collect()
    zero_mad_cols = [
        c for c in feature_cols if (mads_lf[c] <= np.finfo(np.float32).eps).any()
    ]

    if len(zero_mad_cols) != 0:
        logging.warning("Dropping %d zero-mad columns", len(zero_mad_cols))
        medians_lf = medians_lf.select(pl.exclude(zero_mad_cols))
        mads_lf = mads_lf.select(pl.exclude(zero_mad_cols))

    logging.info(
        "Fitting normalizer with %d features", len(feature_cols) - len(zero_mad_cols)
    )
    return MedMadNormalizer(
        medians=medians_lf.collect(),
        mads=mads_lf,
        is_batch_wise=fit_batch_wise,
    )


def normalize(data_lf: pl.LazyFrame, normalizer: MedMadNormalizer) -> pl.LazyFrame:
    """
    Apply a fitted median/MAD normalizer to a Polars LazyFrame.

    This performs robust standardization for each feature column ``c``:

        (c - median[c]) / mad[c]

    where the median/MAD are selected per batch if ``normalizer.is_batch_wise``
    is True (joined on ``_meta_batch``), or globally otherwise.

    Parameters
    ----------
    data_lf : pl.LazyFrame
        Input LazyFrame to normalize. Must contain feature columns (as
        determined by ``get_feature_cols``). If ``normalizer.is_batch_wise`` is
        True, must also contain ``_meta_batch`` (or it will join to null stats
        and produce null outputs).

    normalizer : MedMadNormalizer
        A fitted normalizer produced by :func:`fit_normalizer`.

    Returns
    -------
    pl.LazyFrame
        A LazyFrame with the same non-feature columns as input (minus any
        dropped feature columns), where feature columns have been normalized.

    Behavior
    --------
    - If the normalizer is global (``is_batch_wise=False``), a constant
      ``_meta_batch = 0`` column is added to the input so the join succeeds.
    - Feature columns present in ``data_lf`` but missing from the normalizer
      are dropped (with a warning). This commonly happens if you fit after
      dropping zero-MAD features, then later attempt to transform a dataset
      that still contains them.
    - Temporary join columns ``*_median`` and ``*_mad`` are removed before
      returning.

    Notes
    -----
    - This function assumes the MAD values in the normalizer are nonzero.
      (Zero-MAD features are removed during fitting.)
    - If there are batches in ``data_lf`` not seen during fitting, the join
      will produce null medians/MADs for those rows, resulting in null outputs
      for normalized features. If that's a concern, validate batch coverage
      before calling this function.

    """
    if not normalizer.is_batch_wise:
        data_lf = data_lf.with_columns(pl.lit(0).alias("_meta_batch"))

    logging.info("Scanning for features in data not present in normalizer")
    feature_cols = get_feature_cols(data_lf, as_string=True)
    norm_cols = get_feature_cols(normalizer.medians, as_string=True)
    bad_cols = set(feature_cols) - set(norm_cols)
    feature_cols = list(set(feature_cols) - bad_cols)
    data_lf = data_lf.select(pl.exclude(bad_cols))

    if len(bad_cols) > 0:
        logging.warning(
            "Dropped %d columns from feature_df not present in normalizer",
            len(bad_cols),
        )

    logging.info("Adding queries to apply normalization")
    data_lf = (
        data_lf.join(normalizer.medians, on="_meta_batch", how="left", suffix="_median")
        .join(normalizer.mads, on="_meta_batch", how="left", suffix="_mad")
        .with_columns(
            [
                (pl.col(c) - pl.col(f"{c}_median")) / pl.col(f"{c}_mad").alias(c)
                for c in feature_cols
            ]
        )
        .select(
            pl.exclude(
                [f"{c}_median" for c in feature_cols]
                + [f"{c}_mad" for c in feature_cols]
            )
        )
    )

    return data_lf
