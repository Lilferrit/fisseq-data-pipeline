# aggregate.py
import logging
import pathlib
import re
from os import PathLike
from typing import Any, Optional

import fire
import joblib
import numpy as np
import polars as pl
import scipy.stats

from .normalize import Normalizer, fit_normalizer, normalize
from .utils import get_feature_cols, setup_logging


class EMDAggregator:
    """
    Compute per-group 1D Wasserstein distances (a.k.a. 1D EMD) to a cached
    reference distribution, typically built from control rows.

    This class can be used in two ways:
      1) As a Polars `pl.map_groups(...)` aggregation helper via `agg_emd(...)`,
         computing a single-feature EMD per group.
      2) As a standalone aggregator via `agg_df(...)`, computing all per-feature
         EMDs for each (`_meta_batch`, `_meta_label`) group using joblib.

    It precomputes (batch, feature) reference arrays once in `__init__` so that
    aggregation avoids repeatedly filtering/converting the control dataframe.

    Attributes
    ----------
    feature_cols : list[str]
        Feature columns over which EMDs will be computed.
    col_dict : dict
        Nested dictionary of cached reference distributions:
        `col_dict[batch][feature_name] -> np.ndarray` (1D float array).
        These arrays typically correspond to control values for each feature
        within each batch.
    """

    def __init__(self, aggregate_df: pl.DataFrame) -> None:
        """
        Build per-batch reference distributions for each feature.

        Parameters
        ----------
        aggregate_df : pl.DataFrame
            Dataframe containing the reference rows used to form the reference
            distributions (typically the control subset of a larger dataframe).

            Must contain:
              - `_meta_batch` column
            And the feature columns returned by `get_feature_cols(...)`.

        Notes
        -----
        This constructor converts each (batch, feature) column to a 1D NumPy
        array, which may allocate significant memory if you have many features
        or a large control set.
        """
        self.feature_cols = get_feature_cols(aggregate_df, as_string=True)
        self.col_dict: dict[Any, dict[str, np.ndarray]] = {}

        batches = aggregate_df.get_column("_meta_batch").unique().to_list()
        logging.info(
            "Initializing EMDAggregator from reference df: %d rows, %d batches, %d features",
            aggregate_df.height,
            len(batches),
            len(self.feature_cols),
        )

        for curr_batch in batches:
            curr_df = aggregate_df.filter(pl.col("_meta_batch") == curr_batch)
            self.col_dict[curr_batch] = {}

            for curr_feature in self.feature_cols:
                curr_col = curr_df.select(pl.col(curr_feature)).to_numpy().ravel()
                self.col_dict[curr_batch][curr_feature] = curr_col

        logging.info("Cached reference distributions successfully")

    def agg_emd(self, args: list[pl.Series]) -> pl.Series:
        """
        Polars group aggregation UDF that returns the Wasserstein distance
        between the group's feature distribution and the cached reference
        distribution for the corresponding batch.

        Parameters
        ----------
        args : list[pl.Series]
            A list of Polars Series passed by `pl.map_groups(exprs=[...])`.
            Expected structure:
              - args[0]: Series of feature values for the current group
              - args[1]: Series containing the batch identifier for the current
                         group (typically constant within the group)

        Returns
        -------
        pl.Series
            A length-1 Series containing the Wasserstein distance (float64)
            for this group/feature combination.
        """
        feature_col = args[0]
        batch_col = args[1]
        feature = feature_col.name
        batch = batch_col.first()

        variant_features = feature_col.to_numpy().ravel()
        reference_features = self.col_dict[batch][feature]
        emd = scipy.stats.wasserstein_distance(variant_features, reference_features)
        return pl.Series([emd], dtype=pl.Float64)

    def agg_df(
        self,
        agg_df: pl.DataFrame,
        n_jobs: int = -1,
        backend: str = "loky",
        verbose: int = 0,
    ) -> pl.DataFrame:
        """
        Aggregate EMD/Wasserstein distances by (batch, label) using joblib.

        For each group defined by (`_meta_batch`, `_meta_label`), this computes
        per-feature 1D Wasserstein distances between the group's feature values
        and the cached reference distribution for the same batch.

        This method:
          1) Enumerates all (batch, label) groups in `agg_df`.
          2) Materializes per-feature NumPy arrays for the group (variant) and
             the cached reference arrays for that batch.
          3) Runs the per-group EMD computation in parallel via joblib.

        Parameters
        ----------
        agg_df : pl.DataFrame
            The dataframe to aggregate. Must include:
              - `_meta_batch`
              - `_meta_label`
            And the feature columns in `self.feature_cols`.

        n_jobs : int, default=-1
            Number of workers for joblib. `-1` uses all available cores.

        backend : str, default="loky"
            Joblib backend. "loky" uses multiprocessing (process-based) and
            requires picklable task inputs. This implementation materializes
            NumPy arrays to satisfy that requirement.
            You may choose "threading" if you want threads instead.

        verbose : int, default=0
            Joblib verbosity level.

        Returns
        -------
        pl.DataFrame
            A dataframe with one row per (`_meta_batch`, `_meta_label`) and
            columns:
              - `_meta_batch`
              - `_meta_label`
              - `{feature}_EMD` for each feature in `self.feature_cols`
        """
        logging.info(
            "Building EMD tasks from df: %d rows, n_jobs=%s backend=%s",
            agg_df.height,
            str(n_jobs),
            backend,
        )

        tasks: list[tuple[Any, Any, dict[str, np.ndarray], dict[str, np.ndarray]]] = []
        batches = agg_df.get_column("_meta_batch").unique().to_list()
        logging.info("Found %d batches in input dataframe", len(batches))

        total_groups = 0
        for curr_batch in batches:
            curr_df = agg_df.filter(pl.col("_meta_batch") == curr_batch)

            # If a batch exists in agg_df but not in reference cache, that's a hard error
            if curr_batch not in self.col_dict:
                raise KeyError(
                    f"Batch {curr_batch!r} present in agg_df but missing from reference cache"
                )

            ref_for_batch = self.col_dict[curr_batch]

            # Reference arrays are identical for every label within a batch; build once per batch
            reference_arrays = {
                feat: np.asarray(ref_for_batch[feat]).ravel()
                for feat in self.feature_cols
            }

            labels = curr_df.get_column("_meta_label").unique().to_list()
            logging.info("Batch %r: %d labels", curr_batch, len(labels))

            for curr_label in labels:
                variant_df = curr_df.filter(pl.col("_meta_label") == curr_label)

                # materialize only what's needed (picklable)
                variant_arrays = {
                    feat: variant_df.select(feat).to_numpy().ravel()
                    for feat in self.feature_cols
                }

                tasks.append((curr_batch, curr_label, variant_arrays, reference_arrays))
                total_groups += 1

        logging.info(
            "Prepared %d EMD tasks (%d groups × %d features each)",
            total_groups,
            total_groups,
            len(self.feature_cols),
        )
        logging.info("Computing EMDs with joblib")

        dicts = joblib.Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
            joblib.delayed(_compute_one_label_arrays)(
                curr_batch=curr_batch,
                curr_label=curr_label,
                variant_arrays=variant_arrays,
                reference_arrays=reference_arrays,
            )
            for (curr_batch, curr_label, variant_arrays, reference_arrays) in tasks
        )

        logging.info("EMD computation complete; assembling output dataframe")
        out = pl.DataFrame(dicts)
        logging.info("EMD dataframe shape: %s rows × %s cols", out.height, out.width)
        return out


def _compute_one_label_arrays(
    *,
    curr_batch: Any,
    curr_label: Any,
    variant_arrays: dict[str, np.ndarray],
    reference_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """
    Compute per-feature 1D Wasserstein distances for a single (batch, label).

    This function is designed to be executed in parallel (e.g. via joblib).
    Inputs are simple Python objects / NumPy arrays to be easy to pickle when
    using process-based backends.

    Parameters
    ----------
    curr_batch : Any
        Batch identifier for this task.

    curr_label : Any
        Group label identifier for this task.

    variant_arrays : dict[str, np.ndarray]
        Mapping `{feature_name -> 1D array of values}` for the current group's
        feature distributions.

    reference_arrays : dict[str, np.ndarray]
        Mapping `{feature_name -> 1D array of reference values}` for the batch's
        reference distributions (typically controls).

    Returns
    -------
    dict[str, Any]
        A dictionary containing `_meta_batch`, `_meta_label`, and one key per
        feature of the form `{feature}_EMD`.
    """
    out: dict[str, Any] = {"_meta_label": curr_label, "_meta_batch": curr_batch}
    for feat, v in variant_arrays.items():
        r = reference_arrays[feat]
        out[f"{feat}_EMD"] = scipy.stats.wasserstein_distance(v, r)
    return out


def variant_classification(v: str) -> str:
    """Thanks Sriram"""
    if "fs" in v:
        classification = "Frameshift"
    elif v[-1] == "-":
        vs = v.split("|")
        ncodons_aff = len(vs)
        if ncodons_aff > 2:
            classification = "Other"
        else:
            if ncodons_aff == 1:
                classification = "3nt Deletion"
            elif ncodons_aff == 2:
                if int(vs[0][1:-1]) == (int(vs[1][1:-1]) - 1):
                    classification = "3nt Deletion"
                else:
                    classification = "Other"
            else:
                classification = "Other"
    elif ("X" in v) | ("*" in v):
        classification = "Nonsense"
    elif "WT" in v:
        classification = "WT"
    else:
        regex_match = re.match(r"([A-Z])(\d+)([A-Z])", v)
        if regex_match is None:
            classification = "Other"
        elif regex_match.group(1) == regex_match.group(3):
            classification = "Synonymous"
        else:
            classification = "Single Missense"

    return classification


def aggregate(
    norm_df: pl.DataFrame | PathLike,
    normalize_emds: bool = True,
    norm_only_to_synonymous: bool = False,
) -> tuple[pl.DataFrame, Optional[Normalizer]]:
    """
    Aggregate a normalized feature dataframe by (batch, label), computing both
    per-feature medians and per-feature distances-to-control (EMD/Wasserstein).

    This function performs two complementary per-group summaries over groups
    defined by (`_meta_batch`, `_meta_label`):

      1) **Location summary (median):**
         Computes `median(feature)` for each feature column within the group.

      2) **Distribution-shift summary (EMD/Wasserstein):**
         For each feature, computes the 1D Wasserstein distance between the
         distribution of values in the group and the cached reference
         distribution for the same batch. The reference distributions are
         built from rows where `_meta_is_control == True`.

      3) **Optional EMD normalization:**
         If `normalize_emds=True`, fits a batch-wise z-score normalizer on the
         aggregated EMD columns and returns normalized EMDs.

         If `norm_only_to_synonymous=True`, the EMD normalizer is fit **only**
         on groups whose `_meta_label` classifies as "Synonymous" via
         `variant_classification(...)`. In this mode, a synthetic
         `_meta_is_control` column is created on the aggregated EMD dataframe
         to mark "Synonymous" groups as controls for fitting.

    Parameters
    ----------
    norm_df : pl.DataFrame | PathLike
        Either an in-memory Polars DataFrame or a path to a parquet file.

        The dataframe must include:
          - `_meta_batch` (batch identifier)
          - `_meta_label` (group label identifier)
          - `_meta_is_control` (boolean control indicator used to build the
            per-batch reference distributions)
          - feature columns returned by `get_feature_cols(...)`

    normalize_emds : bool, default=True
        If True, fit a batch-wise normalizer on the aggregated EMD columns and
        return normalized EMDs along with the fitted normalizer. If False,
        EMD columns are returned unnormalized and the returned normalizer is
        `None`.

    norm_only_to_synonymous : bool, default=False
        If True, fit the EMD normalizer only on aggregated groups classified as
        "Synonymous" by `variant_classification(_meta_label)`. This is useful
        when you want the EMD z-scoring baseline to be defined by synonymous
        variants rather than all groups.

        Notes:
        - This option has an effect only when `normalize_emds=True`.
        - It does *not* change which rows are used to build the EMD reference
          distributions; those always come from the input rows where
          `_meta_is_control == True`.

    Returns
    -------
    tuple[pl.DataFrame, Optional[Normalizer]]
        A tuple `(agg_df, normalizer)`.

        - `agg_df` contains one row per (`_meta_batch`, `_meta_label`) group and
            - per-feature medians (native Polars group-by median)
            - per-feature EMD columns named `{feature}_EMD` (raw or normalized)

        - `normalizer` is:
            - `None` if `normalize_emds=False`
            - a fitted `Normalizer` if `normalize_emds=True`
              (possibly fit only on synonymous groups if
               `norm_only_to_synonymous=True`)
    """
    if not isinstance(norm_df, pl.DataFrame):
        logging.info(
            "Loading normalized feature dataframe from parquet: %s", str(norm_df)
        )
        norm_df = pl.read_parquet(norm_df)
    else:
        logging.info(
            "Using in-memory normalized feature dataframe: %d rows × %d cols",
            norm_df.height,
            norm_df.width,
        )

    required_meta = {"_meta_batch", "_meta_label", "_meta_is_control"}
    missing = required_meta - set(norm_df.columns)
    if missing:
        raise ValueError(
            f"norm_df missing required metadata columns: {sorted(missing)}"
        )

    logging.info("Selecting control rows for EMD reference cache")
    control_df = norm_df.filter(pl.col("_meta_is_control"))
    logging.info("Control df: %d rows", control_df.height)

    logging.info("Initializing EMD aggregator (reference cache)")
    aggregator = EMDAggregator(control_df)

    feature_cols = get_feature_cols(norm_df, as_string=True)
    meta_cols = ["_meta_label", "_meta_batch"]
    logging.info(
        "Aggregating medians for %d features over %d meta columns",
        len(feature_cols),
        len(meta_cols),
    )

    median_df = norm_df.select(meta_cols + feature_cols).group_by(meta_cols).median()
    logging.info(
        "Median dataframe shape: %d rows × %d cols", median_df.height, median_df.width
    )

    logging.info("Computing EMD dataframe")
    emd_df = aggregator.agg_df(norm_df)
    logging.info("EMD dataframe shape: %d rows × %d cols", emd_df.height, emd_df.width)

    if not normalize_emds:
        logging.info("Joining median + raw EMD dataframes")
        agg_df = median_df.join(emd_df, on=meta_cols)
        logging.info(
            "Aggregated dataframe shape: %d rows × %d cols", agg_df.height, agg_df.width
        )
        return agg_df, None

    if norm_only_to_synonymous:
        emd_df = emd_df.with_columns(
            (
                pl.col("_meta_label").map_elements(
                    variant_classification, return_dtype=pl.Utf8
                )
                == "Synonymous"
            ).alias("_meta_is_control")
        )

    logging.info("Fitting EMD normalizer (batch-wise)")
    emd_lf = emd_df.lazy()
    normalizer = fit_normalizer(
        emd_lf, fit_batch_wise=True, fit_only_on_control=norm_only_to_synonymous
    )

    logging.info("Normalizing EMD dataframe")
    emd_df = normalize(emd_lf, normalizer).collect()

    logging.info("Joining median + normalized EMD dataframes")
    agg_df = median_df.join(emd_df, on=meta_cols)
    logging.info(
        "Aggregated dataframe shape: %d rows × %d cols", agg_df.height, agg_df.width
    )

    logging.info("Aggregation complete")
    return agg_df, normalizer


def cli_wrapper(
    norm_df: pl.DataFrame | PathLike,
    out_dir: PathLike,
    normalize_emds: bool = True,
    norm_only_to_synonymous: bool = False,
) -> None:
    """
    Command-line entrypoint for aggregating a normalized dataset.

    This wrapper:
      - Configures logging (via `setup_logging(out_dir)`)
      - Runs `aggregate(...)`
      - Writes the aggregated dataframe to `<out_dir>/aggregated.parquet`
      - If EMD normalization is enabled, also writes the fitted EMD normalizer
        to `<out_dir>/emd_normalizer.pkl`

    Parameters
    ----------
    norm_df : pl.DataFrame | PathLike
        Input normalized dataset (DataFrame or parquet path). See `aggregate`
        for required columns and semantics.

    out_dir : PathLike
        Output directory to write results into. The directory is created if it
        does not already exist.

    normalize_emds : bool, default=True
        Whether to normalize EMD columns and save the fitted EMD normalizer.

    norm_only_to_synonymous : bool, default=False
        If True, fit the EMD normalizer only on groups classified as
        "Synonymous" (see `aggregate(...)`). Only relevant when
        `normalize_emds=True`.
    """
    setup_logging(out_dir)
    logging.info(
        "Starting aggregation CLI: norm_df=%s out_dir=%s normalize_emds=%s",
        str(norm_df),
        str(out_dir),
        str(normalize_emds),
    )

    agg_df, normalizer = aggregate(
        norm_df,
        normalize_emds=normalize_emds,
        norm_only_to_synonymous=norm_only_to_synonymous,
    )

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "aggregated.parquet"
    logging.info("Writing aggregated dataframe to %s", str(out_path))
    agg_df.write_parquet(out_path)

    if normalizer is None:
        logging.info("No normalizer returned (normalize_emds=False); done")
        return

    norm_path = out_dir / "emd_normalizer.pkl"
    logging.info("Writing EMD normalizer to %s", str(norm_path))
    normalizer.save(norm_path)
    logging.info("CLI finished successfully")


if __name__ == "__main__":
    """CLI entry"""
    fire.Fire(cli_wrapper)
