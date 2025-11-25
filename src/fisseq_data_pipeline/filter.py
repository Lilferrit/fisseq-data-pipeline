import logging
from typing import Callable, Iterable, TypeAlias

import polars as pl

from .utils import get_feature_cols

FilterFun: TypeAlias = Callable[[pl.LazyFrame], pl.LazyFrame]


def drop_cols_all_nonfinite(data_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Remove columns where *every* value is non-finite.

    A column is dropped if all of its elements are one of:
       - NaN
       - +inf
       - -inf

    This scan is performed eagerly for the minimal number of rows needed
    to determine which columns contain at least one finite value. The
    final returned LazyFrame excludes all-non-finite columns.

    Parameters
    ----------
    data_lf : pl.LazyFrame
        LazyFrame containing feature columns to be checked.

    Returns
    -------
    pl.LazyFrame
        A LazyFrame with all-non-finite columns removed.
    """
    # TODO: Can this also be a lazy expression?
    logging.info("Scanning for all non-finite rows")
    finite_summary = data_lf.select(
        [
            pl.col(c).is_finite().any().alias(c)
            for c in get_feature_cols(data_lf, as_string=True)
        ]
    ).collect()

    logging.info("Adding query to remove columns containing all non-finite values")
    non_finite_cols = [c for c in get_feature_cols(data_lf, as_string=True) if not finite_summary[c][0]]
    data_lf = data_lf.select(pl.exclude(non_finite_cols))
    logging.info(
        "Removing %d columns containing only non-finite values",
        len(non_finite_cols),
    )

    return data_lf


def drop_rows_any_nonfinite(data_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Remove rows containing any non-finite value in the feature columns.

    A row is dropped if *any* feature column contains:
       - NaN
       - +inf
       - -inf

    Parameters
    ----------
    data_lf : pl.LazyFrame
        LazyFrame containing numerical feature columns.

    Returns
    -------
    pl.LazyFrame
        A LazyFrame where all rows with any non-finite feature value have
        been filtered out.
    """
    logging.info(
        "Adding query to remove any remaining rows that contain non-finite"
        " feature values"
    )

    return data_lf.filter(
        pl.all_horizontal([c.is_finite() for c in get_feature_cols(data_lf)])
    )


def clean_data(
    feature_lf: pl.LazyFrame,
    stages: Iterable[str | FilterFun] = [
        "drop_cols_all_nonfinite",
        "drop_rows_any_nonfinite",
    ],
) -> pl.LazyFrame:
    """
    Apply a sequence of filtering stages to a LazyFrame.

    Each stage may be specified by:
      - A string matching a known stage name; or
      - A Callable of type ``FilterFun`` receiving a LazyFrame and returning one.

    Stages are executed in order, and each stage returns a transformed
    LazyFrame that becomes the input for the next stage.

    Parameters
    ----------
    feature_lf : pl.LazyFrame
        LazyFrame containing feature columns to clean.
    stages : Iterable[str | FilterFun], optional
        Ordered list of filter stages. The default pipeline is:
          - ``"drop_cols_all_nonfinite"``: remove columns that are entirely
            NaN/inf/-inf.
          - ``"drop_rows_any_nonfinite"``: remove rows containing any
            non-finite values.

    Returns
    -------
    pl.LazyFrame
        A LazyFrame representing the cleaned feature set.

    Notes
    -----
    - Invalid stage names are skipped with a warning.
    - Custom filtering functions can be inserted as callables.
    """
    stage_lookup: dict[str, FilterFun] = {
        "drop_cols_all_nonfinite": drop_cols_all_nonfinite,
        "drop_rows_any_nonfinite": drop_rows_any_nonfinite,
    }

    for stage in stages:
        if isinstance(stage, str):
            if stage not in stage_lookup:
                logging.warning("Skipping invalid filtering stage: %s", stage)
                continue
            stage = stage_lookup[stage]
        feature_lf = stage(feature_lf)

    return feature_lf
