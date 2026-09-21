"""Row-index-based filtering helpers for the bootstrap feature-selection pipeline.

Defines the ``TMP_IDX_COL`` row-index convention and :func:`filter_by_index_file`,
used to restrict a cell-level LazyFrame to one pseudo-replicate half written by
:func:`fisseq_data_pipeline.generatesplit.main`.
"""

import os
from typing import Optional, Union

import polars as pl

TMP_IDX_COL = "tmp_cell_idx"


def add_row_index(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add a ``TMP_IDX_COL`` integer row-index column to a LazyFrame, in scan order.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input LazyFrame.

    Returns
    -------
    pl.LazyFrame
        The input frame with an additional ``TMP_IDX_COL`` integer column.
    """
    return lf.with_columns(pl.row_index(TMP_IDX_COL))


def get_replicate_lf(
    lf: pl.LazyFrame, rep_idx: Union[list[int], pl.Series]
) -> pl.LazyFrame:
    """
    Filter a LazyFrame to rows belonging to one pseudo-replicate.

    Parameters
    ----------
    lf : pl.LazyFrame
        Cell-level LazyFrame that must already contain a ``TMP_IDX_COL``
        integer column (added by :func:`add_row_index`).
    rep_idx : list[int] or pl.Series
        Row indices that belong to this replicate half.

    Returns
    -------
    pl.LazyFrame
        Subset of ``lf`` containing only the rows whose ``TMP_IDX_COL`` value
        is in ``rep_idx``.

    Notes
    -----
    ``rep_idx`` is kept as a ``pl.Series`` rather than being boxed into a
    Python ``set`` first. ``BaseAggregator.aggregate`` re-applies this
    predicate once per feature chunk, and on a real half-split ``rep_idx``
    holds hundreds of thousands of indices -- materializing those as Python
    ints on every chunk is pure overhead.

    ``.implode()`` is required, not cosmetic: passing a Series of the same
    dtype straight to ``is_in`` is ambiguous between element-wise and
    collection membership, and Polars deprecated it (pola-rs/polars#22149).
    Imploding to a single List value states "membership in this one
    collection" explicitly.
    """
    if not isinstance(rep_idx, pl.Series):
        rep_idx = pl.Series(TMP_IDX_COL, rep_idx)
    return lf.filter(pl.col(TMP_IDX_COL).is_in(rep_idx.implode()))


def filter_by_index_file(
    lf: pl.LazyFrame, index_file: Optional[os.PathLike]
) -> pl.LazyFrame:
    """
    Filter a cell-level LazyFrame to the row indices stored in a single-column
    integer parquet file.

    Adds ``TMP_IDX_COL`` via :func:`add_row_index`. If ``index_file`` is
    given, it is read as a parquet file with a single integer column named
    ``TMP_IDX_COL`` (as written by
    :func:`fisseq_data_pipeline.generatesplit.main`), and ``lf`` is
    filtered to those rows via :func:`get_replicate_lf`. If ``index_file`` is
    ``None``, all rows are kept. ``TMP_IDX_COL`` is dropped from the result
    either way.

    Parameters
    ----------
    lf : pl.LazyFrame
        Cell-level LazyFrame.
    index_file : PathLike or None
        Path to an index parquet file, or ``None`` to keep all rows.

    Returns
    -------
    pl.LazyFrame
        ``lf``, optionally filtered, with ``TMP_IDX_COL`` removed.
    """
    lf = add_row_index(lf)
    if index_file is not None:
        idx = pl.read_parquet(index_file).get_column(TMP_IDX_COL)
        lf = get_replicate_lf(lf, idx)
    return lf.drop(TMP_IDX_COL)
