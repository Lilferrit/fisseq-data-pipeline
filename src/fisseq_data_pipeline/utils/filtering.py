"""Generic row/group filtering and downsampling helpers.

Shared by the barcode- and variant-classifier entry points (:mod:`.wtvwt`,
:mod:`.ovwt`, :mod:`.batchvsbatch`, :mod:`.wtvvariantpool`). Each consuming
module keeps a thin, same-named wrapper around
:func:`drop_small_groups`/:func:`downsample_group_to_target` where its own
call sites depend on module-specific parameter names (e.g. `.wtvwt`'s
``filter_min_cells_per_barcode``, `.ovwt`'s ``downsample_wildtype``) --
:func:`_exclude_blocked_features` is re-exported as-is since every consumer
already calls it positionally.
"""

from typing import Optional

import polars as pl


def _exclude_blocked_features(
    feature_cols: list[str], feature_block_list_file: Optional[str]
) -> list[str]:
    """
    Drop blocked feature names from a feature column list.

    Parameters
    ----------
    feature_cols : list[str]
        Candidate feature column names.
    feature_block_list_file : str or None
        Path to a parquet file with ``feature`` (str) and ``feature_ok``
        (bool) columns, or ``None`` to skip filtering entirely.

    Returns
    -------
    list[str]
        ``feature_cols`` with any feature whose ``feature_ok`` is ``False``
        removed. Unchanged if ``feature_block_list_file`` is ``None``.
    """
    if feature_block_list_file is None:
        return feature_cols
    bl_df = pl.read_parquet(feature_block_list_file)
    blocked = set(bl_df.filter(~pl.col("feature_ok"))["feature"].to_list())
    return [f for f in feature_cols if f not in blocked]


def drop_small_groups(
    data_df: pl.DataFrame,
    group_col: str,
    min_rows: int,
) -> pl.DataFrame:
    """
    Remove groups (by ``group_col``) with fewer than ``min_rows`` rows.

    Every group is treated uniformly -- there is no exemption for any
    particular value (contrast :func:`.ovwt.filter_min_cells`, which always
    retains its wildtype group regardless of size; that is a genuinely
    different policy, not a special case of this function).

    Parameters
    ----------
    data_df : pl.DataFrame
        DataFrame containing ``group_col``.
    group_col : str
        Name of the column identifying each row's group.
    min_rows : int
        Minimum number of rows a group must have to be retained.

    Returns
    -------
    pl.DataFrame
        ``data_df`` with small groups removed.
    """
    counts = data_df.group_by(group_col).len()
    keep = counts.filter(pl.col("len") >= min_rows).get_column(group_col).to_list()
    return data_df.filter(pl.col(group_col).is_in(keep))


def downsample_group_to_target(
    data_df: pl.DataFrame,
    group_col: str,
    target_value,
    seed: int,
    n: Optional[int] = None,
) -> pl.DataFrame:
    """
    Downsample rows where ``group_col == target_value`` to a target count.

    If the targeted group is larger than the target, a random sample of its
    rows is drawn without replacement; every other group is left untouched.

    Parameters
    ----------
    data_df : pl.DataFrame
        DataFrame containing ``group_col``.
    group_col : str
        Name of the column identifying each row's group.
    target_value : Any
        The ``group_col`` value identifying the rows to downsample.
    seed : int
        Random seed for sampling.
    n : int or None
        Target row count for the targeted group. If ``None``, the target is
        the largest per-value row count among every *other* ``group_col``
        value.

    Returns
    -------
    pl.DataFrame
        ``data_df`` with the targeted group downsampled, or unchanged if
        already at or below the target.
    """
    if n is None:
        target = (
            data_df.filter(pl.col(group_col) != target_value)
            .group_by(group_col)
            .len()
            .get_column("len")
            .max()
        )
    else:
        target = n
    target_df = data_df.filter(pl.col(group_col) == target_value)
    if target is not None and len(target_df) > target:
        target_df = target_df.sample(n=target, seed=seed)
    return pl.concat([data_df.filter(pl.col(group_col) != target_value), target_df])
