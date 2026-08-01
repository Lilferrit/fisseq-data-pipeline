"""Helpers for reconstructing a combined per-variant aggregate table from
per-feature-type aggregate files (the output of
:func:`fisseq_data_pipeline.aggregatefeaturetype.main`).
"""

import polars as pl


def join_feature_type_files(paths: list[str], label_column: str) -> pl.DataFrame:
    """
    Join per-feature-type aggregate parquet files into one combined table.

    Each file is expected to contain ``[label_column] + <feature type's stat
    columns>``, one row per variant. Files are read eagerly and joined
    sequentially on ``label_column``, in the order given.

    Parameters
    ----------
    paths : list[str]
        Paths to per-feature-type aggregate parquet files. Must be non-empty.
    label_column : str
        Name of the column identifying variant labels, used as the join key.

    Returns
    -------
    pl.DataFrame
        Combined table with ``label_column`` plus every input file's stat
        columns.
    """
    agg_df = pl.read_parquet(paths[0])
    for p in paths[1:]:
        agg_df = agg_df.join(pl.read_parquet(p), on=label_column)
    return agg_df
