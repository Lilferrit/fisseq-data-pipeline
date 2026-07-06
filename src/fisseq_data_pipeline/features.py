import dataclasses
import logging
import pathlib

import hydra
import polars as pl
import pycytominer
import scipy.stats
import sklearn.model_selection
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .aggregate import AggregateConfig, aggregate, variant_classification
from .utils.batches import load_batches
from .utils.constants import FEATURE_SELECTOR
from .utils.log import setup_logging
from .utils.metadata import get_aggregate_meta_data, get_column
from .utils.vectors import compute_impact_score

TMP_IDX_COL = "tmp_cell_idx"

_cs = ConfigStore.instance()


@dataclasses.dataclass
class FeatureSelectConfig(AggregateConfig):
    """
    Hydra structured configuration for the feature-selection entry point.

    Extends :class:`.aggregate.AggregateConfig` with parameters controlling
    pseudo-replicate splitting and reproducibility filtering.

    Attributes
    ----------
    random_state : int
        Seed passed to :func:`sklearn.model_selection.train_test_split` when
        splitting cells into pseudo-replicates. Defaults to ``42``.
    minimum_correlation : float
        Minimum Pearson *r* required for a feature to pass the reproducibility
        filter. Features whose pseudo-replicate correlation falls below this
        threshold are excluded from aggregation. Defaults to ``0.5``.
    """

    random_state: int = 42
    minimum_correlation: float = 0.5


_cs.store(name="feature_select_main", node=FeatureSelectConfig)


def get_replicate_lf(lf: pl.LazyFrame, rep_idx: list[int]) -> pl.LazyFrame:
    """
    Filter a LazyFrame to rows belonging to one pseudo-replicate.

    Parameters
    ----------
    lf : pl.LazyFrame
        Cell-level LazyFrame that must contain a ``TMP_IDX_COL`` integer column
        added by :func:`pseudo_replicate_correlation` before splitting.
    rep_idx : list[int]
        Row indices that belong to this replicate half.

    Returns
    -------
    pl.LazyFrame
        Subset of ``lf`` containing only the rows whose ``TMP_IDX_COL`` value
        is in ``rep_idx``.
    """
    return lf.filter(pl.col(TMP_IDX_COL).is_in(set(rep_idx)))


def compute_feature_correlations(
    df1: pl.DataFrame, df2: pl.DataFrame, label_col: str
) -> pl.DataFrame:
    """
    Compute per-feature Pearson correlations between two aggregate DataFrames.

    Both DataFrames are joined on ``label_col`` so that each row pairs the
    per-label aggregate from replicate 1 with the corresponding value from
    replicate 2. Pearson *r*, *r²*, and the two-tailed *p*-value are reported
    for each feature column.

    Parameters
    ----------
    df1 : pl.DataFrame
        Aggregate DataFrame for the first pseudo-replicate. Must contain
        ``label_col`` and one or more feature columns (no ``meta_`` prefix).
    df2 : pl.DataFrame
        Aggregate DataFrame for the second pseudo-replicate. Same schema as
        ``df1``.
    label_col : str
        Name of the column used to align the two DataFrames (e.g.
        ``"meta_aa_changes"``).

    Returns
    -------
    pl.DataFrame
        One row per feature with columns ``feature``, ``r``, ``r_squared``,
        and ``p_value``.
    """
    df1 = df1.select(FEATURE_SELECTOR, pl.col(label_col))
    df2 = df2.select(FEATURE_SELECTOR, pl.col(label_col))
    df_joined = df1.join(df2, on=label_col, suffix="_right")

    result = []
    for feature in df1.columns:
        if feature == label_col:
            continue

        left_col = df_joined.get_column(feature).to_numpy()
        right_col = df_joined.get_column(f"{feature}_right").to_numpy()

        corr, p_val = scipy.stats.pearsonr(left_col, right_col)
        stat = corr**2
        result.append(
            ({"feature": feature, "r": corr, "r_squared": stat, "p_value": p_val})
        )

    return pl.DataFrame(result, schema=["feature", "r", "r_squared", "p_value"])


def pseudo_replicate_correlation(
    lf: pl.LazyFrame, label_col: str, aggregator_name: str, random_state: int
) -> pl.DataFrame:
    """
    Estimate feature reliability via pseudo-replicate Pearson correlations.

    Cells are split into two equal halves stratified by variant label. Each
    half is aggregated independently, and per-feature Pearson *r* between the
    two aggregate results is returned as a proxy for measurement
    reproducibility.

    Parameters
    ----------
    lf : pl.LazyFrame
        Cell-level LazyFrame. Must contain ``label_col`` and feature columns.
    label_col : str
        Column identifying variant labels, used both for stratified splitting
        and for aligning the two aggregate results.
    aggregator_name : str
        Aggregation method passed through to :func:`.aggregate.aggregate`.
    random_state : int
        Random seed for the stratified split, forwarded to
        :func:`sklearn.model_selection.train_test_split`.

    Returns
    -------
    pl.DataFrame
        One row per feature with columns ``feature``, ``r``, ``r_squared``,
        and ``p_value`` (from :func:`compute_feature_correlations`).
    """
    lf = lf.with_columns(pl.row_index(TMP_IDX_COL))
    idx = get_column(lf, TMP_IDX_COL)
    labels = get_column(lf, label_col)

    rep_one_idx, rep_two_idx, _, _ = sklearn.model_selection.train_test_split(
        idx, labels, stratify=labels, random_state=random_state, test_size=0.5
    )

    rep_one_lf = get_replicate_lf(lf, rep_one_idx).drop(TMP_IDX_COL)
    rep_two_lf = get_replicate_lf(lf, rep_two_idx).drop(TMP_IDX_COL)

    rep_one_aggregate_df = aggregate(rep_one_lf, label_col, aggregator_name).collect()
    rep_two_aggregate_df = aggregate(rep_two_lf, label_col, aggregator_name).collect()
    rep_correlation_df = compute_feature_correlations(
        rep_one_aggregate_df, rep_two_aggregate_df, label_col
    )

    return rep_correlation_df


def pyc_feature_select(agg_df: pl.DataFrame) -> pl.DataFrame:
    """
    Select informative features from a per-variant aggregate DataFrame using
    pycytominer.

    Applies three sequential filters via :func:`pycytominer.feature_select`:
    low-variance removal (``variance_threshold``), pycytominer's built-in
    blocklist, and redundancy removal (``correlation_threshold``).

    Parameters
    ----------
    agg_df : pl.DataFrame
        Per-variant aggregate DataFrame. Feature columns must match
        ``FEATURE_SELECTOR`` (i.e. no ``meta_`` prefix).

    Returns
    -------
    pl.DataFrame
        Subset of ``agg_df`` retaining only the features that pass all three
        filters. Non-feature (``meta_``) columns are preserved unchanged.
    """
    select_agg_df_pd = pycytominer.feature_select(
        profiles=agg_df.to_pandas(),
        features=agg_df.select(FEATURE_SELECTOR).columns,
        image_features=False,
        samples="all",
        operation=["variance_threshold", "blocklist", "correlation_threshold"],
    )

    return pl.from_pandas(select_agg_df_pd)


@hydra.main(version_base=None, config_path=None, config_name="feature_select_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: feature selection via pseudo-replicate correlation and
    pycytominer.

    Steps
    -----
    1. Compute per-feature pseudo-replicate Pearson correlations to estimate
       measurement reproducibility.
    2. Mark features with ``r < minimum_correlation`` as blocked and write the
       full correlation table (including a ``feature_ok`` column) to the output
       directory.
    3. Aggregate the full input using :func:`.aggregate.aggregate`, skipping any
       blocked features.
    4. Apply :func:`pyc_feature_select` to remove low-variance and redundant
       features.
    5. Write the final feature-selected aggregate to disk.

    Output files
    ------------
    - ``{output_root}.feature_correlations.parquet`` or
      ``{output_dir}/feature_correlations.parquet``
    - ``{output_root}.{stem}.{ext}`` or ``{output_dir}/{filename}``

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.features \\
            output_dir=./out \\
            input_file=data/cells.parquet \\
            minimum_correlation=0.5
    """
    feat_cfg: FeatureSelectConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(feat_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_cfg.output_dir = output_dir
    setup_logging(feat_cfg, "features")

    logging.info("Loading input from %s", feat_cfg.input_file)
    lf, output_stem = load_batches(feat_cfg.input_file)

    logging.info("Computing pseudo-replicate correlations")
    corr_df = pseudo_replicate_correlation(
        lf,
        label_col=feat_cfg.label_column,
        aggregator_name=feat_cfg.aggregator,
        random_state=feat_cfg.random_state,
    )

    corr_df = corr_df.with_columns(
        (pl.col("r") >= feat_cfg.minimum_correlation).alias("feature_ok")
    )
    block_list = set(corr_df.filter(~pl.col("feature_ok"))["feature"].to_list())
    logging.info(
        "Blocking %d features with r < %g",
        len(block_list),
        feat_cfg.minimum_correlation,
    )

    if feat_cfg.output_root is not None:
        corr_path = pathlib.Path(f"{feat_cfg.output_root}.feature_correlations.parquet")
        out_path = pathlib.Path(f"{feat_cfg.output_root}.{output_stem}.parquet")
    else:
        corr_path = output_dir / "feature_correlations.parquet"
        out_path = output_dir / f"{output_stem}.parquet"

    logging.info("Writing feature correlations to %s", corr_path)
    corr_df.write_parquet(corr_path)

    logging.info("Aggregating input with block list")
    agg_df = aggregate(
        lf,
        label_col=feat_cfg.label_column,
        aggregator_name=feat_cfg.aggregator,
        block_list=block_list,
    ).collect()

    logging.info("Running pycytominer feature selection")
    selected_df = pyc_feature_select(agg_df)

    if cfg.compute_impact_score:
        logging.info("Computing impact scores")
        selected_lf = variant_classification(selected_df.lazy(), feat_cfg.label_column)
        selected_df = compute_impact_score(selected_lf).collect()

    meta_lf = get_aggregate_meta_data(lf, feat_cfg.label_column)
    selected_lf = selected_df.lazy().join(meta_lf, on=feat_cfg.label_column)

    logging.info("Writing output to %s", out_path)
    selected_lf.sink_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
