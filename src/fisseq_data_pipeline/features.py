"""Bootstrap pseudo-replicate feature selection pipeline.

Hydra entry points backing the Nextflow processes ``GENERATE_SPLIT``,
``CORRELATE_FEATURES``, ``BLOCKLIST``, ``COMBINE_BLOCKLISTS``, and
``FINALIZE_FEATURE_SELECT`` (per-feature-type aggregation itself lives in
:mod:`.aggregate`, run as ``AGGREGATE_HALF``): split cells into stratified
pseudo-replicate halves, correlate per-feature-type aggregates between halves across
bootstrap replicates, derive a per-feature blocklist from the median correlation, and
apply it plus pycytominer feature selection to produce the final per-variant
aggregate.
"""

import dataclasses
import glob
import logging
import pathlib

import hydra
import polars as pl
import pycytominer
import scipy.stats
import sklearn.model_selection
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .aggregate import variant_classification
from .config import AppConfig, LabeledInputConfig
from .utils.batches import load_batches
from .utils.constants import FEATURE_SELECTOR
from .utils.log import setup_logging
from .utils.metadata import get_aggregate_meta_data, get_column
from .utils.splits import TMP_IDX_COL, add_row_index
from .utils.vectors import compute_impact_score

_cs = ConfigStore.instance()


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


@dataclasses.dataclass
class GenerateSplitConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the pseudo-replicate split-generation
    entry point.

    Attributes
    ----------
    random_state : int
        Seed for the stratified 50/50
        :func:`sklearn.model_selection.train_test_split` by ``label_column``.
        In the Nextflow pipeline this is set directly to the bootstrap-loop
        index (``1..params.bootstrap``), so each bootstrap replicate gets a
        distinct, reproducible split. Required.
    """

    random_state: int = MISSING


_cs.store(name="generate_split_main", node=GenerateSplitConfig)


@hydra.main(version_base=None, config_path=None, config_name="generate_split_main")
def generate_split_main(cfg: DictConfig) -> None:
    """
    Hydra entry point: generate one pseudo-replicate 50/50 split.

    Loads ``input_file``, adds a row-index column
    (:func:`.utils.splits.add_row_index`), and performs a stratified (by
    ``label_column``) 50/50 :func:`sklearn.model_selection.train_test_split`
    at seed ``random_state``. Each half's row indices are written as a
    single-column (``TMP_IDX_COL``) parquet file, consumable by
    :func:`fisseq_data_pipeline.aggregate.feature_type_main`'s ``index_file``
    option.

    Output files
    ------------
    - ``{output_dir}/half1.parquet``
    - ``{output_dir}/half2.parquet``

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.features \\
            output_dir=./out \\
            input_file=data/normalized.parquet \\
            random_state=3
    """
    split_cfg: GenerateSplitConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(split_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_cfg.output_dir = output_dir
    setup_logging(split_cfg, "generate_split")

    logging.info("Loading input from %s", split_cfg.input_file)
    lf, _ = load_batches(split_cfg.input_file)
    lf = add_row_index(lf)

    idx = get_column(lf, TMP_IDX_COL)
    labels = get_column(lf, split_cfg.label_column)

    half1_idx, half2_idx, _, _ = sklearn.model_selection.train_test_split(
        idx, labels, stratify=labels, random_state=split_cfg.random_state, test_size=0.5
    )

    logging.info(
        "Writing half1.parquet (%d rows) and half2.parquet (%d rows)",
        len(half1_idx),
        len(half2_idx),
    )
    pl.DataFrame({TMP_IDX_COL: half1_idx}).write_parquet(output_dir / "half1.parquet")
    pl.DataFrame({TMP_IDX_COL: half2_idx}).write_parquet(output_dir / "half2.parquet")

    logging.info("Done")


@dataclasses.dataclass
class CorrelateFeaturesConfig(AppConfig):
    """
    Hydra structured configuration for the pseudo-replicate correlation entry
    point. Extends :class:`.config.AppConfig` (not
    :class:`.config.LabeledInputConfig`) since there is no cell-level
    ``input_file`` here — the inputs are two already aggregated
    per-feature-type parquet files.

    Attributes
    ----------
    half1_file : str
        Path to the first split half's per-feature-type aggregate parquet
        (output of :func:`fisseq_data_pipeline.aggregate.feature_type_main`
        with ``index_file=half1.parquet``). Required.
    half2_file : str
        Path to the second split half's per-feature-type aggregate parquet.
        Required.
    label_column : str
        Name of the column identifying variant labels. Defaults to
        ``"meta_aa_changes"``.
    """

    half1_file: str = MISSING
    half2_file: str = MISSING
    label_column: str = "meta_aa_changes"


_cs.store(name="correlate_features_main", node=CorrelateFeaturesConfig)


@hydra.main(version_base=None, config_path=None, config_name="correlate_features_main")
def correlate_features_main(cfg: DictConfig) -> None:
    """
    Hydra entry point: compute per-feature pseudo-replicate Pearson
    correlations between two aggregate halves.

    Reads ``half1_file`` and ``half2_file`` (both outputs of
    :func:`fisseq_data_pipeline.aggregate.feature_type_main` for the same
    feature type, one per split half) and calls
    :func:`compute_feature_correlations`.

    Output file
    -----------
    - ``{output_dir}/correlations.parquet``
    """
    corr_cfg: CorrelateFeaturesConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(corr_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corr_cfg.output_dir = output_dir
    setup_logging(corr_cfg, "correlate_features")

    df1 = pl.read_parquet(corr_cfg.half1_file)
    df2 = pl.read_parquet(corr_cfg.half2_file)
    corr_df = compute_feature_correlations(df1, df2, corr_cfg.label_column)

    out_path = output_dir / "correlations.parquet"
    logging.info("Writing correlations to %s", out_path)
    corr_df.write_parquet(out_path)

    logging.info("Done")


@dataclasses.dataclass
class BlocklistConfig(AppConfig):
    """
    Hydra structured configuration for the per-feature-type blocklist
    generation entry point.

    Attributes
    ----------
    correlation_files : str
        Glob pattern matching all bootstrap-replicate correlation parquet
        files for one feature type (outputs of
        :func:`correlate_features_main`). Required.
    minimum_correlation : float
        Minimum median Pearson *r* (across bootstrap replicates) required for
        a feature to pass. Defaults to ``0.5``.
    """

    correlation_files: str = MISSING
    minimum_correlation: float = 0.5


_cs.store(name="blocklist_main", node=BlocklistConfig)


@hydra.main(version_base=None, config_path=None, config_name="blocklist_main")
def blocklist_main(cfg: DictConfig) -> None:
    """
    Hydra entry point: compute a per-feature-type blocklist from N bootstrap
    correlation tables.

    This is the one intentional synchronization point across bootstrap
    replicates in the feature-selection pipeline. Globs
    ``correlation_files``, concatenates all bootstrap-replicate correlation
    tables for one feature type, computes each feature's median ``r`` across
    replicates, and marks ``feature_ok = median_r >= minimum_correlation``.

    Output file
    -----------
    - ``{output_dir}/blocklist.parquet`` with columns ``feature``,
      ``median_r``, ``feature_ok``.

    Raises
    ------
    ValueError
        If ``correlation_files`` matches no files.
    """
    bl_cfg: BlocklistConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(bl_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bl_cfg.output_dir = output_dir
    setup_logging(bl_cfg, "blocklist")

    paths = sorted(glob.glob(bl_cfg.correlation_files))
    if not paths:
        raise ValueError(f"No files matched glob pattern: {bl_cfg.correlation_files!r}")
    logging.info("Found %d bootstrap correlation file(s)", len(paths))
    corr_df = pl.concat([pl.read_parquet(p) for p in paths])

    blocklist_df = (
        corr_df.group_by("feature")
        .agg(pl.col("r").median().alias("median_r"))
        .with_columns(
            (pl.col("median_r") >= bl_cfg.minimum_correlation).alias("feature_ok")
        )
    )

    out_path = output_dir / "blocklist.parquet"
    logging.info("Writing blocklist to %s", out_path)
    blocklist_df.write_parquet(out_path)

    logging.info("Done")


@dataclasses.dataclass
class CombineBlocklistsConfig(AppConfig):
    """
    Hydra structured configuration for the blocklist-combination entry point.

    Attributes
    ----------
    blocklist_files : str
        Glob pattern matching all per-feature-type blocklist parquet files
        (outputs of :func:`blocklist_main`). Required.
    """

    blocklist_files: str = MISSING


_cs.store(name="combine_blocklists_main", node=CombineBlocklistsConfig)


@hydra.main(version_base=None, config_path=None, config_name="combine_blocklists_main")
def combine_blocklists_main(cfg: DictConfig) -> None:
    """
    Hydra entry point: concatenate all per-feature-type blocklists into one
    combined blocklist.

    Globs ``blocklist_files`` and concatenates them with no deduplication —
    each feature type's blocklist covers a disjoint set of feature columns
    (stat-suffixed column names like ``f1_mean`` and ``f1_EMD`` never
    collide across feature types), so a plain concat is correct.

    Output file
    -----------
    - ``{output_dir}/blocklist.parquet``

    Raises
    ------
    ValueError
        If ``blocklist_files`` matches no files.
    """
    cb_cfg: CombineBlocklistsConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(cb_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cb_cfg.output_dir = output_dir
    setup_logging(cb_cfg, "combine_blocklists")

    paths = sorted(glob.glob(cb_cfg.blocklist_files))
    if not paths:
        raise ValueError(f"No files matched glob pattern: {cb_cfg.blocklist_files!r}")
    logging.info("Found %d per-feature-type blocklist file(s)", len(paths))
    combined = pl.concat([pl.read_parquet(p) for p in paths])

    out_path = output_dir / "blocklist.parquet"
    logging.info("Writing combined blocklist to %s", out_path)
    combined.write_parquet(out_path)

    logging.info("Done")


@dataclasses.dataclass
class FinalizeFeatureSelectConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the final feature-selection entry
    point: joins per-feature-type aggregates, applies the combined
    blocklist, and runs pycytominer feature selection.

    ``input_file`` (inherited) is the raw/normalized cell-level input, used
    only to derive per-variant metadata (:func:`.utils.metadata.get_aggregate_meta_data`).

    Attributes
    ----------
    feature_type_files : str
        Glob pattern matching the per-feature-type full aggregate parquet
        files produced by :func:`fisseq_data_pipeline.aggregate.feature_type_main`.
        Each file contains ``[label_column] + <feature type's stat
        columns>``; all matching files are joined on ``label_column`` to
        reconstruct the combined per-variant aggregate table. Required.
    block_list_file : str
        Path to the combined blocklist parquet, with ``feature`` and
        ``feature_ok`` columns. Required.
    compute_impact_score : bool
        If ``True``, compute per-variant impact score (cosine distance vs
        synonymous baseline) after feature selection. Defaults to ``True``.
    """

    feature_type_files: str = MISSING
    block_list_file: str = MISSING
    compute_impact_score: bool = True


_cs.store(name="feature_select_main", node=FinalizeFeatureSelectConfig)


@hydra.main(version_base=None, config_path=None, config_name="feature_select_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: final feature-selection stage.

    Steps
    -----
    1. Load raw cell-level input at ``input_file`` (for metadata join only).
    2. Glob ``feature_type_files`` and join each per-feature-type aggregate
       parquet on ``label_column``, reconstructing the combined per-variant
       aggregate table.
    3. Load ``block_list_file`` and drop blocked feature columns.
    4. Run :func:`pyc_feature_select` (variance threshold, pycytominer
       blocklist, correlation threshold).
    5. Optionally compute impact score (:func:`.aggregate.variant_classification`
       + :func:`.utils.vectors.compute_impact_score`).
    6. Join per-variant metadata via :func:`.utils.metadata.get_aggregate_meta_data`.
    7. Write output.

    Output path
    -----------
    - Glob input: ``{output_root}.output.parquet`` or ``{output_dir}/output.parquet``
    - Single-file input: ``{output_root}.{stem}.parquet`` or
      ``{output_dir}/{stem}.parquet``

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.features \\
            output_dir=./out \\
            input_file=data/normalized.parquet \\
            'feature_type_files=./ft/*.parquet' \\
            block_list_file=./blocklist.parquet
    """
    feat_cfg: FinalizeFeatureSelectConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(feat_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_cfg.output_dir = output_dir
    setup_logging(feat_cfg, "features")

    logging.info("Loading raw input from %s", feat_cfg.input_file)
    lf, output_stem = load_batches(feat_cfg.input_file)

    logging.info(
        "Loading per-feature-type aggregates from %s", feat_cfg.feature_type_files
    )
    ft_paths = sorted(glob.glob(feat_cfg.feature_type_files))
    if not ft_paths:
        raise ValueError(
            f"No files matched glob pattern: {feat_cfg.feature_type_files!r}"
        )
    agg_df = pl.read_parquet(ft_paths[0])
    for p in ft_paths[1:]:
        agg_df = agg_df.join(pl.read_parquet(p), on=feat_cfg.label_column)

    logging.info("Loading block list from %s", feat_cfg.block_list_file)
    bl_df = pl.read_parquet(feat_cfg.block_list_file)
    block_list = set(bl_df.filter(~pl.col("feature_ok"))["feature"].to_list())
    logging.info(
        "Dropping %d blocked feature(s)", len(block_list & set(agg_df.columns))
    )
    agg_df = agg_df.drop([c for c in block_list if c in agg_df.columns])

    logging.info("Running pycytominer feature selection")
    selected_df = pyc_feature_select(agg_df)

    if feat_cfg.compute_impact_score:
        logging.info("Computing impact scores")
        selected_lf = variant_classification(selected_df.lazy(), feat_cfg.label_column)
        selected_df = compute_impact_score(selected_lf).collect()

    logging.info("Adding queries to retrieve metadata")
    meta_lf = get_aggregate_meta_data(lf, feat_cfg.label_column)
    selected_lf = selected_df.lazy().join(meta_lf, on=feat_cfg.label_column)

    if feat_cfg.output_root is not None:
        out_path = pathlib.Path(f"{feat_cfg.output_root}.{output_stem}.parquet")
    else:
        out_path = output_dir / f"{output_stem}.parquet"

    logging.info("Writing output to %s", out_path)
    selected_lf.sink_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
