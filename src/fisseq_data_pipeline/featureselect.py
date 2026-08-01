"""Final feature-selection stage of the bootstrap pseudo-replicate pipeline.

Hydra entry point backing the Nextflow process ``FINALIZE_FEATURE_SELECT``: joins
per-feature-type aggregates (per-feature-type aggregation itself lives in
:mod:`.aggregatefeaturetype`, run as ``AGGREGATE_HALF``), applies the combined
blocklist (from :mod:`.combineblocklists`), and runs pycytominer feature
selection.
"""

import dataclasses
import glob
import logging
import pathlib

import hydra
import polars as pl
import pycytominer
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .aggregate import variant_classification
from .config import LabeledInputConfig
from .normalize import Normalizer
from .utils.batches import load_batches
from .utils.constants import FEATURE_SELECTOR
from .utils.featuretypes import join_feature_type_files
from .utils.log import setup_logging
from .utils.metadata import get_aggregate_meta_data
from .utils.vectors import compute_impact_score

_cs = ConfigStore.instance()


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
        files produced by :func:`fisseq_data_pipeline.aggregatefeaturetype.main`.
        Each file contains ``[label_column] + <feature type's stat
        columns>``; all matching files are joined on ``label_column`` to
        reconstruct the combined per-variant aggregate table. Required.
    block_list_file : str
        Path to the combined blocklist parquet, with ``feature`` and
        ``feature_ok`` columns. Required.
    compute_impact_score : bool
        If ``True``, compute per-variant impact score (cosine distance vs
        synonymous baseline) after feature selection and normalization.
        Defaults to ``True``.
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
    5. Mark synonymous variants as the normalization reference
       (:func:`fisseq_data_pipeline.aggregate.variant_classification`), fit a
       :class:`.normalize.Normalizer` on those rows, and apply it — the
       output features are z-score normalized to the synonymous baseline.
    6. Optionally compute impact score on the normalized features
       (:func:`.utils.vectors.compute_impact_score`).
    7. Join per-variant metadata via :func:`.utils.metadata.get_aggregate_meta_data`.
    8. Write output.

    Output path
    -----------
    - Glob input: ``{output_root}.output.parquet`` or ``{output_dir}/output.parquet``
    - Single-file input: ``{output_root}.{stem}.parquet`` or
      ``{output_dir}/{stem}.parquet``

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.featureselect \\
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
    agg_df = join_feature_type_files(ft_paths, feat_cfg.label_column)

    logging.info("Loading block list from %s", feat_cfg.block_list_file)
    bl_df = pl.read_parquet(feat_cfg.block_list_file)
    block_list = set(bl_df.filter(~pl.col("feature_ok"))["feature"].to_list())
    logging.info(
        "Dropping %d blocked feature(s)", len(block_list & set(agg_df.columns))
    )
    agg_df = agg_df.drop([c for c in block_list if c in agg_df.columns])

    logging.info("Running pycytominer feature selection")
    selected_df = pyc_feature_select(agg_df)

    logging.info(
        "Classifying variants and marking synonymous as normalization reference"
    )
    selected_lf = variant_classification(selected_df.lazy(), feat_cfg.label_column)

    logging.info("Fitting normalizer on synonymous rows")
    normalizer = Normalizer.from_lazyframe(selected_lf, fit_only_on_control=True)
    logging.info("Applying normalizer")
    normalized_lf = normalizer.apply(selected_lf)

    if feat_cfg.compute_impact_score:
        logging.info("Computing impact scores")
        normalized_lf = compute_impact_score(normalized_lf)

    logging.info("Adding queries to retrieve metadata")
    meta_lf = get_aggregate_meta_data(lf, feat_cfg.label_column)
    selected_lf = normalized_lf.join(meta_lf, on=feat_cfg.label_column)

    if feat_cfg.output_root is not None:
        out_path = pathlib.Path(f"{feat_cfg.output_root}.{output_stem}.parquet")
    else:
        out_path = output_dir / f"{output_stem}.parquet"

    logging.info("Writing output to %s", out_path)
    selected_lf.sink_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
