"""Pseudo-replicate feature correlation.

Hydra entry point backing the Nextflow process ``CORRELATE_FEATURES``: computes
per-feature Pearson correlations between two aggregated pseudo-replicate halves
(outputs of :func:`fisseq_data_pipeline.aggregatefeaturetype.main`), part of the
bootstrap feature-selection pipeline.

Optionally supports per-bootstrap-replicate variant downsampling
(``bootstrap_variant_downsample``): a random subset of the variants present in
both halves is used for each replicate's correlation, adding variant-sampling
variance on top of the half-split randomness. See
:func:`compute_feature_correlations`.
"""

import dataclasses
import logging
import pathlib
from typing import Optional

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .utils.constants import FEATURE_SELECTOR
from .utils.log import setup_logging

_cs = ConfigStore.instance()


def compute_feature_correlations(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    label_col: str,
    bootstrap_variant_downsample: Optional[int] = None,
    seed: Optional[int] = None,
) -> pl.DataFrame:
    """
    Compute per-feature Pearson correlations between two aggregate DataFrames.

    Both DataFrames are joined on ``label_col`` so that each row pairs the
    per-label aggregate from replicate 1 with the corresponding value from
    replicate 2. Pearson *r* and *r²* are reported for each feature column.

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
    bootstrap_variant_downsample : int or None
        If set, randomly sample this many variants from the joint set
        present in both halves (after the ``label_col`` join) before
        computing correlations. ``None`` (the default) disables this: every
        joint variant is used, matching prior behavior exactly. If the
        requested count exceeds the number of joint variants available, all
        of them are used instead (logged once as a warning, not per
        feature) — this usually signals a misconfiguration.
    seed : int or None
        Random seed for the ``bootstrap_variant_downsample`` draw. Ignored
        when ``bootstrap_variant_downsample`` is ``None``.

    Returns
    -------
    pl.DataFrame
        One row per feature with columns ``feature``, ``r``, and ``r_squared``.
    """
    df1 = df1.select(FEATURE_SELECTOR, pl.col(label_col))
    df2 = df2.select(FEATURE_SELECTOR, pl.col(label_col))
    df_joined = df1.join(df2, on=label_col, suffix="_right")

    if bootstrap_variant_downsample is not None:
        n_joint = df_joined.height
        n_sample = min(bootstrap_variant_downsample, n_joint)
        if bootstrap_variant_downsample > n_joint:
            logging.warning(
                "bootstrap_variant_downsample=%d exceeds the %d variant(s) "
                "present in both halves; using all %d instead. This usually "
                "means bootstrap_variant_downsample is set too high for "
                "this dataset.",
                bootstrap_variant_downsample,
                n_joint,
                n_joint,
            )
        df_joined = df_joined.sample(n=n_sample, seed=seed, shuffle=True)

    features = [c for c in df1.columns if c != label_col]
    corrs = df_joined.select(pl.corr(f, f"{f}_right").alias(f) for f in features).row(0)

    result = [
        {"feature": feature, "r": corr, "r_squared": None if corr is None else corr**2}
        for feature, corr in zip(features, corrs)
    ]
    return pl.DataFrame(result, schema=["feature", "r", "r_squared"])


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
        (output of :func:`fisseq_data_pipeline.aggregatefeaturetype.main`
        with ``index_file=half1.parquet``). Required.
    half2_file : str
        Path to the second split half's per-feature-type aggregate parquet.
        Required.
    label_column : str
        Name of the column identifying variant labels. Defaults to
        ``"meta_aa_changes"``.
    bootstrap_variant_downsample : int or None
        If set, randomly sample this many variants from the set present in
        BOTH halves (after the ``label_column`` join) before computing
        correlations, independently per bootstrap replicate — adds
        variant-subsampling variance on top of the half-split randomness
        that already exists between the two halves. ``None`` (the default)
        disables this: every variant present in both halves is used,
        matching prior behavior exactly. This is distinct from
        ``feature_select_downsample_wt`` (cell-level control-row
        downsampling, applied earlier at aggregation time) — this knob
        subsamples *variants*, at *correlation* time. If the requested
        count exceeds the number of variants available in the joint set,
        all available variants are used instead (logged once as a
        warning, not per feature) — this usually signals a
        misconfiguration, not a normal operating mode.
    bootstrap_idx : int
        This replicate's bootstrap-loop index (in the Nextflow pipeline,
        ``1..params.feature_select_bootstrap_reps``). Combined with
        ``seed`` to derive a per-replicate seed for
        ``bootstrap_variant_downsample``'s sampling draw
        (``seed + bootstrap_idx * 1000``) — deliberately independent of
        ``GENERATE_SPLIT``'s own per-replicate seed (set directly to
        ``bootstrap_idx``, no offset), so the variant-subsample draw does
        not correlate with the half-split draw. Ignored when
        ``bootstrap_variant_downsample`` is ``None``. Defaults to ``0``.
    seed : int
        Base seed combined with ``bootstrap_idx`` to derive the
        per-replicate variant-downsample seed (see ``bootstrap_idx``).
        Ignored when ``bootstrap_variant_downsample`` is ``None``.
        Defaults to ``0`` (mirrors
        :class:`fisseq_data_pipeline.aggregatefeaturetype.FeatureTypeAggregateConfig`'s
        ``seed`` field's role for ``downsample_wt``).
    """

    half1_file: str = MISSING
    half2_file: str = MISSING
    label_column: str = "meta_aa_changes"
    bootstrap_variant_downsample: Optional[int] = None
    bootstrap_idx: int = 0
    seed: int = 0


_cs.store(name="correlate_features_main", node=CorrelateFeaturesConfig)


@hydra.main(version_base=None, config_path=None, config_name="correlate_features_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: compute per-feature pseudo-replicate Pearson
    correlations between two aggregate halves.

    Reads ``half1_file`` and ``half2_file`` (both outputs of
    :func:`fisseq_data_pipeline.aggregatefeaturetype.main` for the same
    feature type, one per split half) and calls
    :func:`compute_feature_correlations`. When
    ``bootstrap_variant_downsample`` is set, the per-replicate seed passed
    to that sampling draw is ``seed + bootstrap_idx * 1000``.

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

    seed = None
    if corr_cfg.bootstrap_variant_downsample is not None:
        seed = corr_cfg.seed + corr_cfg.bootstrap_idx * 1000

    corr_df = compute_feature_correlations(
        df1,
        df2,
        corr_cfg.label_column,
        bootstrap_variant_downsample=corr_cfg.bootstrap_variant_downsample,
        seed=seed,
    )

    out_path = output_dir / "correlations.parquet"
    logging.info("Writing correlations to %s", out_path)
    corr_df.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
