"""WT-null bootstrap replicate computation.

Hydra entry point backing the Nextflow process ``WT_NULL_AGGREGATE``: for one
bootstrap replicate of one feature type, splits the batch's control (wildtype)
pool into two disjoint halves
(:func:`fisseq_data_pipeline.aggregate.split_control_pool`), optionally
downsamples each half independently
(:func:`fisseq_data_pipeline.aggregate.downsample_control`), and computes the
configured aggregator's WT-null (same-population) comparison between the two
halves for every feature — a comparison with zero true biological signal by
construction, used as that feature's null noise floor for one bootstrap
replicate. Part of the WT-null bootstrap reproducibility check
(:mod:`.wtnullblocklist` gathers these across bootstraps and applies the
Tukey-fence gate).

The split/downsample/relabel plumbing lives here; the per-aggregator
null-comparison logic itself lives on the aggregator classes in
:mod:`fisseq_data_pipeline.aggregate` (:meth:`BaseAggregator.null_comparison_statistic`
for the one-sample path used by aggregators with no reference pool to compare
against, e.g. ``mean``/``median``/``std``; ``ReferenceBasedAggregator``'s
override for the two-sample relabel-and-compare path used by
``KS``/``signedKS``/``QQ``/``AUROC``), invoked here via
:func:`fisseq_data_pipeline.aggregate.null_comparison_statistic`. Eligibility
is determined by :func:`fisseq_data_pipeline.aggregate.is_null_eligible` /
:func:`fisseq_data_pipeline.aggregate.null_eligible_aggregator_names` — an
aggregator is WT-null-eligible iff its ``null_statistic_transform`` is
concretely overridden rather than raising ``NotImplementedError`` (see
``BaseAggregator``'s docstring in :mod:`.aggregate`), not membership in a
hand-maintained dict.
"""

import dataclasses
import logging
import pathlib
from typing import Optional, Union

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .aggregate import (
    downsample_control,
    is_null_eligible,
    null_comparison_statistic,
    null_eligible_aggregator_names,
    split_control_pool,
)
from .config import InputConfig
from .utils.batches import load_batches
from .utils.constants import META_BARCODE_COL
from .utils.log import setup_logging

_cs = ConfigStore.instance()

# Constant synthetic group used to re-run the aggregator's normal
# self.aggregate() machinery on h1-vs-h2 instead of variant-vs-control (the
# ReferenceBasedAggregator two-sample path relabels h1 as the "group" and h2
# as the CONTROL_COLUMN reference pool; the BaseAggregator one-sample path
# relabels both halves independently under this same constant label). MUST
# carry the ``meta_`` prefix: FEATURE_SELECTOR (used by aggregate()'s
# ``_feature_columns``) excludes columns by that prefix alone, so an
# unprefixed synthetic label column would itself be swept up as a "feature"
# to aggregate -- colliding with its own use as the group-by key.
_WT_NULL_LABEL_COL = "meta_wt_null_group"
_WT_NULL_LABEL_VALUE = "wt_null"


@dataclasses.dataclass
class WtNullAggregateConfig(InputConfig):
    """
    Hydra structured configuration for the WT-null bootstrap replicate entry
    point.

    Attributes
    ----------
    aggregator : str
        A WT-null-eligible aggregator name: ``mean``, ``median``, ``std``,
        ``KS``, ``signedKS``, ``QQ``, or ``AUROC``. See
        :func:`fisseq_data_pipeline.aggregate.null_eligible_aggregator_names`
        for the authoritative, current list (``MAD``, ``KSnegLogP``, and
        ``AUROCnegLogP`` are deliberately excluded -- see their classes'
        docstrings in :mod:`.aggregate`). Note: this Python entry point
        accepts ``mean``/``median``/``std`` directly, but
        ``workflows/fisseq.nf``'s ``feature_select_wt_null_types`` default
        still only includes ``KS``/``QQ``/``AUROC`` -- opting the summary
        statistics into the pipeline-level bootstrap is a separate config
        change. Required.
    downsample_wt : float, int, or None
        Optional downsampling of each split half's control (wildtype) rows.
        A float in ``(0, 1)`` keeps that fraction; an int keeps that many
        (and is clamped -- with a logged warning -- to however many rows a
        half actually has, if the requested count is larger). ``None``
        disables downsampling: each half is the full disjoint split of the
        control pool. Defaults to ``None``.
    bootstrap_idx : int
        This replicate's bootstrap-loop index (in the Nextflow pipeline,
        ``1..params.feature_select_wt_null_bootstraps``). Seeds the disjoint
        split directly (``seed=bootstrap_idx``) and, when ``downsample_wt``
        is set, seeds each half's downsample independently
        (``bootstrap_idx*2 + 1`` for h1, ``bootstrap_idx*2 + 2`` for h2) --
        the same per-``(bootstrap_idx, half_num)`` seed derivation the old
        ``AGGREGATE_HALF`` used. Required.
    per_barcode : bool
        If ``True``, compute each feature's statistic per (synthetic group,
        barcode) first, then reduce to one value by median across barcodes,
        instead of pooling each half's cells directly. Must match
        ``AGGREGATE_FEATURE_TYPE``'s setting for the same batch, or the
        WT-null check stops being apples-to-apples. Defaults to ``False``.
    barcode_column : str
        Column identifying the barcode a cell was measured from. Only
        consulted when ``per_barcode`` is ``True``. Defaults to
        ``utils.constants.META_BARCODE_COL`` (``"meta_barcode"``).
    """

    aggregator: str = MISSING
    downsample_wt: Optional[Union[float, int]] = None
    bootstrap_idx: int = MISSING
    per_barcode: bool = False
    barcode_column: str = META_BARCODE_COL


_cs.store(name="wt_null_aggregate_main", node=WtNullAggregateConfig)


def _validate_downsample_wt(downsample_wt: Optional[Union[float, int]]) -> None:
    if downsample_wt is None:
        return
    if isinstance(downsample_wt, float) and not (0 < downsample_wt < 1):
        raise ValueError(
            f"downsample_wt float must satisfy 0 < x < 1, got {downsample_wt}"
        )
    if isinstance(downsample_wt, int) and downsample_wt <= 0:
        raise ValueError(f"downsample_wt int must be positive, got {downsample_wt}")


def _downsample_half(
    half_lf: pl.LazyFrame,
    downsample_wt: Union[float, int],
    seed: int,
    half_label: str,
) -> pl.LazyFrame:
    """
    Downsample one already-split control-only half, warning once (rather
    than silently falling back, which is :func:`downsample_control`'s
    existing behavior for an over-large int target) when the requested size
    exceeds what this half actually has available.
    """
    if isinstance(downsample_wt, int):
        n_available = half_lf.select(pl.len()).collect().item()
        if downsample_wt > n_available:
            logging.warning(
                "downsample_wt=%d exceeds the %d control cell(s) available in "
                "%s; using all %d instead. This usually means downsample_wt is "
                "set too high for this dataset.",
                downsample_wt,
                n_available,
                half_label,
                n_available,
            )
    return downsample_control(half_lf, downsample_wt, seed)


@hydra.main(version_base=None, config_path=None, config_name="wt_null_aggregate_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: compute one WT-null bootstrap replicate for one
    feature type.

    Loads ``input_file``, splits its control pool into two disjoint halves
    seeded by ``bootstrap_idx`` (:func:`.aggregate.split_control_pool`),
    optionally downsamples each half independently, then delegates the
    configured aggregator's WT-null comparison between the two halves to
    :func:`fisseq_data_pipeline.aggregate.null_comparison_statistic` -- which
    dispatches to :meth:`BaseAggregator.null_comparison_statistic`'s
    one-sample path or :class:`ReferenceBasedAggregator`'s two-sample relabel
    path depending on the aggregator, with each aggregator's own
    ``null_statistic_transform`` already applied so larger always means more
    suspicious -- then writes the result in long format.

    Output file
    -----------
    - ``{output_dir}/wt_null.parquet`` with columns ``feature``, ``value``.

    Raises
    ------
    ValueError
        If ``aggregator`` is not a WT-null-eligible aggregator, or if
        ``downsample_wt`` fails its range check.
    """
    ft_cfg: WtNullAggregateConfig = OmegaConf.to_object(cfg)

    if not is_null_eligible(ft_cfg.aggregator):
        raise ValueError(
            f"Unknown or ineligible WT-null aggregator {ft_cfg.aggregator!r}. "
            f"Choose from: {null_eligible_aggregator_names()}"
        )
    _validate_downsample_wt(ft_cfg.downsample_wt)

    output_dir = pathlib.Path(ft_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ft_cfg.output_dir = output_dir
    setup_logging(ft_cfg, "wt_null_aggregate")

    logging.info("Loading input from %s", ft_cfg.input_file)
    lf, _ = load_batches(ft_cfg.input_file)

    logging.info("Splitting control pool: seed=%d", ft_cfg.bootstrap_idx)
    h1_lf, h2_lf = split_control_pool(lf, seed=ft_cfg.bootstrap_idx)

    if ft_cfg.downsample_wt is not None:
        h1_lf = _downsample_half(
            h1_lf, ft_cfg.downsample_wt, ft_cfg.bootstrap_idx * 2 + 1, "h1"
        )
        h2_lf = _downsample_half(
            h2_lf, ft_cfg.downsample_wt, ft_cfg.bootstrap_idx * 2 + 2, "h2"
        )

    logging.info("Running %s aggregator between WT halves", ft_cfg.aggregator)
    # Keeps each column's full stat-suffixed name (e.g. "f1_KS") as the
    # "feature" identity, unchanged -- this must match the column names in
    # AGGREGATE_FEATURE_TYPE's output exactly, since FINALIZE_FEATURE_SELECT
    # drops blocked columns by looking them up under this same name. It's
    # also what keeps different feature types' features from colliding once
    # WT_NULL_BLOCKLIST's output is combined with every other feature type's
    # (see combineblocklists.py's docstring).
    long_df = null_comparison_statistic(
        h1_lf,
        h2_lf,
        label_col=_WT_NULL_LABEL_COL,
        label_value=_WT_NULL_LABEL_VALUE,
        aggregator_name=ft_cfg.aggregator,
        per_barcode=ft_cfg.per_barcode,
        barcode_column=ft_cfg.barcode_column,
    )

    out_path = output_dir / "wt_null.parquet"
    logging.info("Writing WT-null replicate to %s", out_path)
    long_df.collect().write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
