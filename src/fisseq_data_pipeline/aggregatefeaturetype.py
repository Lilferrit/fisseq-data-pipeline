"""Lean per-feature-type cell-level aggregation.

Hydra entry point backing the Nextflow processes ``AGGREGATE_FEATURE_TYPE`` and
``AGGREGATE_HALF`` — shared by the feature-selection pipeline's stage 1 (full
aggregation) and stage 2b (per-pseudo-replicate-half aggregation). Also supports
optionally downsampling control (wildtype) rows before aggregation via
``downsample_wt``/``seed`` — see :func:`fisseq_data_pipeline.aggregate.downsample_control`.
"""

import dataclasses
import logging
import pathlib
from typing import Optional, Union

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .aggregate import aggregate, downsample_control
from .config import LabeledInputConfig
from .utils.batches import load_batches
from .utils.log import setup_logging
from .utils.splits import filter_by_index_file

_cs = ConfigStore.instance()


@dataclasses.dataclass
class FeatureTypeAggregateConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the lean per-feature-type aggregation
    entry point.

    Shared by the feature-selection pipeline's stage 1 (full aggregation)
    and stage 2b (per-pseudo-replicate-half aggregation).

    Attributes
    ----------
    aggregator : str
        A concrete key in ``fisseq_data_pipeline.aggregate._AGGREGATORS``
        (``mean``, ``median``, ``MAD``, ``std``, ``KS``, ``signedKS``, ``QQ``,
        ``AUROC``). Required.
    index_file : str or None
        Optional path to a single-column ``TMP_IDX_COL`` parquet file (as
        written by :func:`fisseq_data_pipeline.generatesplit.main`)
        naming a subset of cell-level rows to aggregate over (e.g. one
        pseudo-replicate half). When ``None``, all rows are aggregated.
        Defaults to ``None``.
    downsample_wt : float, int, or None
        Optional downsampling of control (wildtype) rows before aggregation.
        A float in ``(0, 1)`` keeps that fraction of control rows; an int
        keeps that many. ``None`` disables downsampling. Defaults to
        ``None``.
    seed : int
        Random seed for the ``downsample_wt`` draw. Ignored when
        ``downsample_wt`` is ``None``. Defaults to ``0``.
    """

    aggregator: str = MISSING
    index_file: Optional[str] = None
    downsample_wt: Optional[Union[float, int]] = None
    seed: int = 0


_cs.store(name="aggregate_feature_type_main", node=FeatureTypeAggregateConfig)


@hydra.main(
    version_base=None, config_path=None, config_name="aggregate_feature_type_main"
)
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: aggregate cell-level features for one feature type.

    ``input_file`` is interpreted as a glob pattern via :func:`load_batches`
    (a concrete non-glob path is a single-file pattern). Rows are optionally
    filtered to ``index_file`` via :func:`.utils.splits.filter_by_index_file`.
    Runs the configured single aggregator via
    :func:`fisseq_data_pipeline.aggregate.aggregate` and writes a lean output
    containing only ``[label_column] + <feature type's stat columns>`` — no
    normalizer, no metadata join, no impact score (those happen once, later,
    in the final feature-selection stage).

    Relies on the ``meta_is_control`` column already present on the input
    (set upstream by ``normalize.py``'s WT-based ``control_sample_query``) as
    the aggregator's reference/control group; does not call
    :func:`fisseq_data_pipeline.aggregate.variant_classification`.

    Output path
    -----------
    - Glob input: ``{output_root}.output.parquet`` or ``{output_dir}/output.parquet``
    - Single-file input: ``{output_root}.{stem}.parquet`` or
      ``{output_dir}/{stem}.parquet``

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.aggregatefeaturetype \\
            output_dir=./out \\
            input_file=data/normalized.parquet \\
            aggregator=mean \\
            index_file=./half1.parquet \\
            downsample_wt=0.5 \\
            seed=1
    """
    ft_cfg: FeatureTypeAggregateConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(ft_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ft_cfg.output_dir = output_dir
    setup_logging(ft_cfg, "aggregate_feature_type")

    logging.info("Loading input from %s", ft_cfg.input_file)
    lf, output_stem = load_batches(ft_cfg.input_file)

    logging.info("Filtering by index_file=%s", ft_cfg.index_file)
    lf = filter_by_index_file(lf, ft_cfg.index_file)

    if ft_cfg.downsample_wt is not None:
        if isinstance(ft_cfg.downsample_wt, float) and not (
            0 < ft_cfg.downsample_wt < 1
        ):
            raise ValueError(
                f"downsample_wt float must satisfy 0 < x < 1, got {ft_cfg.downsample_wt}"
            )
        if isinstance(ft_cfg.downsample_wt, int) and ft_cfg.downsample_wt <= 0:
            raise ValueError(
                f"downsample_wt int must be positive, got {ft_cfg.downsample_wt}"
            )
        logging.info(
            "Downsampling control rows: downsample_wt=%s, seed=%d",
            ft_cfg.downsample_wt,
            ft_cfg.seed,
        )
        lf = downsample_control(lf, ft_cfg.downsample_wt, ft_cfg.seed)

    logging.info("Running %s aggregator", ft_cfg.aggregator)
    agg_lf = aggregate(
        lf,
        label_col=ft_cfg.label_column,
        aggregator_name=ft_cfg.aggregator,
    )

    if ft_cfg.output_root is not None:
        out_path = pathlib.Path(f"{ft_cfg.output_root}.{output_stem}.parquet")
    else:
        out_path = output_dir / f"{output_stem}.parquet"

    logging.info("Writing output to %s", out_path)
    agg_lf.sink_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
