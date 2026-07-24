"""Per-feature one-way ANOVA batch-effect assessment via sufficient statistics.

Hydra entry point ``fisseq-anova``, backing the Nextflow process ``ANOVA``
(run once against normalized cells, once against batch-corrected cells). Restricts
to cells classified as Synonymous (via :func:`.utils.variant.classify_variant`)
whose label does not carry a ``:downsampled`` tag, then for each feature column,
computes a one-way ANOVA F-statistic from per-batch-group sufficient statistics
(sum, sum of squares, count) and a closed-form p-value via the F-distribution
survival function.
"""

import dataclasses
import logging
import pathlib
import traceback
from typing import Optional

import hydra
import numpy as np
import polars as pl
import scipy.stats
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .config import LabeledInputConfig
from .utils.batches import load_batches
from .utils.constants import FEATURE_SELECTOR, META_BATCH_COL
from .utils.log import setup_logging
from .utils.variant import classify_variant

_cs = ConfigStore.instance()


@dataclasses.dataclass
class AnovaConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the ANOVA entry point.

    Extends :class:`.config.LabeledInputConfig` with no additional fields;
    kept as a distinct type since it's registered separately with Hydra's
    ``ConfigStore``.

    ``input_file`` is interpreted as a glob pattern. Each matching file is
    treated as a separate batch; the batch name is the filename stem.
    """


_cs.store(name="anova_main", node=AnovaConfig)


def _f_statistic(
    sum_g: np.ndarray,
    sumsq_g: np.ndarray,
    n_g: np.ndarray,
) -> float:
    """
    Compute the one-way ANOVA F-statistic from per-group sufficient statistics.

    ``n``  (total sample count) and ``a`` (number of groups) are derived
    from ``n_g``, so callers must pre-filter ``sum_g``/``sumsq_g``/``n_g``
    to groups with at least one sample (no zero entries), since the formula
    divides elementwise by ``n_g``.

    Parameters
    ----------
    sum_g : np.ndarray
        Per-group sum of values, indexed by group label.
    sumsq_g : np.ndarray
        Per-group sum of squared values, indexed by group label.
    n_g : np.ndarray
        Number of samples in each group, indexed by group label. Must not
        contain zero entries.

    Returns
    -------
    float
        F-statistic. May be ``nan``/``inf`` for degenerate inputs (e.g. every
        group has exactly one member, so within-group degrees of freedom is 0).
    """
    n = n_g.sum()
    a = n_g.shape[0]
    ss_within = (sumsq_g - sum_g**2 / n_g).sum()
    ss_total = sumsq_g.sum() - sum_g.sum() ** 2 / n
    ss_between = ss_total - ss_within
    return (ss_between / (a - 1)) / (ss_within / (n - a))


def _compute_anova_stats(
    n_g: np.ndarray,
    sum_g: np.ndarray,
    sumsq_g: np.ndarray,
) -> Optional[dict]:
    """
    Compute the one-way ANOVA F-statistic and closed-form p-value for one
    feature from its already-aggregated per-batch-group sufficient statistics.

    ``n_g``/``sum_g``/``sumsq_g`` are expected to come from a
    ``group_by(batch_col).agg(...)`` query whose count/sum expressions skip
    nulls, so a group with only null values for this feature shows up as
    ``n_g == 0`` (not null) rather than being silently mixed into other
    groups' statistics. Such groups are excluded here before deriving ``a``
    (number of groups with non-null data) and ``n`` (total non-null sample
    count) -- both computed per feature, since null patterns can differ
    per feature.

    Parameters
    ----------
    n_g : np.ndarray
        Per-batch-group non-null count for this feature.
    sum_g : np.ndarray
        Per-batch-group sum of non-null values for this feature.
    sumsq_g : np.ndarray
        Per-batch-group sum of squared non-null values for this feature.

    Returns
    -------
    dict or None
        ``{"f_statistic": float, "p_value": float}``, or ``None`` (with a
        logged warning) if fewer than 2 non-null samples or fewer than 2
        groups with non-null data are present.
    """
    mask = n_g > 0
    n_g, sum_g, sumsq_g = n_g[mask], sum_g[mask], sumsq_g[mask]
    a = n_g.shape[0]
    n = int(n_g.sum())

    if n < 2 or a < 2:
        logging.warning(
            "Skipping feature: %d non-null sample(s) across %d batch(es) "
            "with non-null data (need >= 2 samples and >= 2 batches)",
            n,
            a,
        )
        return None

    f_obs = _f_statistic(sum_g, sumsq_g, n_g)
    p_value = float(scipy.stats.f.sf(f_obs, dfn=a - 1, dfd=n - a))

    logging.info("f_statistic=%.4f, p_value=%.4f", f_obs, p_value)
    return {"f_statistic": f_obs, "p_value": p_value}


def _group_stat_exprs(feature_col: str, alias_prefix: str) -> list[pl.Expr]:
    """
    Build per-batch-group non-null count/sum/sum-of-squares expressions for
    one feature column, to be used inside a ``group_by(batch_col).agg(...)``.

    Polars' ``.count()``/``.sum()`` skip null values by default, so a group
    with only null values for ``feature_col`` aggregates to ``n=0, sum=0.0``
    rather than null or a value contaminated by nulls.

    Parameters
    ----------
    feature_col : str
        Name of the feature column to aggregate.
    alias_prefix : str
        Prefix for the output column aliases (``f"{alias_prefix}n"``,
        ``f"{alias_prefix}sum"``, ``f"{alias_prefix}sumsq"``). Callers
        aggregating multiple features in one query should pass a unique
        prefix per feature (e.g. an index-based prefix) to avoid alias
        collisions regardless of feature-name contents.

    Returns
    -------
    list[pl.Expr]
        The three aggregation expressions for this feature.
    """
    c = pl.col(feature_col).cast(pl.Float64)
    return [
        c.count().alias(f"{alias_prefix}n"),
        c.sum().alias(f"{alias_prefix}sum"),
        (c**2).sum().alias(f"{alias_prefix}sumsq"),
    ]


def compute_feature_anova(
    feature_lf: pl.LazyFrame,
    feature_col: str,
    batch_col: str,
) -> Optional[dict]:
    """
    Compute the one-way ANOVA F-statistic and p-value for one feature.

    Aggregates per-batch-group non-null count/sum/sum-of-squares for this
    feature in a single ``group_by`` query (one row per batch, so cheap to
    collect regardless of ``feature_lf``'s row count) and delegates to
    :func:`_compute_anova_stats`.

    Parameters
    ----------
    feature_lf : pl.LazyFrame
        Cell-level lazy frame already filtered to the desired rows, containing
        ``feature_col`` and ``batch_col``.
    feature_col : str
        Name of the feature column to test.
    batch_col : str
        Name of the batch grouping column.

    Returns
    -------
    dict or None
        ``{"f_statistic": float, "p_value": float}``, or ``None`` (with
        a logged warning) if fewer than 2 non-null samples or fewer than 2
        groups with non-null data are present.
    """
    grouped = (
        feature_lf.group_by(batch_col).agg(_group_stat_exprs(feature_col, "")).collect()
    )
    n_g = grouped.get_column("n").to_numpy()
    sum_g = grouped.get_column("sum").to_numpy()
    sumsq_g = grouped.get_column("sumsq").to_numpy()

    return _compute_anova_stats(n_g, sum_g, sumsq_g)


@hydra.main(version_base=None, config_path=None, config_name="anova_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: per-feature one-way ANOVA batch-effect assessment.

    Steps
    -----
    1. Glob ``input_file`` to find batch files; each file's stem becomes its
       batch label (added as ``meta_batch``).
    2. Restrict to cells classified as ``"Synonymous"`` (via
       :func:`.utils.variant.classify_variant`) whose ``label_column`` value
       does not end with ``":downsampled"``. From this restriction, a single
       lazy ``group_by(batch_col).agg(...)`` query (built via
       :func:`_group_stat_exprs`, one call per feature column) computes every
       feature's per-batch-group non-null count/sum/sum-of-squares at once;
       the result is one row per batch, so it's cheap to collect in full
       regardless of the input row count.
    3. For each feature column, compute the one-way ANOVA F-statistic and
       closed-form p-value from that feature's aggregated statistics via
       :func:`_compute_anova_stats`. Features with fewer than 2 non-null
       samples/groups after excluding nulls, or that raise an exception, are
       skipped with a logged warning rather than aborting the whole run.
    4. Write one row per feature to a Parquet file via ``sink_parquet``.

    Output files
    ------------
    - ``{prefix}anova.parquet`` — one row per feature, with columns
      ``feature``, ``f_value``, and ``p_value``.

    where ``prefix`` is ``{output_root}.`` when ``output_root`` is set,
    otherwise empty.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.anova \\
            output_dir=./out \\
            'input_file=data/batches/*.parquet'
    """
    anova_cfg: AnovaConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(anova_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    anova_cfg.output_dir = output_dir
    setup_logging(anova_cfg, "anova")

    label_col = anova_cfg.label_column
    batch_col = META_BATCH_COL

    lf, _ = load_batches(anova_cfg.input_file)

    feature_cols = list(lf.select(FEATURE_SELECTOR).collect_schema().names())
    logging.info("Using %d feature column(s)", len(feature_cols))

    # Restrict, lazily, to cells classified as Synonymous and not tagged
    # with a ":downsampled" suffix. No separate "seen in >1 batch"
    # prequalification is needed: that check is inherent to
    # _compute_anova_stats's own n<2 / a<2 bail-out, applied per feature.
    filtered_lf = lf.filter(
        pl.col(label_col).map_elements(
            lambda v: classify_variant(v) == "Synonymous", return_dtype=pl.Boolean
        )
        & ~pl.col(label_col).str.ends_with(":downsampled")
    )

    # Single lazy query: per-batch-group non-null count/sum/sum-of-squares
    # for every feature column at once. The result has one row per batch
    # (small), so it's collected in full regardless of how many cells feed
    # into it.
    agg_exprs = [
        expr
        for i, feature_col in enumerate(feature_cols)
        for expr in _group_stat_exprs(feature_col, f"{i}_")
    ]
    grouped = (
        filtered_lf.group_by(batch_col).agg(agg_exprs).collect()
        if agg_exprs
        else pl.DataFrame()
    )

    records: list[dict] = []
    for i, feature_col in enumerate(feature_cols):
        try:
            n_g = grouped.get_column(f"{i}_n").to_numpy()
            sum_g = grouped.get_column(f"{i}_sum").to_numpy()
            sumsq_g = grouped.get_column(f"{i}_sumsq").to_numpy()
            result = _compute_anova_stats(n_g, sum_g, sumsq_g)
        except Exception:
            logging.warning(
                "Failed to compute ANOVA for feature %r, skipping:\n%s",
                feature_col,
                traceback.format_exc(),
            )
            continue
        if result is None:
            continue
        records.append(
            {
                "feature": feature_col,
                "f_value": result["f_statistic"],
                "p_value": result["p_value"],
            }
        )

    stats_df = pl.DataFrame(
        records,
        schema={"feature": pl.Utf8, "f_value": pl.Float64, "p_value": pl.Float64},
    )

    prefix = f"{anova_cfg.output_root}." if anova_cfg.output_root is not None else ""
    out_path = output_dir / f"{prefix}anova.parquet"
    logging.info("Writing results to %s", out_path)
    stats_df.lazy().sink_parquet(out_path)
    logging.info("Done")


if __name__ == "__main__":
    main()
