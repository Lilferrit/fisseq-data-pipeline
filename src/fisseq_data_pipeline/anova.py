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

    Extends :class:`.config.LabeledInputConfig` with parameters controlling
    per-feature one-way ANOVA batch-effect assessment.

    ``input_file`` is interpreted as a glob pattern. Each matching file is
    treated as a separate batch; the batch name is the filename stem.

    Attributes
    ----------
    feature_batch_size : Optional[int]
        If set, feature columns are collected and processed in chunks of at
        most this many columns at a time, each chunk collected in its own
        query (alongside the batch column, so row order stays aligned
        within that chunk) instead of collecting every feature column into
        one large in-memory frame up front. Trades one large collect for
        several smaller ones to bound peak memory when there are many
        feature columns; does not change results. Must be a positive
        integer if set. Defaults to ``None`` (collect all feature columns
        in a single query).
    """

    feature_batch_size: Optional[int] = None


_cs.store(name="anova_main", node=AnovaConfig)


def _f_statistic(
    sum_g: np.ndarray,
    sumsq_g: np.ndarray,
    n_g: np.ndarray,
    n: int,
    a: int,
) -> float:
    """
    Compute the one-way ANOVA F-statistic from per-group sufficient statistics.

    Parameters
    ----------
    sum_g : np.ndarray
        Per-group sum of values, indexed by group label.
    sumsq_g : np.ndarray
        Per-group sum of squared values, indexed by group label.
    n_g : np.ndarray
        Number of samples in each group, indexed by group label.
    n : int
        Total number of samples.
    a : int
        Number of groups (batches).

    Returns
    -------
    float
        F-statistic. May be ``nan``/``inf`` for degenerate inputs (e.g. every
        group has exactly one member, so within-group degrees of freedom is 0).
    """
    ss_within = (sumsq_g - sum_g**2 / n_g).sum()
    ss_total = sumsq_g.sum() - sum_g.sum() ** 2 / n
    ss_between = ss_total - ss_within
    return (ss_between / (a - 1)) / (ss_within / (n - a))


def _compute_anova_stats(
    x: np.ndarray,
    batch_labels: np.ndarray,
) -> Optional[dict]:
    """
    Compute the one-way ANOVA F-statistic and closed-form p-value for one
    feature's already-collected values and batch labels.

    Computes per-batch-group sum, sum of squares, and count in a single pass
    (via ``np.bincount`` over the group coding), derives the F-statistic via
    :func:`_f_statistic`'s sum-of-squares decomposition, and the p-value via
    the F-distribution survival function (``scipy.stats.f.sf``).

    Parameters
    ----------
    x : np.ndarray
        Length-n array of one feature's values.
    batch_labels : np.ndarray
        Length-n array of batch identifiers.

    Returns
    -------
    dict or None
        ``{"f_statistic": float, "p_value": float}``, or ``None`` (with a
        logged warning) if fewer than 2 samples or fewer than 2 batches are
        present.
    """
    n = x.shape[0]
    _, group_of_sample, group_sizes = np.unique(
        batch_labels, return_inverse=True, return_counts=True
    )
    a = group_sizes.shape[0]

    if n < 2 or a < 2:
        logging.warning(
            "Skipping feature: %d sample(s) across %d batch(es) "
            "(need >= 2 samples and >= 2 batches)",
            n,
            a,
        )
        return None

    sum_g = np.bincount(group_of_sample, weights=x, minlength=a)
    sumsq_g = np.bincount(group_of_sample, weights=x**2, minlength=a)

    f_obs = _f_statistic(sum_g, sumsq_g, group_sizes, n, a)
    p_value = float(scipy.stats.f.sf(f_obs, dfn=a - 1, dfd=n - a))

    logging.info("f_statistic=%.4f, p_value=%.4f", f_obs, p_value)
    return {"f_statistic": f_obs, "p_value": p_value}


def compute_feature_anova(
    feature_lf: pl.LazyFrame,
    feature_col: str,
    batch_col: str,
) -> Optional[dict]:
    """
    Compute the one-way ANOVA F-statistic and p-value for one feature.

    Collects the feature's values and batch labels to NumPy and delegates to
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
        a logged warning) if fewer than 2 samples or fewer than 2 batches
        are present.
    """
    collected = feature_lf.select(
        pl.col(feature_col).cast(pl.Float64),
        pl.col(batch_col).cast(pl.Utf8),
    ).collect()
    x = collected.get_column(feature_col).to_numpy()
    batch_labels = collected.get_column(batch_col).to_numpy()

    return _compute_anova_stats(x, batch_labels)


def _iter_feature_batches(
    filtered_lf: pl.LazyFrame,
    feature_cols: list[str],
    batch_col: str,
    feature_batch_size: Optional[int],
):
    """
    Yield ``(collected, chunk)`` pairs covering all of ``feature_cols``.

    ``filtered_lf`` must already carry row-level filters but must not yet be
    column-selected. If ``feature_batch_size`` is ``None``, all of
    ``feature_cols`` are collected together in a single query (one chunk) --
    today's default behavior. Otherwise ``feature_cols`` is split into
    chunks of at most ``feature_batch_size`` columns, each collected in its
    own query.

    ``batch_col`` is always selected alongside a chunk's feature columns in
    the same ``.collect()`` call: Polars does not guarantee that two
    separate ``.collect()`` calls on the same lazy plan return rows in the
    same order, so a ``batch_col`` array collected separately would not be
    reliably row-aligned with a chunk's feature values. Two different
    chunks' frames are not guaranteed to share row order with each other --
    that's fine, since each feature's computation only ever reads from its
    own chunk's frame.

    Note: each chunk re-runs the filter in ``filtered_lf`` from scratch, so
    a smaller ``feature_batch_size`` trades more redundant filtering compute
    for lower peak memory -- this should not be "optimized" into a single
    shared collect, which would reintroduce the row-order-misalignment bug
    this design avoids.

    Parameters
    ----------
    filtered_lf : pl.LazyFrame
        Row-filtered (not yet column-selected) lazy frame containing
        ``batch_col`` and every column in ``feature_cols``.
    feature_cols : list[str]
        Feature column names, in the order they should appear in output.
    batch_col : str
        Name of the batch grouping column.
    feature_batch_size : Optional[int]
        Maximum number of feature columns per chunk, or ``None`` for a
        single chunk containing all of ``feature_cols``.

    Yields
    ------
    tuple[pl.DataFrame, list[str]]
        ``(collected, chunk)`` where ``collected`` contains ``batch_col``
        cast to ``Utf8`` plus every column in ``chunk`` cast to ``Float64``,
        and ``chunk`` is the ordered, non-overlapping slice of
        ``feature_cols`` collected in this iteration.

    Raises
    ------
    ValueError
        If ``feature_batch_size`` is not ``None`` and is <= 0.
    """
    if feature_batch_size is not None and feature_batch_size <= 0:
        raise ValueError(
            f"feature_batch_size must be a positive integer, got {feature_batch_size}"
        )
    if not feature_cols:
        return
    size = feature_batch_size if feature_batch_size is not None else len(feature_cols)
    for i in range(0, len(feature_cols), size):
        chunk = feature_cols[i : i + size]
        collected = filtered_lf.select(
            pl.col(batch_col).cast(pl.Utf8),
            *[pl.col(c).cast(pl.Float64) for c in chunk],
        ).collect()
        yield collected, chunk


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
       does not end with ``":downsampled"``. Feature columns are collected
       from this restriction in chunks of ``feature_batch_size`` columns at
       a time (or all at once if ``feature_batch_size`` is unset) via
       :func:`_iter_feature_batches`.
    3. For each feature column, compute the one-way ANOVA F-statistic and
       closed-form p-value via :func:`_compute_anova_stats`. Features with
       fewer than 2 samples/batches after filtering, or that raise an
       exception, are skipped with a logged warning rather than aborting
       the whole run.
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
            'input_file=data/batches/*.parquet' \\
            feature_batch_size=50
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
    # with a ":downsampled" suffix. Not collected here: _iter_feature_batches
    # below collects batch_col alongside each feature chunk in the same
    # query, since Polars does not guarantee row order matches across
    # separate .collect() calls on the same lazy plan. No separate "seen in
    # >1 batch" prequalification is needed: that check is inherent to
    # _compute_anova_stats's own n<2 / a<2 bail-out, applied per feature.
    filtered_lf = lf.filter(
        pl.col(label_col).map_elements(
            lambda v: classify_variant(v) == "Synonymous", return_dtype=pl.Boolean
        )
        & ~pl.col(label_col).str.ends_with(":downsampled")
    )

    # Per-feature loop, processed in chunks of feature_batch_size columns
    # (or all columns at once if unset). feature_batch_size is purely a
    # memory/compute trade-off and never changes results.
    records: list[dict] = []
    for collected, chunk in _iter_feature_batches(
        filtered_lf, feature_cols, batch_col, anova_cfg.feature_batch_size
    ):
        batch_labels = collected.get_column(batch_col).to_numpy()
        for feature_col in chunk:
            try:
                x = collected.get_column(feature_col).to_numpy()
                result = _compute_anova_stats(x, batch_labels)
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
