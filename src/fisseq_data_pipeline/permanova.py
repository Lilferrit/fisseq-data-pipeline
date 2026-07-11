import dataclasses
import functools
import logging
import pathlib
import traceback
from typing import Optional

import hydra
import numpy as np
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .config import LabeledInputConfig
from .utils.batches import load_batches
from .utils.constants import FEATURE_SELECTOR, META_BATCH_COL
from .utils.log import setup_logging
from .utils.metadata import get_aggregate_meta_data
from .utils.vectors import COSINE_DIST_COL, compute_cosine_distance

_cs = ConfigStore.instance()


@dataclasses.dataclass
class PermanovaConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the PERMANOVA entry point.

    Extends :class:`.config.LabeledInputConfig` with parameters controlling
    per-variant PERMANOVA batch-effect assessment.

    ``input_file`` is interpreted as a glob pattern. Each matching file is
    treated as a separate batch; the batch name is the filename stem.

    Attributes
    ----------
    n_permutations : int
        Number of label permutations used to estimate the p-value for each
        variant. ``0`` skips the permutation test entirely (``p_value`` is
        ``None``). Defaults to ``999``.
    seed : int
        Base random seed for label permutation; variant ``i`` (in sorted
        order) uses ``seed + i``. Defaults to ``42``.
    """

    n_permutations: int = 999
    seed: int = 42


_cs.store(name="permanova_main", node=PermanovaConfig)

_TMP_IDX = "__row_idx__"
_PAIRS_COL = "__pairs__"
_N_COL = "__n__"
_A_COL = "__a__"
_VARIANT_IDX_COL = "__variant_idx__"


def _f_statistic(
    d2: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    group_of_sample: np.ndarray,
    group_sizes: np.ndarray,
    n: int,
    a: int,
) -> float:
    """
    Compute the PERMANOVA pseudo-F statistic via the Anderson (2001)
    sum-of-squares decomposition.

    Parameters
    ----------
    d2 : np.ndarray
        Squared distances, one per unordered sample pair.
    idx_a, idx_b : np.ndarray
        Sample indices (into ``group_of_sample``/``group_sizes``) for each
        pair in ``d2``.
    group_of_sample : np.ndarray
        Integer group (batch) label for each sample, aligned with sample index.
    group_sizes : np.ndarray
        Number of samples in each group, indexed by group label.
    n : int
        Total number of samples.
    a : int
        Number of groups (batches).

    Returns
    -------
    float
        Pseudo-F statistic. May be ``nan``/``inf`` for degenerate inputs
        (e.g. every group has exactly one member).
    """
    ss_total = d2.sum() / n
    group_a = group_of_sample[idx_a]
    group_b = group_of_sample[idx_b]
    same = group_a == group_b
    weights = 1.0 / group_sizes[group_a]
    ss_within = (d2[same] * weights[same]).sum()
    ss_between = ss_total - ss_within
    return (ss_between / (a - 1)) / (ss_within / (n - a))


def compute_variant_permanova(
    variant_lf: pl.LazyFrame,
    feature_cols: list[str],
    batch_col: str,
    n_permutations: int,
    seed: int,
) -> Optional[dict]:
    """
    Compute the PERMANOVA pseudo-F statistic (and optional p-value) for one variant.

    Forms all unique unordered sample pairs (including cross-batch pairs) via
    a self cross-join, computes cosine distance per pair with
    :func:`.utils.vectors.compute_cosine_distance`, and derives the F-statistic from
    the sum-of-squares decomposition. Null, ``NaN``, and infinite feature values
    are handled by :func:`.utils.vectors.compute_cosine_distance` itself (excluded
    per pair rather than dropping whole rows/samples).

    Parameters
    ----------
    variant_lf : pl.LazyFrame
        Cell-level lazy frame already filtered to a single variant, containing
        ``feature_cols`` and ``batch_col``.
    feature_cols : list[str]
        Names of the feature columns to use for cosine distance.
    batch_col : str
        Name of the batch grouping column.
    n_permutations : int
        Number of label permutations for the p-value. ``0`` skips the test.
    seed : int
        Random seed for label permutation.

    Returns
    -------
    dict or None
        ``{"f_statistic": float, "p_value": float or None}``, or ``None`` (with
        a logged warning) if fewer than 2 samples or fewer than 2 batches
        are present.
    """
    group_counts_df = variant_lf.group_by(batch_col).agg(pl.len().alias("_n")).collect()
    n = int(group_counts_df.get_column("_n").sum())
    a = group_counts_df.height

    if n < 2 or a < 2:
        logging.warning(
            "Skipping variant: %d sample(s) across %d batch(es) "
            "(need >= 2 samples and >= 2 batches)",
            n,
            a,
        )
        return None

    indexed_lf = variant_lf.select([*feature_cols, batch_col]).with_row_index(_TMP_IDX)
    pairs_lf = indexed_lf.join(indexed_lf, how="cross", suffix="_b").filter(
        pl.col(_TMP_IDX) < pl.col(f"{_TMP_IDX}_b")
    )
    dist_lf = compute_cosine_distance(pairs_lf, feature_cols, suffix="_b")
    pair_df = dist_lf.select(
        pl.col(_TMP_IDX).alias("idx_a"),
        pl.col(f"{_TMP_IDX}_b").alias("idx_b"),
        pl.col(batch_col).alias("batch_a"),
        pl.col(f"{batch_col}_b").alias("batch_b"),
        pl.col(COSINE_DIST_COL).alias("dist"),
    ).collect()

    idx_a = pair_df.get_column("idx_a").to_numpy()
    idx_b = pair_df.get_column("idx_b").to_numpy()
    d2 = pair_df.get_column("dist").to_numpy() ** 2

    all_idx = np.concatenate([idx_a, idx_b])
    all_labels = np.concatenate(
        [
            pair_df.get_column("batch_a").cast(pl.Utf8).to_numpy(),
            pair_df.get_column("batch_b").cast(pl.Utf8).to_numpy(),
        ]
    )
    sample_labels = np.empty(n, dtype=all_labels.dtype)
    sample_labels[all_idx] = all_labels

    _, group_of_sample, group_sizes = np.unique(
        sample_labels, return_inverse=True, return_counts=True
    )

    f_obs = _f_statistic(d2, idx_a, idx_b, group_of_sample, group_sizes, n, a)

    if n_permutations <= 0:
        logging.info("f_statistic=%.4f, p_value=None", f_obs)
        return {"f_statistic": f_obs, "p_value": None}

    rng = np.random.default_rng(seed)
    count_ge = 0
    for _ in range(n_permutations):
        perm_group = rng.permutation(group_of_sample)
        f_perm = _f_statistic(d2, idx_a, idx_b, perm_group, group_sizes, n, a)
        if f_perm >= f_obs:
            count_ge += 1
    p_value = (count_ge + 1) / (n_permutations + 1)

    logging.info("f_statistic=%.4f, p_value=%.4f", f_obs, p_value)
    return {"f_statistic": f_obs, "p_value": p_value}


def _compute_variant_stats_from_pairs(
    row: dict, n_permutations: int, seed: int
) -> dict:
    """
    ``map_elements`` callback: PERMANOVA f-statistic/p-value for one variant's
    pre-aggregated sample pairs.

    Mirrors the second half of :func:`compute_variant_permanova` (sample-label
    reconstruction, :func:`_f_statistic`, permutation loop), but reads its
    inputs from a dict of already-materialized Python lists (as handed to it
    by Polars per aggregated row) instead of collecting a per-variant
    LazyFrame itself. This lets ``main`` compute every variant's pairs and
    statistics in a single lazy query instead of one query per variant.

    Parameters
    ----------
    row : dict
        One aggregated row with keys ``_N_COL``, ``_A_COL``,
        ``_VARIANT_IDX_COL``, and ``_PAIRS_COL`` (a list of
        ``{"idx_a", "idx_b", "batch_a", "batch_b", "dist"}`` dicts, one per
        unordered sample pair).
    n_permutations : int
        Number of label permutations for the p-value. ``0`` skips the test.
    seed : int
        Base random seed; this variant's trial uses ``seed + row[_VARIANT_IDX_COL]``.

    Returns
    -------
    dict
        ``{"f_statistic": float or None, "p_value": float or None}``.
        Both are ``None`` if fewer than 2 samples or fewer than 2 batches are
        present, or if computation raised an exception (logged as a warning).
    """
    try:
        n, a = row[_N_COL], row[_A_COL]
        if n < 2 or a < 2:
            logging.warning(
                "Skipping variant: %d sample(s) across %d batch(es) "
                "(need >= 2 samples and >= 2 batches)",
                n,
                a,
            )
            return {"f_statistic": None, "p_value": None}

        pairs = row[_PAIRS_COL]
        idx_a = np.array([p["idx_a"] for p in pairs], dtype=np.int64)
        idx_b = np.array([p["idx_b"] for p in pairs], dtype=np.int64)
        d2 = np.array([p["dist"] for p in pairs], dtype=np.float64) ** 2
        all_idx = np.concatenate([idx_a, idx_b])
        all_labels = np.concatenate(
            [
                np.array([p["batch_a"] for p in pairs]),
                np.array([p["batch_b"] for p in pairs]),
            ]
        )
        sample_labels = np.empty(n, dtype=all_labels.dtype)
        sample_labels[all_idx] = all_labels
        _, group_of_sample, group_sizes = np.unique(
            sample_labels, return_inverse=True, return_counts=True
        )

        f_obs = _f_statistic(d2, idx_a, idx_b, group_of_sample, group_sizes, n, a)

        if n_permutations <= 0:
            logging.info("f_statistic=%.4f, p_value=None", f_obs)
            return {"f_statistic": f_obs, "p_value": None}

        rng = np.random.default_rng(seed + row[_VARIANT_IDX_COL])
        count_ge = 0
        for _ in range(n_permutations):
            perm_group = rng.permutation(group_of_sample)
            f_perm = _f_statistic(d2, idx_a, idx_b, perm_group, group_sizes, n, a)
            if f_perm >= f_obs:
                count_ge += 1
        p_value = (count_ge + 1) / (n_permutations + 1)

        logging.info("f_statistic=%.4f, p_value=%.4f", f_obs, p_value)
        return {"f_statistic": f_obs, "p_value": p_value}
    except Exception:
        logging.warning(
            "Failed to compute PERMANOVA for variant, skipping:\n%s",
            traceback.format_exc(),
        )
        return {"f_statistic": None, "p_value": None}


@hydra.main(version_base=None, config_path=None, config_name="permanova_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: per-variant PERMANOVA batch-effect assessment.

    Steps
    -----
    1. Glob ``input_file`` to find batch files; each file's stem becomes its
       batch label (added as ``meta_batch``).
    2. In a single lazy query: self-join the frame on the label column to form
       every unordered same-variant sample pair, compute per-pair cosine
       distance, and aggregate each variant's pairs into a list-of-structs
       column. A single ``map_elements`` call
       (:func:`_compute_variant_stats_from_pairs`) then computes the PERMANOVA
       pseudo-F statistic (and optional permutation p-value) per variant, reusing
       :func:`_f_statistic`. Variants with fewer than 2 samples/batches, or that
       raise an exception, are dropped with a warning.
    3. Join per-variant metadata from :func:`.utils.metadata.get_aggregate_meta_data`
       and write results to a Parquet file via ``sink_parquet``.

    Output files
    ------------
    - ``{prefix}permanova.parquet`` — one row per variant, with metadata
      columns, ``f_statistic``, and ``p_value``.

    where ``prefix`` is ``{output_root}.`` when ``output_root`` is set,
    otherwise empty.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.permanova \\
            output_dir=./out \\
            'input_file=data/batches/*.parquet' \\
            n_permutations=999
    """
    perm_cfg: PermanovaConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(perm_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    perm_cfg.output_dir = output_dir
    setup_logging(perm_cfg, "permanova")

    label_col = perm_cfg.label_column
    batch_col = META_BATCH_COL

    lf, _ = load_batches(perm_cfg.input_file)

    feature_cols = list(lf.select(FEATURE_SELECTOR).collect_schema().names())
    logging.info("Using %d feature column(s)", len(feature_cols))

    # Per-variant sample/batch counts; also used to filter to variants seen in
    # more than one batch (mirrors compute_variant_permanova's own bail-out).
    variant_stats_lf = (
        lf.group_by(label_col)
        .agg(pl.len().alias(_N_COL), pl.col(batch_col).n_unique().alias(_A_COL))
        .filter(pl.col(label_col).is_not_null())
    )
    qualifying_lf = variant_stats_lf.filter(pl.col(_A_COL) > 1).select(label_col)

    # Restrict to qualifying variants and assign a per-variant-local row index.
    indexed_lf = (
        lf.join(qualifying_lf, on=label_col, how="semi")
        .select([label_col, batch_col, *feature_cols])
        .with_columns(pl.int_range(pl.len()).over(label_col).alias(_TMP_IDX))
    )

    # Self merge on the variant column: joining on label_col restricts pairs
    # to same-variant rows. Filter by index to keep each unordered pair once.
    pairs_lf = indexed_lf.join(indexed_lf, on=label_col, suffix="_b").filter(
        pl.col(_TMP_IDX) < pl.col(f"{_TMP_IDX}_b")
    )
    dist_lf = compute_cosine_distance(pairs_lf, feature_cols, suffix="_b")

    # Aggregate each variant's pairs into a single list-of-structs column.
    pairs_agg_lf = (
        dist_lf.select(
            label_col,
            pl.col(_TMP_IDX).alias("idx_a"),
            pl.col(f"{_TMP_IDX}_b").alias("idx_b"),
            pl.col(batch_col).cast(pl.Utf8).alias("batch_a"),
            pl.col(f"{batch_col}_b").cast(pl.Utf8).alias("batch_b"),
            pl.col(COSINE_DIST_COL).alias("dist"),
        )
        .group_by(label_col)
        .agg(
            pl.struct(["idx_a", "idx_b", "batch_a", "batch_b", "dist"]).alias(
                _PAIRS_COL
            )
        )
        .join(variant_stats_lf, on=label_col, how="inner")
        .with_columns(
            (pl.col(label_col).rank(method="dense").cast(pl.Int64) - 1).alias(
                _VARIANT_IDX_COL
            )
        )
    )

    # Single map_elements call computes f_statistic/p_value per variant.
    stats_lf = (
        pairs_agg_lf.with_columns(
            pl.struct([label_col, _PAIRS_COL, _N_COL, _A_COL, _VARIANT_IDX_COL])
            .map_elements(
                functools.partial(
                    _compute_variant_stats_from_pairs,
                    n_permutations=perm_cfg.n_permutations,
                    seed=perm_cfg.seed,
                ),
                return_dtype=pl.Struct(
                    {"f_statistic": pl.Float64, "p_value": pl.Float64}
                ),
            )
            .alias("__stats__")
        )
        .unnest("__stats__")
        .select([label_col, "f_statistic", "p_value"])
        .filter(pl.col("f_statistic").is_not_null())
    )

    meta_lf = get_aggregate_meta_data(lf, label_col)
    output_lf = stats_lf.join(meta_lf, on=label_col, how="inner")

    prefix = f"{perm_cfg.output_root}." if perm_cfg.output_root is not None else ""
    out_path = output_dir / f"{prefix}permanova.parquet"
    logging.info("Writing results to %s", out_path)
    output_lf.sink_parquet(out_path)
    logging.info("Done")


if __name__ == "__main__":
    main()
