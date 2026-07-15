"""Per-variant PERMANOVA batch-effect assessment via pairwise cosine distance.

Hydra entry point ``fisseq-permanova``, backing the Nextflow process ``PERMANOVA``
(run once against normalized cells, once against batch-corrected cells). For each
variant seen in more than one batch, computes a pseudo-F statistic from all
pairwise cosine distances (Anderson 2001 sum-of-squares decomposition) and an
optional permutation p-value.
"""

import dataclasses
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

_N_COL = "__n__"
_A_COL = "__a__"


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


def _pairwise_cosine_distance(X: np.ndarray) -> np.ndarray:
    """
    Compute the full symmetric n x n cosine-distance matrix for a feature
    matrix ``X`` (n samples x p features), replicating
    :func:`.utils.vectors.compute_cosine_distance`'s per-pair, per-dimension
    null/``NaN``/infinite-value masking without materializing an n^2 x p
    intermediate.

    For a pair of rows (i, j), feature dimension k contributes to the dot
    product and both norms only if ``X[i, k]`` and ``X[j, k]`` are both
    finite; excluded dimensions contribute 0. If a row's resulting norm is 0
    (e.g. every dimension was excluded for that pair), it is treated as 1 so
    the division does not produce ``NaN`` -- matching
    :func:`.utils.vectors.compute_cosine_distance`'s zero-norm guard exactly,
    including the edge case where every dimension is excluded (distance is
    then exactly ``1.0``).

    Derivation: let ``M`` be the 0/1 finite-value mask, ``Z`` the
    zero-filled values, and ``W`` the zero-filled squared values (all n x p).
    Then for pair (i, j), ``dot = sum_k M[i,k]*M[j,k]*X[i,k]*X[j,k] =
    (Z @ Z.T)[i,j]`` (since a zeroed factor already contributes 0), and
    ``norm_a^2 = sum_k M[i,k]*M[j,k]*X[i,k]**2 = sum_k M[j,k]*W[i,k] =
    (W @ M.T)[i,j]``, with ``norm_b^2`` (the pair's other side) equal to
    ``M @ W.T = (W @ M.T).T`` by symmetry. This reduces the computation to
    three O(n^2 x p)-time, O(n^2)-memory matrix multiplications instead of
    an O(n^2 x p)-memory pairwise join.

    Parameters
    ----------
    X : np.ndarray
        n x p array of feature values; non-finite entries (``NaN``, ``inf``)
        are treated as missing.

    Returns
    -------
    np.ndarray
        n x n symmetric matrix of cosine distances.
    """
    finite = np.isfinite(X)
    mask = finite.astype(np.float64)
    zeroed = np.where(finite, X, 0.0)
    squared = np.where(finite, X * X, 0.0)

    dot = zeroed @ zeroed.T
    norm_a_sq = squared @ mask.T
    norm_b_sq = norm_a_sq.T

    norm_a = np.sqrt(norm_a_sq)
    norm_b = np.sqrt(norm_b_sq)
    safe_norm_a = np.where(norm_a == 0.0, 1.0, norm_a)
    safe_norm_b = np.where(norm_b == 0.0, 1.0, norm_b)

    return 1.0 - dot / (safe_norm_a * safe_norm_b)


def _compute_permanova_stats(
    X: np.ndarray,
    batch_labels: np.ndarray,
    n_permutations: int,
    seed: int,
) -> Optional[dict]:
    """
    Compute the PERMANOVA pseudo-F statistic (and optional p-value) for one
    variant's already-collected feature matrix and batch labels.

    Computes the full pairwise cosine-distance matrix via
    :func:`_pairwise_cosine_distance`, extracts the condensed upper triangle
    as unordered sample pairs, and derives the F-statistic via
    :func:`_f_statistic`'s sum-of-squares decomposition.

    Parameters
    ----------
    X : np.ndarray
        n x p feature matrix for one variant. Non-finite entries are
        treated as missing (handled per-pair, per-dimension).
    batch_labels : np.ndarray
        Length-n array of batch identifiers.
    n_permutations : int
        Number of label permutations used to estimate the p-value. ``0``
        skips the permutation test entirely (``p_value`` is ``None``).
    seed : int
        Random seed for label permutation.

    Returns
    -------
    dict or None
        ``{"f_statistic": float, "p_value": float or None}``, or ``None``
        (with a logged warning) if fewer than 2 samples or fewer than 2
        batches are present.
    """
    n = X.shape[0]
    _, group_of_sample, group_sizes = np.unique(
        batch_labels, return_inverse=True, return_counts=True
    )
    a = group_sizes.shape[0]

    if n < 2 or a < 2:
        logging.warning(
            "Skipping variant: %d sample(s) across %d batch(es) "
            "(need >= 2 samples and >= 2 batches)",
            n,
            a,
        )
        return None

    cosine_dist = _pairwise_cosine_distance(X)
    idx_a, idx_b = np.triu_indices(n, k=1)
    d2 = cosine_dist[idx_a, idx_b] ** 2

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


def compute_variant_permanova(
    variant_lf: pl.LazyFrame,
    feature_cols: list[str],
    batch_col: str,
    n_permutations: int,
    seed: int,
) -> Optional[dict]:
    """
    Compute the PERMANOVA pseudo-F statistic (and optional p-value) for one variant.

    Collects the variant's feature matrix and batch labels to NumPy and
    delegates to :func:`_compute_permanova_stats`, which computes all unique
    unordered sample pairs (including cross-batch pairs) via
    :func:`_pairwise_cosine_distance` and derives the F-statistic from the
    sum-of-squares decomposition. Null, ``NaN``, and infinite feature values
    are excluded per pair rather than dropping whole rows/samples.

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
    collected = variant_lf.select(
        [pl.col(c).cast(pl.Float64) for c in feature_cols]
        + [pl.col(batch_col).cast(pl.Utf8)]
    ).collect()
    X = collected.select(feature_cols).to_numpy()
    batch_labels = collected.get_column(batch_col).to_numpy()

    return _compute_permanova_stats(X, batch_labels, n_permutations, seed)


@hydra.main(version_base=None, config_path=None, config_name="permanova_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: per-variant PERMANOVA batch-effect assessment.

    Steps
    -----
    1. Glob ``input_file`` to find batch files; each file's stem becomes its
       batch label (added as ``meta_batch``).
    2. Restrict to variants seen in more than one batch, collect each such
       variant's feature matrix and batch labels to NumPy in turn, and
       compute the PERMANOVA pseudo-F statistic (and optional permutation
       p-value) via :func:`_compute_permanova_stats`. Variants with fewer
       than 2 samples/batches, or that raise an exception, are skipped with
       a logged warning rather than aborting the whole run.
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

    # Per-variant sample/batch counts; used to filter to variants seen in
    # more than one batch (mirrors _compute_permanova_stats's own bail-out).
    variant_stats_lf = (
        lf.group_by(label_col)
        .agg(pl.len().alias(_N_COL), pl.col(batch_col).n_unique().alias(_A_COL))
        .filter(pl.col(label_col).is_not_null())
    )
    qualifying_lf = variant_stats_lf.filter(pl.col(_A_COL) > 1).select(label_col)

    # Collect once, restricted to qualifying variants and pruned to the
    # columns needed, so each variant's feature matrix can be sliced out in
    # the loop below without re-scanning the (possibly multi-file) input.
    restricted_df = (
        lf.join(qualifying_lf, on=label_col, how="semi")
        .select(
            pl.col(label_col),
            pl.col(batch_col).cast(pl.Utf8),
            *[pl.col(c).cast(pl.Float64) for c in feature_cols],
        )
        .sort(label_col)
        .collect()
    )

    # Serial per-variant loop: each variant's pairwise distance computation
    # is now O(n^2) in memory (not O(n^2 * p) fused across all variants), so
    # no batching/parallelism is needed here.
    records: list[dict] = []
    for variant_idx, variant_df in enumerate(
        restricted_df.partition_by(label_col, maintain_order=True)
    ):
        label = variant_df.get_column(label_col)[0]
        try:
            X = variant_df.select(feature_cols).to_numpy()
            batch_labels = variant_df.get_column(batch_col).to_numpy()
            result = _compute_permanova_stats(
                X,
                batch_labels,
                perm_cfg.n_permutations,
                perm_cfg.seed + variant_idx,
            )
        except Exception:
            logging.warning(
                "Failed to compute PERMANOVA for variant %r, skipping:\n%s",
                label,
                traceback.format_exc(),
            )
            continue
        if result is None:
            continue
        records.append(
            {
                label_col: label,
                "f_statistic": result["f_statistic"],
                "p_value": result["p_value"],
            }
        )

    stats_df = pl.DataFrame(
        records,
        schema={
            label_col: restricted_df.schema[label_col],
            "f_statistic": pl.Float64,
            "p_value": pl.Float64,
        },
    )

    meta_lf = get_aggregate_meta_data(lf, label_col)
    output_lf = stats_df.lazy().join(meta_lf, on=label_col, how="inner")

    prefix = f"{perm_cfg.output_root}." if perm_cfg.output_root is not None else ""
    out_path = output_dir / f"{prefix}permanova.parquet"
    logging.info("Writing results to %s", out_path)
    output_lf.sink_parquet(out_path)
    logging.info("Done")


if __name__ == "__main__":
    main()
