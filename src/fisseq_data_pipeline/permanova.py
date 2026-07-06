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
from .utils.vectors import compute_cosine_distance

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
    the sum-of-squares decomposition. Rows with any non-finite feature value
    are dropped before pairing.

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
        remain after dropping non-finite rows.
    """
    finite_mask = pl.all_horizontal(
        pl.col(c).is_not_null() & pl.col(c).is_finite() for c in feature_cols
    )
    indexed_lf = (
        variant_lf.select([*feature_cols, batch_col])
        .filter(finite_mask)
        .with_row_index(_TMP_IDX)
    )
    sample_df = indexed_lf.select(batch_col).collect()
    n = sample_df.height
    labels = sample_df.get_column(batch_col).cast(pl.Utf8).to_numpy()

    _, group_of_sample, group_sizes = np.unique(
        labels, return_inverse=True, return_counts=True
    )
    a = group_sizes.shape[0]

    if n < 2 or a < 2:
        logging.warning(
            "Skipping variant: %d finite sample(s) across %d batch(es) "
            "(need >= 2 samples and >= 2 batches)",
            n,
            a,
        )
        return None

    pairs_lf = indexed_lf.join(indexed_lf, how="cross", suffix="_b").filter(
        pl.col(_TMP_IDX) < pl.col(f"{_TMP_IDX}_b")
    )
    dist_lf = compute_cosine_distance(pairs_lf, feature_cols, suffix="_b")
    pair_df = dist_lf.select(
        pl.col(_TMP_IDX).alias("idx_a"),
        pl.col(f"{_TMP_IDX}_b").alias("idx_b"),
        pl.col("tmp_cosine_distance").alias("dist"),
    ).collect()

    idx_a = pair_df.get_column("idx_a").to_numpy()
    idx_b = pair_df.get_column("idx_b").to_numpy()
    d2 = pair_df.get_column("dist").to_numpy() ** 2

    f_obs = _f_statistic(d2, idx_a, idx_b, group_of_sample, group_sizes, n, a)

    if n_permutations <= 0:
        return {"f_statistic": f_obs, "p_value": None}

    rng = np.random.default_rng(seed)
    count_ge = 0
    for _ in range(n_permutations):
        perm_group = rng.permutation(group_of_sample)
        f_perm = _f_statistic(d2, idx_a, idx_b, perm_group, group_sizes, n, a)
        if f_perm >= f_obs:
            count_ge += 1
    p_value = (count_ge + 1) / (n_permutations + 1)

    return {"f_statistic": f_obs, "p_value": p_value}


@hydra.main(version_base=None, config_path=None, config_name="permanova_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: per-variant PERMANOVA batch-effect assessment.

    Steps
    -----
    1. Glob ``input_file`` to find batch files; each file's stem becomes its
       batch label (added as ``meta_batch``).
    2. For each variant seen in more than one batch, compute the PERMANOVA
       pseudo-F statistic (and optional permutation p-value) via
       :func:`compute_variant_permanova`. Variants that raise an exception are
       skipped with a warning.
    3. Join per-variant metadata from :func:`.utils.metadata.get_aggregate_meta_data`
       and write results to a Parquet file.

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

    variant_batches = (
        lf.group_by(label_col)
        .agg(pl.col(batch_col).n_unique().alias("n_batches"))
        .filter(pl.col("n_batches") > 1)
        .collect()
    )
    variants = sorted(variant_batches.get_column(label_col).drop_nulls().to_list())
    logging.info("Found %d variant(s) with more than one batch", len(variants))

    results = []
    for i, variant in enumerate(variants):
        logging.info("Computing PERMANOVA for variant '%s'", variant)
        variant_lf = lf.filter(pl.col(label_col) == variant)
        try:
            stats = compute_variant_permanova(
                variant_lf,
                feature_cols,
                batch_col,
                n_permutations=perm_cfg.n_permutations,
                seed=perm_cfg.seed + i,
            )
        except Exception:
            logging.warning(
                "Failed to compute PERMANOVA for variant '%s', skipping:\n%s",
                variant,
                traceback.format_exc(),
            )
            continue
        if stats is not None:
            results.append({label_col: variant, **stats})

    results_df = (
        pl.DataFrame(results)
        if results
        else pl.DataFrame(
            schema={
                label_col: pl.Utf8,
                "f_statistic": pl.Float64,
                "p_value": pl.Float64,
            }
        )
    )

    meta_lf = get_aggregate_meta_data(lf, label_col)
    output_df = results_df.join(meta_lf.collect(), on=label_col, how="inner")

    prefix = f"{perm_cfg.output_root}." if perm_cfg.output_root is not None else ""
    out_path = output_dir / f"{prefix}permanova.parquet"
    logging.info("Writing results to %s", out_path)
    output_df.write_parquet(out_path)
    logging.info("Done")


if __name__ == "__main__":
    main()
