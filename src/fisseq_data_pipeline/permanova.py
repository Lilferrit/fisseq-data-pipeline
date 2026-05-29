import dataclasses
import logging
import pathlib
from typing import Optional

import hydra
import numpy as np
import polars as pl
import tqdm
from hydra.core.config_store import ConfigStore
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from skbio.stats.distance import DistanceMatrix, permanova

from .config import LabeledInputConfig
from .constants import META_BATCH_COL
from .utils import classify_variant, load_batches, setup_logging

_cs = ConfigStore.instance()


@dataclasses.dataclass
class PermanovaConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the PERMANOVA entry point.

    Extends :class:`.config.LabeledInputConfig` with parameters controlling
    bootstrap PERMANOVA batch-effect assessment.

    ``input_file`` is interpreted as a glob pattern. Each matching file is
    treated as a separate batch; the batch name is the filename stem.

    Attributes
    ----------
    variant_class_filter : str or None
        If set, only rows whose variant label maps to this class via
        :func:`.utils.classify_variant` are included (e.g. ``"WT"``,
        ``"Synonymous"``). ``None`` uses all rows. Defaults to ``None``.
    n_bootstraps : int
        Number of bootstrap samples to draw. Defaults to ``200``.
    sample_size : int
        Number of rows per bootstrap sample. Defaults to ``1000``.
    seed : int
        Base random seed; each bootstrap uses ``seed + i``. Defaults to ``42``.
    n_jobs : int
        Number of parallel workers when ``parallel`` is ``True``. ``-1`` uses
        all available cores. Defaults to ``-1``.
    parallel : bool
        If ``True``, run bootstraps in parallel via joblib. Defaults to
        ``False``.
    """

    variant_class_filter: Optional[str] = "WT"
    n_bootstraps: int = 200
    sample_size: int = 1000
    seed: int = 42
    n_jobs: int = -1
    parallel: bool = False


_cs.store(name="permanova_main", node=PermanovaConfig)

_TMP_IDX = "__row_idx__"


def cosine_dists_matrix(x: np.ndarray) -> np.ndarray:
    """
    Compute a pairwise cosine distance matrix.

    Zero-norm rows are treated as unit vectors to avoid NaN propagation.
    Off-diagonal values are clipped to ``[0, inf)`` to suppress floating-point
    underflow below zero.

    Parameters
    ----------
    x : np.ndarray
        2-D array of shape ``(n_samples, n_features)``.

    Returns
    -------
    np.ndarray
        Symmetric ``(n_samples, n_samples)`` cosine distance matrix with a
        zero diagonal.
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    x_norm = x / norms
    cos_dist = 1.0 - x_norm @ x_norm.T
    np.fill_diagonal(cos_dist, 0.0)
    np.clip(cos_dist, 0.0, None, out=cos_dist)
    i_lower = np.tril_indices_from(cos_dist, k=-1)
    cos_dist.T[i_lower] = cos_dist[i_lower]
    return cos_dist


def _compute_f_stat(dist_matrix: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the PERMANOVA pseudo-F statistic (no permutation test).

    Parameters
    ----------
    dist_matrix : np.ndarray
        Square symmetric distance matrix.
    labels : np.ndarray
        1-D grouping array aligned with the rows/columns of ``dist_matrix``.

    Returns
    -------
    float
        PERMANOVA pseudo-F statistic.
    """
    result = permanova(DistanceMatrix(dist_matrix), labels, permutations=0)
    return result["test statistic"]


def compute_permanova_sample(
    sampled: pl.DataFrame,
    batch_col: str,
    seed: int,
) -> pl.DataFrame:
    """
    Compute observed and shuffled PERMANOVA F-statistics on a pre-sampled DataFrame.

    The distance matrix is computed once and reused for both the observed and
    null (label-shuffled) statistics.

    Parameters
    ----------
    sampled : pl.DataFrame
        Pre-sampled cell-level DataFrame containing feature columns and
        ``batch_col``. Rows with non-finite feature values are silently dropped.
    batch_col : str
        Name of the batch grouping column.
    seed : int
        Random seed for label shuffling.

    Returns
    -------
    pl.DataFrame
        Single-row DataFrame with columns ``f_value`` (observed) and
        ``f_value_shuffled`` (null).
    """
    feature_cols = [
        c for c in sampled.columns if len(c) > 0 and c[0].isupper() and "_" in c
    ]
    feature_matrix = sampled.select(feature_cols).cast(pl.Float64).to_numpy()
    batch_labels = sampled.get_column(batch_col).cast(pl.Utf8).to_numpy()

    valid_rows = np.isfinite(feature_matrix).all(axis=1)
    feature_matrix = feature_matrix[valid_rows]
    batch_labels = batch_labels[valid_rows]

    if len(np.unique(batch_labels)) < 2:
        return pl.DataFrame(
            {"f_value": [float("nan")], "f_value_shuffled": [float("nan")]}
        )

    dist_matrix = cosine_dists_matrix(feature_matrix)
    f_val = _compute_f_stat(dist_matrix, batch_labels)

    shuffled_labels = batch_labels.copy()
    np.random.default_rng(seed).shuffle(shuffled_labels)
    f_val_shuffled = _compute_f_stat(dist_matrix, shuffled_labels)

    logging.debug("Seed %d: F=%.4f, F_shuffled=%.4f", seed, f_val, f_val_shuffled)
    return pl.DataFrame({"f_value": [f_val], "f_value_shuffled": [f_val_shuffled]})


def bootstrap_permanova(
    lf: pl.LazyFrame,
    batch_col: str,
    n_bootstraps: int,
    sample_size: int,
    seed: int,
    n_jobs: int,
    parallel: bool,
) -> pl.DataFrame:
    """
    Run bootstrapped PERMANOVA to assess batch effects in cell-level data.

    Data is collected once before spawning workers so each bootstrap samples
    from an in-memory DataFrame rather than re-scanning the source files.

    Parameters
    ----------
    lf : pl.LazyFrame
        Cell-level lazy frame with a batch label column (already filtered to
        the desired variant class).
    batch_col : str
        Name of the batch grouping column.
    n_bootstraps : int
        Number of bootstrap samples to run.
    sample_size : int
        Rows per bootstrap sample.
    seed : int
        Base random seed; bootstrap ``i`` uses ``seed + i``.
    n_jobs : int
        Parallel workers (``-1`` = all cores). Ignored when ``parallel=False``.
    parallel : bool
        Whether to use joblib for parallel execution.

    Returns
    -------
    pl.DataFrame
        DataFrame with ``n_bootstraps`` rows and columns ``f_value`` and
        ``f_value_shuffled``.
    """
    logging.info(
        "Bootstrapping PERMANOVA: %d samples of size %d (seed=%d, parallel=%s)",
        n_bootstraps,
        sample_size,
        seed,
        parallel,
    )
    all_cols = lf.collect_schema().names()
    feature_cols = [c for c in all_cols if len(c) > 0 and c[0].isupper() and "_" in c]

    finite_mask = pl.all_horizontal(
        pl.col(f).is_not_null() & pl.col(f).is_finite() for f in feature_cols
    )
    filtered_lf = lf.filter(finite_mask).with_row_index(_TMP_IDX)
    idx_arr = filtered_lf.select(_TMP_IDX).collect()[_TMP_IDX].to_numpy()

    seeds = (seed + np.arange(n_bootstraps, dtype=np.int64)).tolist()

    def _run_one(s: int) -> pl.DataFrame:
        chosen = np.random.default_rng(s).choice(idx_arr, size=sample_size, replace=False)
        sampled = (
            filtered_lf
            .filter(pl.col(_TMP_IDX).is_in(set(chosen.tolist())))
            .drop(_TMP_IDX)
            .collect()
        )
        return compute_permanova_sample(sampled, batch_col, s)

    if parallel:
        dfs = Parallel(n_jobs=n_jobs)(delayed(_run_one)(s) for s in seeds)
    else:
        dfs = [_run_one(s) for s in tqdm.tqdm(seeds, desc="Bootstraps")]

    return pl.concat(dfs)


@hydra.main(version_base=None, config_path=None, config_name="permanova_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: bootstrap PERMANOVA batch-effect validation.

    Steps
    -----
    1. Glob ``input_file`` to find batch files; each file's stem becomes its
       batch label (added as ``meta_batch``).
    2. Optionally filter to a single variant class via ``variant_class_filter``
       (applied using :func:`.utils.classify_variant` on ``label_column``).
    3. Run :func:`bootstrap_permanova` to estimate observed and null F-statistics
       across ``n_bootstraps`` samples.
    4. Write results to a Parquet file.

    Output files
    ------------
    - ``{prefix}permanova.parquet``

    where ``prefix`` is ``{output_root}.`` when ``output_root`` is set,
    otherwise empty.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.permanova \\
            output_dir=./out \\
            'input_file=data/batches/*.parquet' \\
            variant_class_filter=WT \\
            n_bootstraps=200
    """
    perm_cfg: PermanovaConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(perm_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    perm_cfg.output_dir = output_dir
    setup_logging(perm_cfg, "permanova")

    lf, _ = load_batches(perm_cfg.input_file)

    if perm_cfg.variant_class_filter is not None:
        logging.info("Filtering to variant class: %s", perm_cfg.variant_class_filter)
        lf = lf.filter(
            pl.col(perm_cfg.label_column).map_elements(
                classify_variant, return_dtype=pl.Utf8
            )
            == perm_cfg.variant_class_filter
        )

    permanova_df = bootstrap_permanova(
        lf,
        batch_col=META_BATCH_COL,
        n_bootstraps=perm_cfg.n_bootstraps,
        sample_size=perm_cfg.sample_size,
        seed=perm_cfg.seed,
        n_jobs=perm_cfg.n_jobs,
        parallel=perm_cfg.parallel,
    )

    prefix = f"{perm_cfg.output_root}." if perm_cfg.output_root is not None else ""
    out_path = output_dir / f"{prefix}permanova.parquet"
    logging.info("Writing results to %s", out_path)
    permanova_df.write_parquet(out_path)
    logging.info("Done")


if __name__ == "__main__":
    main()
