"""Per-variant barcode-outlier detection via pairwise Tukey HSD, vectorized.

Hydra entry point (``python -m fisseq_data_pipeline.checkbarcodes``), backing the Nextflow process
``CHECK_BARCODES`` (per batch, gated by ``params.run_check_barcodes``). Consumes
the wide-format ``cell_scores.parquet`` produced by
:mod:`.ovwtcellscores` (one column per trained one-vs-wildtype model, plus
``meta_*`` columns), takes each cell's own-model score as its response variable
(the value of the column named by its own ``label_column`` value), groups by
variant and barcode, and for each variant with more than one distinct barcode
runs a pairwise Tukey HSD across its barcodes to flag barcodes whose cells score
anomalously relative to the variant's other barcodes.

Follows :mod:`.anova`'s pattern: per-group sufficient statistics (non-null
count, sum, sum of squares) are computed via a single lazy
``group_by(...).agg(...)`` query rather than looping per group, and the
studentized-range p-value is computed via ``scipy.stats.studentized_range.sf``
(the same closed-form distribution :mod:`statsmodels`'
``pairwise_tukeyhsd``/``MultiComparison`` uses internally, confirmed to match
bit-for-bit) applied to a vectorized array of every pairwise comparison across
every variant at once -- not by instantiating ``MultiComparison``/
``pairwise_tukeyhsd`` in a per-variant Python loop.
"""

import dataclasses
import logging
import pathlib

import hydra
import numpy as np
import polars as pl
import scipy.stats
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .config import LabeledInputConfig
from .utils.constants import FEATURE_SELECTOR, META_BARCODE_COL
from .utils.log import setup_logging

_cs = ConfigStore.instance()


@dataclasses.dataclass
class CheckBarcodesConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the barcode-outlier-check entry point.

    Attributes
    ----------
    min_cells : int
        Minimum number of cells required for a barcode (within a variant) to
        be included in the comparison. Barcodes with fewer cells are dropped
        before grouping. Defaults to ``10``.
    alpha : float
        Family-wise significance level for Tukey HSD. A barcode pair is
        flagged (``reject = True``) when its adjusted p-value is strictly
        less than this. Defaults to ``0.05``.
    """

    min_cells: int = 10
    alpha: float = 0.05


_cs.store(name="checkbarcodes_main", node=CheckBarcodesConfig)

_RESULT_SCHEMA = {
    "variant": pl.Utf8,
    "barcode": pl.Utf8,
    "group_mean": pl.Float64,
    "comparison_barcode": pl.Utf8,
    "comparison_group_mean": pl.Float64,
    "mean_diff": pl.Float64,
    "p_adj": pl.Float64,
    "reject": pl.Boolean,
}


def _group_stat_exprs(value_col: str, alias_prefix: str) -> list[pl.Expr]:
    """
    Build non-null count/sum/sum-of-squares expressions for one column, for
    use inside a ``group_by(...).agg(...)``.

    Same shape as :func:`.anova._group_stat_exprs` (duplicated here rather
    than imported, since that helper is module-private) -- polars'
    ``.count()``/``.sum()`` skip nulls by default.

    Parameters
    ----------
    value_col : str
        Name of the column to aggregate.
    alias_prefix : str
        Prefix for the output column aliases.

    Returns
    -------
    list[pl.Expr]
        The three aggregation expressions.
    """
    c = pl.col(value_col).cast(pl.Float64)
    return [
        c.count().alias(f"{alias_prefix}n"),
        c.sum().alias(f"{alias_prefix}sum"),
        (c**2).sum().alias(f"{alias_prefix}sumsq"),
    ]


def compute_barcode_tukey(
    scores_lf: pl.LazyFrame,
    label_col: str,
    min_cells: int,
    alpha: float,
) -> pl.DataFrame:
    """
    Compute pairwise Tukey HSD across barcodes, within each variant.

    Parameters
    ----------
    scores_lf : pl.LazyFrame
        Wide-format cell scores: one column per variant model, plus
        ``label_col`` and :data:`.utils.constants.META_BARCODE_COL`.
    label_col : str
        Name of the column identifying each cell's true variant label.
    min_cells : int
        Minimum cells required per barcode to include it in the comparison.
    alpha : float
        Family-wise significance level.

    Returns
    -------
    pl.DataFrame
        One row per barcode pair within a variant that had at least two
        distinct barcodes passing ``min_cells``, with columns matching
        :data:`_RESULT_SCHEMA`. Empty (but correctly typed) if no variant
        qualifies.
    """
    variant_cols = list(scores_lf.select(FEATURE_SELECTOR).collect_schema().names())
    logging.info("Found %d variant model column(s)", len(variant_cols))

    # Each cell's own-model score: the value of the column named by its own
    # label. A single vectorized coalesce over all variant columns, built by
    # looping over column names (cheap -- no data touched) rather than an
    # unpivot/melt, which would explode every cell to len(variant_cols) rows
    # before filtering back down. Cells whose label matches no variant column
    # (e.g. wildtype, or a variant whose model training failed upstream)
    # coalesce to null and are naturally excluded by _group_stat_exprs's
    # null-skipping count/sum below.
    own_score_expr = pl.coalesce(
        [pl.when(pl.col(label_col) == v).then(pl.col(v)) for v in variant_cols]
    ).alias("score")

    cell_lf = scores_lf.select(
        pl.col(label_col).alias("variant"),
        pl.col(META_BARCODE_COL).alias("barcode"),
        own_score_expr,
    )

    grouped = (
        cell_lf.group_by(["variant", "barcode"])
        .agg(_group_stat_exprs("score", ""))
        .collect()
    )

    n_before = grouped.height
    grouped = grouped.filter(pl.col("n") >= min_cells)
    logging.info(
        "Dropped %d of %d (variant, barcode) group(s) below min_cells=%d",
        n_before - grouped.height,
        n_before,
        min_cells,
    )

    barcode_counts = grouped.group_by("variant").agg(pl.len().alias("n_barcodes"))
    eligible_variants = barcode_counts.filter(pl.col("n_barcodes") >= 2)[
        "variant"
    ].to_list()
    n_variants_before = grouped["variant"].n_unique()
    grouped = grouped.filter(pl.col("variant").is_in(eligible_variants))
    logging.info(
        "Skipping %d of %d variant(s) with fewer than 2 qualifying barcodes",
        n_variants_before - len(eligible_variants),
        n_variants_before,
    )

    if grouped.height == 0:
        return pl.DataFrame(schema=_RESULT_SCHEMA)

    # Per-variant pooled within-group MSE and degrees of freedom, from the
    # same sufficient statistics -- same algebra as anova.py's
    # _f_statistic's ss_within, scoped per variant instead of globally.
    variant_stats = (
        grouped.with_columns(
            (pl.col("sumsq") - pl.col("sum") ** 2 / pl.col("n")).alias("ss_within_g")
        )
        .group_by("variant")
        .agg(
            pl.col("ss_within_g").sum().alias("ss_within"),
            pl.col("n").sum().alias("n_total"),
            pl.len().alias("k"),
        )
        .with_columns((pl.col("n_total") - pl.col("k")).alias("df_within"))
        .filter(pl.col("df_within") >= 1)
        .with_columns((pl.col("ss_within") / pl.col("df_within")).alias("mse"))
    )

    grouped = grouped.join(variant_stats.select("variant"), on="variant", how="inner")
    if grouped.height == 0:
        return pl.DataFrame(schema=_RESULT_SCHEMA)

    barcode_stats = grouped.select(
        "variant",
        "barcode",
        "n",
        (pl.col("sum") / pl.col("n")).alias("mean"),
    )

    # All barcode pairs within each variant, via one vectorized self-join
    # (not a Python loop per variant), deduped to barcode_a < barcode_b.
    pairs = (
        barcode_stats.select(
            pl.col("variant"),
            pl.col("barcode").alias("barcode_a"),
            pl.col("n").alias("n_a"),
            pl.col("mean").alias("mean_a"),
        )
        .join(
            barcode_stats.select(
                pl.col("variant"),
                pl.col("barcode").alias("barcode_b"),
                pl.col("n").alias("n_b"),
                pl.col("mean").alias("mean_b"),
            ),
            on="variant",
            how="inner",
        )
        .filter(pl.col("barcode_a") < pl.col("barcode_b"))
        .join(
            variant_stats.select("variant", "mse", "df_within", "k"),
            on="variant",
            how="inner",
        )
    )

    if pairs.height == 0:
        return pl.DataFrame(schema=_RESULT_SCHEMA)

    n_a = pairs["n_a"].to_numpy().astype(np.float64)
    n_b = pairs["n_b"].to_numpy().astype(np.float64)
    mean_a = pairs["mean_a"].to_numpy()
    mean_b = pairs["mean_b"].to_numpy()
    mse = pairs["mse"].to_numpy()
    df_within = pairs["df_within"].to_numpy()
    k = pairs["k"].to_numpy()

    mean_diff = mean_b - mean_a
    # Tukey-Kramer standard error for unequal group sizes.
    se = np.sqrt(mse / 2 * (1.0 / n_a + 1.0 / n_b))
    q = np.abs(mean_diff) / se
    p_adj = scipy.stats.studentized_range.sf(q, k, df_within)
    reject = p_adj < alpha

    result = pairs.select(
        "variant",
        pl.col("barcode_a").alias("barcode"),
        pl.col("mean_a").alias("group_mean"),
        pl.col("barcode_b").alias("comparison_barcode"),
        pl.col("mean_b").alias("comparison_group_mean"),
    ).with_columns(
        pl.Series("mean_diff", mean_diff, dtype=pl.Float64),
        pl.Series("p_adj", p_adj, dtype=pl.Float64),
        pl.Series("reject", reject, dtype=pl.Boolean),
    )

    return result.select(list(_RESULT_SCHEMA.keys()))


@hydra.main(version_base=None, config_path=None, config_name="checkbarcodes_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: per-variant barcode-outlier detection via Tukey HSD.

    Steps
    -----
    1. Load ``input_file`` (a ``cell_scores.parquet`` produced by
       :mod:`.ovwtcellscores`).
    2. Compute each cell's own-model score, group by (variant, barcode), and
       run a pairwise Tukey HSD across barcodes within each variant that has
       at least two barcodes with at least ``min_cells`` cells each -- see
       :func:`compute_barcode_tukey`.
    3. Write the result to ``{output_dir}/results.parquet``.

    Output files
    ------------
    - ``{output_dir}/results.parquet`` -- one row per compared barcode pair,
      with columns ``variant``, ``barcode``, ``group_mean``,
      ``comparison_barcode``, ``comparison_group_mean``, ``mean_diff``,
      ``p_adj``, ``reject``.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.checkbarcodes \\
            output_dir=./out \\
            input_file=out/cell_scores.parquet \\
            min_cells=10 \\
            alpha=0.05
    """
    cb_cfg: CheckBarcodesConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(cb_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cb_cfg.output_dir = output_dir
    setup_logging(cb_cfg, "checkbarcodes")

    logging.info("Loading cell scores from %s", cb_cfg.input_file)
    scores_lf = pl.scan_parquet(cb_cfg.input_file)

    result_df = compute_barcode_tukey(
        scores_lf, cb_cfg.label_column, cb_cfg.min_cells, cb_cfg.alpha
    )

    out_path = output_dir / "results.parquet"
    logging.info("Writing %d comparison row(s) to %s", len(result_df), out_path)
    result_df.write_parquet(out_path)
    logging.info("Done")


if __name__ == "__main__":
    main()
