"""Two-pass centroid batch correction for cell-level features.

Hydra entry points ``fisseq-batch-correct-fit`` / ``fisseq-batch-correct-transform``,
backing the Nextflow processes ``BATCH_CORRECT_FIT`` and ``BATCH_CORRECT_TRANSFORM``.
Fits per-(variant, batch) statistics and per-variant centroids across all batches,
then rescales each batch's cells to its variant's centroid and finally to the
wildtype centroid via :class:`BatchCorrector`.
"""

import dataclasses
import logging
import pathlib
from os import PathLike
from typing import Optional

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf
from polars import selectors as cs

from .config import LabeledInputConfig
from .utils.batches import load_batches
from .utils.constants import EPS, FEATURE_SELECTOR, META_BATCH_COL
from .utils.log import setup_logging

_STAT_COL = "_stat"


@dataclasses.dataclass
class BatchCorrector:
    """
    Container object storing per-variant, per-batch statistics and per-variant
    centroids used for two-pass centroid batch correction.

    Attributes
    ----------
    stats_vb : pl.DataFrame
        Per-(variant, batch) feature statistics. Columns: ``label_col``,
        ``batch_col``, ``_stat`` (``"mean"`` or ``"std"``), and one column per
        feature. Only variants present in more than one batch are retained.
    centroids : pl.DataFrame
        Per-variant centroid statistics, averaged over that variant's
        per-batch statistics (not over raw cells). Columns: ``label_col``,
        ``_stat`` (``"mean"`` or ``"std"``), and one column per feature.
    label_col : str
        Name of the column identifying variant labels.
    batch_col : str
        Name of the column identifying batch labels. Defaults to
        ``META_BATCH_COL`` (``"meta_batch"``).
    wt_label : str
        Value of ``label_col`` identifying wild-type rows, used as the target
        centroid for the second rescale pass. Defaults to ``"WT"``.
    """

    stats_vb: pl.DataFrame
    centroids: pl.DataFrame
    label_col: str
    batch_col: str = META_BATCH_COL
    wt_label: str = "WT"

    @classmethod
    def from_lazyframe(
        cls,
        lf: pl.LazyFrame,
        label_col: str,
        wt_label: str = "WT",
        batch_col: str = META_BATCH_COL,
    ) -> "BatchCorrector":
        """
        Fit a BatchCorrector by computing per-(variant, batch) statistics and
        per-variant centroids.

        Variants present in only one batch are dropped, since batch
        correction requires at least two batches to define a per-variant
        centroid. NaN values are excluded before computing statistics, and
        features with zero or near-zero variance (std < EPS) are stored as
        ``None`` (matches :class:`.normalize.Normalizer`'s convention).

        Parameters
        ----------
        lf : pl.LazyFrame
            Cell-level LazyFrame containing ``label_col``, ``batch_col``, and
            feature columns matched by ``FEATURE_SELECTOR``.
        label_col : str
            Name of the column identifying variant labels.
        wt_label : str, default "WT"
            Value of ``label_col`` identifying wild-type rows.
        batch_col : str, default META_BATCH_COL
            Name of the column identifying batch labels.

        Returns
        -------
        BatchCorrector
            A fitted ``BatchCorrector`` instance.

        Raises
        ------
        ValueError
            If ``wt_label`` is not present in at least two batches after
            dropping single-batch variants, since the WT centroid needed for
            the second rescale pass cannot be computed.
        """
        feature_cols = list(lf.select(FEATURE_SELECTOR).collect_schema().names())
        logging.info("Using %d feature column(s)", len(feature_cols))

        feature_lf = lf.with_columns(cs.numeric().fill_nan(None))

        logging.info("Computing per-(variant, batch) feature means")
        means_lf = (
            feature_lf.group_by([label_col, batch_col])
            .agg([pl.col(f).mean().alias(f) for f in feature_cols])
            .with_columns(pl.lit("mean").alias(_STAT_COL))
        )

        logging.info("Computing per-(variant, batch) feature standard deviations")
        stds_lf = (
            feature_lf.group_by([label_col, batch_col])
            .agg([pl.col(f).std().alias(f) for f in feature_cols])
            .with_columns(
                pl.when(cs.numeric().abs() < EPS)
                .then(None)
                .otherwise(cs.numeric())
                .name.keep()
            )
            .with_columns(pl.lit("std").alias(_STAT_COL))
        )

        counts_lf = feature_lf.group_by([label_col, batch_col]).agg(
            pl.len().alias("_n_cells")
        )

        stats_vb = pl.concat([means_lf, stds_lf]).collect()
        counts = counts_lf.collect()

        batch_counts = counts.group_by(label_col).agg(
            pl.col(batch_col).n_unique().alias("_n_batches"),
            pl.col("_n_cells").sum().alias("_n_cells"),
        )
        dropped = batch_counts.filter(pl.col("_n_batches") <= 1)
        if dropped.height > 0:
            logging.info(
                "Dropping %d variant(s) present in only one batch (%d cell(s) total): %s",
                dropped.height,
                int(dropped.get_column("_n_cells").sum()),
                dropped.get_column(label_col).to_list(),
            )
        else:
            logging.info("No variants dropped; all variants span more than one batch")

        keep = (
            batch_counts.filter(pl.col("_n_batches") > 1)
            .get_column(label_col)
            .to_list()
        )
        stats_vb = stats_vb.filter(pl.col(label_col).is_in(keep))

        logging.info("Computing per-variant centroids")
        centroids = stats_vb.group_by([label_col, _STAT_COL]).agg(
            [pl.col(f).mean().alias(f) for f in feature_cols]
        )

        wt_variants = (
            centroids.filter(pl.col(_STAT_COL) == "mean")
            .get_column(label_col)
            .to_list()
        )
        if wt_label not in wt_variants:
            raise ValueError(
                f"WT label {wt_label!r} is not present in at least two batches "
                "after dropping single-batch variants; cannot compute a WT "
                "centroid for the second rescale pass."
            )

        return cls(
            stats_vb=stats_vb,
            centroids=centroids,
            label_col=label_col,
            batch_col=batch_col,
            wt_label=wt_label,
        )

    def save(self, stats_path: PathLike, centroids_path: PathLike) -> None:
        """
        Serialize the BatchCorrector to two Parquet files.

        Parameters
        ----------
        stats_path : PathLike
            Destination file path for the per-(variant, batch) statistics
            (e.g. ``stats_vb.parquet``).
        centroids_path : PathLike
            Destination file path for the per-variant centroids (e.g.
            ``centroids.parquet``).
        """
        self.stats_vb.write_parquet(stats_path)
        self.centroids.write_parquet(centroids_path)

    @classmethod
    def load(
        cls,
        stats_path: PathLike,
        centroids_path: PathLike,
        label_col: str,
        batch_col: str = META_BATCH_COL,
        wt_label: str = "WT",
    ) -> "BatchCorrector":
        """
        Deserialize a BatchCorrector from Parquet files written by :meth:`save`.

        Parameters
        ----------
        stats_path : PathLike
            Path to the per-(variant, batch) statistics Parquet file.
        centroids_path : PathLike
            Path to the per-variant centroids Parquet file.
        label_col : str
            Name of the column identifying variant labels.
        batch_col : str, default META_BATCH_COL
            Name of the column identifying batch labels.
        wt_label : str, default "WT"
            Value of ``label_col`` identifying wild-type rows.

        Returns
        -------
        BatchCorrector
            A ``BatchCorrector`` instance with ``stats_vb`` and ``centroids``
            restored.
        """
        return cls(
            stats_vb=pl.read_parquet(stats_path),
            centroids=pl.read_parquet(centroids_path),
            label_col=label_col,
            batch_col=batch_col,
            wt_label=wt_label,
        )

    def transform(self, lf: pl.LazyFrame, batch: str) -> pl.LazyFrame:
        """
        Apply two-pass centroid batch correction to a single batch's LazyFrame.

        Each feature value ``x`` for variant ``v`` in this batch is first
        z-scored against ``v``'s own per-``batch`` statistics and rescaled to
        ``v``'s centroid::

            x' = (x - mean_vb) / std_vb * centroid_std_v + centroid_mean_v

        then rescaled a second time from ``v``'s centroid to the wild-type
        centroid::

            x'' = (x' - centroid_mean_v) / centroid_std_v * centroid_std_WT + centroid_mean_WT

        Rows belonging to variants dropped during :meth:`from_lazyframe`
        (present in only one batch) are excluded, since no statistics exist
        for them.

        Parameters
        ----------
        lf : pl.LazyFrame
            Input LazyFrame for a single batch, containing ``label_col`` and
            feature columns matched by ``FEATURE_SELECTOR``. Must not already
            contain ``batch_col`` (unnecessary — the batch is identified by
            the ``batch`` argument since every batch file is scanned
            independently).
        batch : str
            Label identifying which batch this LazyFrame belongs to. Must
            match a value of ``batch_col`` seen during :meth:`from_lazyframe`.

        Returns
        -------
        pl.LazyFrame
            A LazyFrame with feature columns corrected to the wild-type
            centroid. Non-feature columns are passed through unchanged.
        """
        feature_cols = [
            c
            for c in self.stats_vb.columns
            if c not in (self.label_col, self.batch_col, _STAT_COL)
        ]

        def _wide(df: pl.DataFrame, stat: str, suffix: str) -> pl.LazyFrame:
            sub = df.filter(pl.col(_STAT_COL) == stat).drop(_STAT_COL)
            if self.batch_col in sub.columns:
                sub = sub.drop(self.batch_col)
            return sub.rename({f: f"{f}{suffix}" for f in feature_cols}).lazy()

        batch_stats = self.stats_vb.filter(pl.col(self.batch_col) == batch)
        mean_vb = _wide(batch_stats, "mean", "__mean_vb")
        std_vb = _wide(batch_stats, "std", "__std_vb")
        centroid_mean = _wide(self.centroids, "mean", "__centroid_mean_v")
        centroid_std = _wide(self.centroids, "std", "__centroid_std_v")

        wt_mean_row: dict[str, Optional[float]] = (
            self.centroids.filter(
                (pl.col(_STAT_COL) == "mean")
                & (pl.col(self.label_col) == self.wt_label)
            )
            .drop([self.label_col, _STAT_COL])
            .row(0, named=True)
        )
        wt_std_row: dict[str, Optional[float]] = (
            self.centroids.filter(
                (pl.col(_STAT_COL) == "std") & (pl.col(self.label_col) == self.wt_label)
            )
            .drop([self.label_col, _STAT_COL])
            .row(0, named=True)
        )

        logging.info(
            "Batch %r: joining against fitted statistics (inner join drops rows "
            "whose variant was not fitted, e.g. present in only one batch during fit)",
            batch,
        )
        joined = (
            lf.with_columns(cs.numeric().fill_nan(None))
            .join(mean_vb, on=self.label_col, how="inner")
            .join(std_vb, on=self.label_col, how="inner")
            .join(centroid_mean, on=self.label_col, how="inner")
            .join(centroid_std, on=self.label_col, how="inner")
        )

        step4 = joined.with_columns(
            [
                (
                    (pl.col(f) - pl.col(f"{f}__mean_vb"))
                    / pl.col(f"{f}__std_vb")
                    * pl.col(f"{f}__centroid_std_v")
                    + pl.col(f"{f}__centroid_mean_v")
                ).alias(f)
                for f in feature_cols
            ]
        )

        step5 = step4.with_columns(
            [
                (
                    (pl.col(f) - pl.col(f"{f}__centroid_mean_v"))
                    / pl.col(f"{f}__centroid_std_v")
                    * wt_std_row.get(f)
                    + wt_mean_row.get(f)
                ).alias(f)
                for f in feature_cols
            ]
        )

        helper_cols = [
            f"{f}{suffix}"
            for f in feature_cols
            for suffix in (
                "__mean_vb",
                "__std_vb",
                "__centroid_mean_v",
                "__centroid_std_v",
            )
        ]

        return step5.drop(helper_cols).with_columns(cs.numeric().fill_nan(None))


@dataclasses.dataclass
class BatchCorrectFitConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the batch-correction fit entry point.

    ``input_file`` is interpreted as a glob pattern. Each matching file is
    treated as a separate batch.

    Attributes
    ----------
    wt_label : str
        Value of ``label_column`` identifying wild-type rows. Defaults to
        ``"WT"``.
    use_parent_name : bool
        If ``True``, label each batch by its file's parent directory name
        instead of the file stem. Useful when every batch file shares the
        same name (e.g. ``qc_filter/*/filtered_cells.parquet``).
    """

    wt_label: str = "WT"
    use_parent_name: bool = False


@dataclasses.dataclass
class BatchCorrectTransformConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the batch-correction transform entry point.

    Attributes
    ----------
    stats_file : str
        Path to the per-(variant, batch) statistics Parquet file written by
        the fit entry point.
    centroids_file : str
        Path to the per-variant centroids Parquet file written by the fit
        entry point.
    batch : str
        Label identifying which batch ``input_file`` belongs to. Passed
        explicitly rather than inferred from the filename, since batch files
        may share an identical name (e.g. every batch's QC-filtered output is
        named ``filtered_cells.parquet``).
    wt_label : str
        Value of ``label_column`` identifying wild-type rows. Defaults to
        ``"WT"``.
    """

    stats_file: str = MISSING
    centroids_file: str = MISSING
    batch: str = MISSING
    wt_label: str = "WT"


_cs = ConfigStore.instance()
_cs.store(name="batch_correct_fit_main", node=BatchCorrectFitConfig)
_cs.store(name="batch_correct_transform_main", node=BatchCorrectTransformConfig)


@hydra.main(version_base=None, config_path=None, config_name="batch_correct_fit_main")
def fit_main(cfg: DictConfig) -> None:
    """
    Hydra entry point: fit a BatchCorrector across all batch files.

    Globs ``input_file`` to find batch files (see :func:`.utils.batches.load_batches`),
    fits a :class:`BatchCorrector`, and writes its statistics to two Parquet
    files.

    Output files
    ------------
    - ``{output_dir}/stats_vb.parquet`` — per-(variant, batch) statistics.
    - ``{output_dir}/centroids.parquet`` — per-variant centroids.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.batchcorrect \\
            output_dir=./out \\
            'input_file=data/batches/*.parquet'
    """
    fit_cfg: BatchCorrectFitConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(fit_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_cfg.output_dir = output_dir
    setup_logging(fit_cfg, "batch_correct_fit")

    lf, _ = load_batches(fit_cfg.input_file, use_parent_name=fit_cfg.use_parent_name)

    logging.info("Fitting batch corrector")
    corrector = BatchCorrector.from_lazyframe(
        lf, label_col=fit_cfg.label_column, wt_label=fit_cfg.wt_label
    )

    stats_path = output_dir / "stats_vb.parquet"
    centroids_path = output_dir / "centroids.parquet"
    logging.info("Writing stats to %s and centroids to %s", stats_path, centroids_path)
    corrector.save(stats_path, centroids_path)

    logging.info("Done")


@hydra.main(
    version_base=None, config_path=None, config_name="batch_correct_transform_main"
)
def transform_main(cfg: DictConfig) -> None:
    """
    Hydra entry point: apply a fitted BatchCorrector to a single batch file.

    Reads the input file at ``input_file``, loads a :class:`BatchCorrector`
    from ``stats_file``/``centroids_file``, applies it, and writes the
    corrected batch to its own output file.

    Output path
    -----------
    - If ``output_root`` is set: ``{output_root}.{stem}.{ext}``
    - Otherwise: ``{output_dir}/{filename}`` (same name as the input file)

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.batchcorrect \\
            output_dir=./out \\
            input_file=data/batch1.parquet \\
            batch=batch1 \\
            stats_file=./fit/stats_vb.parquet \\
            centroids_file=./fit/centroids.parquet
    """
    trans_cfg: BatchCorrectTransformConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(trans_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trans_cfg.output_dir = output_dir
    setup_logging(trans_cfg, "batch_correct_transform")

    input_path = pathlib.Path(trans_cfg.input_file)
    logging.info("Loading input from %s", input_path)
    lf = pl.scan_parquet(input_path)

    logging.info("Loading batch corrector")
    corrector = BatchCorrector.load(
        trans_cfg.stats_file,
        trans_cfg.centroids_file,
        label_col=trans_cfg.label_column,
        wt_label=trans_cfg.wt_label,
    )

    logging.info("Applying batch correction for batch %r", trans_cfg.batch)
    lf = corrector.transform(lf, batch=trans_cfg.batch)

    stem = input_path.stem
    ext = input_path.suffix.lstrip(".")
    if trans_cfg.output_root is not None:
        out_path = pathlib.Path(f"{trans_cfg.output_root}.{stem}.{ext}")
    else:
        out_path = output_dir / input_path.name

    logging.info("Writing output to %s", out_path)
    lf.sink_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    fit_main()
