import dataclasses
import logging
import pathlib
from os import PathLike
from typing import Optional

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from polars import selectors as cs

from .config import InputConfig
from .utils.constants import CONTROL_COLUMN, CONTROL_COLUMN_NAME, EPS, FEATURE_SELECTOR
from .utils.log import setup_logging


@dataclasses.dataclass
class Normalizer:
    """
    Container object storing per-feature normalization statistics.

    Attributes
    ----------
    means : pl.DataFrame
        A DataFrame of shape (n_batches, n_features) containing the mean value
        of each feature for each batch. When batch-wise normalization is not
        used, this has shape (1, n_features).
    stds : pl.DataFrame
        A DataFrame of shape (n_batches, n_features) containing the standard
        deviation of each feature for each batch. When batch-wise normalization
        is not used, this has shape (1, n_features).
    """

    means: pl.DataFrame
    stds: pl.DataFrame

    @classmethod
    def from_lazyframe(
        cls, lf: pl.LazyFrame, fit_only_on_control: bool = True
    ) -> "Normalizer":
        """
        Fit a Normalizer by computing per-feature means and standard deviations.

        NaN values are excluded before computing statistics. Features with zero
        or near-zero variance (std < EPS) are stored as ``None`` and will
        produce ``NaN`` when applied, acting as a natural indicator that the
        feature should be dropped.

        Parameters
        ----------
        lf : pl.LazyFrame
            Input LazyFrame. Must contain a boolean ``CONTROL_COLUMN`` column
            when ``fit_only_on_control=True``, and feature columns matched by
            ``FEATURE_SELECTOR``.
        fit_only_on_control : bool, default True
            If ``True``, statistics are computed using only rows where
            ``CONTROL_COLUMN`` is ``True``.

        Returns
        -------
        Normalizer
            A fitted ``Normalizer`` instance with ``means`` and ``stds``
            DataFrames of shape ``(1, n_features)``.
        """
        if fit_only_on_control:
            logging.info("Adding query to filter for control samples")
            lf = lf.filter(CONTROL_COLUMN)

        logging.info("")
        feature_lf = lf.select(FEATURE_SELECTOR).with_columns(
            cs.numeric().fill_nan(None)
        )

        logging.info("Computing feature means")
        means = feature_lf.mean().collect()

        logging.info("Computing feature standard deviations")
        stds = (
            feature_lf.std()
            .with_columns(
                pl.when(cs.numeric().abs() < EPS)
                .then(None)
                .otherwise(cs.numeric())
                .name.keep()
            )
            .collect()
        )

        return cls(means=means, stds=stds)

    def save(self, path: PathLike) -> None:
        """
        Serialize the Normalizer to a Parquet file.

        Both ``means`` and ``stds`` are written as a single DataFrame with a
        ``_stat`` column set to ``"mean"`` or ``"std"`` to distinguish the two
        rows. Reload with :meth:`load`.

        Parameters
        ----------
        path : PathLike
            Destination file path (e.g. ``normalizer.parquet``).
        """
        pl.concat(
            [
                self.means.with_columns(pl.lit("mean").alias("_stat")),
                self.stds.with_columns(pl.lit("std").alias("_stat")),
            ]
        ).write_parquet(path)

    @classmethod
    def load(cls, path: PathLike) -> "Normalizer":
        """
        Deserialize a Normalizer from a Parquet file written by :meth:`save`.

        Parameters
        ----------
        path : PathLike
            Path to a Parquet file previously written by :meth:`save`.

        Returns
        -------
        Normalizer
            A ``Normalizer`` instance with ``means`` and ``stds`` restored.
        """
        df = pl.read_parquet(path)
        means = df.filter(pl.col("_stat") == "mean").drop("_stat")
        stds = df.filter(pl.col("_stat") == "std").drop("_stat")
        return cls(means=means, stds=stds)

    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Apply z-score normalization to a LazyFrame using the fitted statistics.

        Each feature column ``c`` is transformed as ``(c - mean_c) / std_c``
        using the values stored in ``self.means`` and ``self.stds``.

        NaNs are converted to nulls both before and after normalization: the
        pre-pass ensures NaN inputs don't propagate into the arithmetic, and
        the post-pass converts any NaNs produced by division by a zero-variance
        feature (whose std is ``None``) into nulls for consistent downstream
        handling. Non-feature columns are passed through unchanged.

        Parameters
        ----------
        lf : pl.LazyFrame
            Input LazyFrame containing feature columns matched by
            ``FEATURE_SELECTOR``.

        Returns
        -------
        pl.LazyFrame
            A LazyFrame with feature columns z-score normalized in-place.
            Any non-finite inputs or zero-variance features are represented
            as nulls.
        """
        logging.info("Adding normalization queries")
        means: dict[str, Optional[float]] = self.means.row(0, named=True)
        stds: dict[str, Optional[float]] = self.stds.row(0, named=True)

        lf = (
            lf.with_columns(cs.numeric().fill_nan(None))
            .with_columns(
                (pl.col(c) - means.get(c)) / stds.get(c)
                for c in lf.select(FEATURE_SELECTOR).columns
            )
            .with_columns(cs.numeric().fill_nan(None))
        )

        return lf


@dataclasses.dataclass
class NormalizeConfig(InputConfig):
    """
    Hydra structured configuration for the normalization entry point.

    Attributes
    ----------
    control_sample_query : str
        SQL-like WHERE clause identifying control rows used to fit the
        normalizer (e.g. ``"meta_aa_changes = 'WT'"``).
    save_normalizer : bool
        If ``True``, persist the fitted :class:`Normalizer` to a parquet file
        alongside the normalized output.
    """

    control_sample_query: str = "meta_aa_changes = 'WT'"
    save_normalizer: bool = True


_cs = ConfigStore.instance()
_cs.store(name="normalize_main", node=NormalizeConfig)


def add_control_indicator_column(
    lf: pl.LazyFrame, cfg: NormalizeConfig
) -> pl.LazyFrame:
    """
    Append a boolean ``CONTROL_COLUMN`` to a LazyFrame using a SQL predicate.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input LazyFrame to annotate.
    cfg : NormalizeConfig
        Configuration supplying ``control_sample_query``, a SQL-like WHERE
        clause evaluated against the frame (e.g. ``"meta_aa_changes = 'WT'"``).

    Returns
    -------
    pl.LazyFrame
        The input frame with an additional boolean ``CONTROL_COLUMN`` column
        that is ``True`` for rows matching the query.
    """
    return lf.with_columns(
        pl.sql_expr(cfg.control_sample_query).alias(CONTROL_COLUMN_NAME)
    )


@hydra.main(version_base=None, config_path=None, config_name="normalize_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: fit and apply z-score normalization to a parquet file.

    Reads the input file at ``input_file``, adds a control indicator column
    via :func:`add_control_indicator_column`, fits a :class:`Normalizer` on
    the control rows, applies it, and writes the result.

    Output path
    -----------
    - If ``output_root`` is set: ``{output_root}.{stem}.{ext}``
    - Otherwise: ``{output_dir}/{filename}`` (same name as the input file)

    If ``save_normalizer`` is ``True``, the fitted :class:`Normalizer` is also
    written alongside the output using the same root/dir convention with the
    name ``normalizer.parquet``.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.normalize \\
            output_dir=./out \\
            input_file=data/cells.parquet
    """
    norm_cfg: NormalizeConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(norm_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_cfg.output_dir = output_dir
    setup_logging(norm_cfg, "normalize")

    input_path = pathlib.Path(norm_cfg.input_file)
    logging.info("Loading input from %s", input_path)
    lf = pl.scan_parquet(input_path)
    lf = add_control_indicator_column(lf, norm_cfg)

    logging.info("Fitting normalizer")
    normalizer = Normalizer.from_lazyframe(lf)
    logging.info("Applying normalizer")
    lf = normalizer.apply(lf)

    stem = input_path.stem
    ext = input_path.suffix.lstrip(".")
    if norm_cfg.output_root is not None:
        out_path = pathlib.Path(f"{norm_cfg.output_root}.{stem}.{ext}")
    else:
        out_path = output_dir / input_path.name

    logging.info("Writing output to %s", out_path)
    lf.sink_parquet(out_path)

    if norm_cfg.save_normalizer:
        if norm_cfg.output_root is not None:
            norm_path = pathlib.Path(f"{norm_cfg.output_root}.normalizer.parquet")
        else:
            norm_path = output_dir / "normalizer.parquet"
        logging.info("Saving normalizer to %s", norm_path)
        normalizer.save(norm_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
