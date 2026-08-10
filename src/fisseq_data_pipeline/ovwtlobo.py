"""Leave-one-barcode-out (LOBO) OvWT generalization testing.

Hydra entry point (``python -m fisseq_data_pipeline.ovwtlobo``), backing the Nextflow
process ``OVWTLOBO_BATCHWISE``. Tests whether :mod:`.ovwt`'s classifiers generalize to
unseen barcodes: for each non-wildtype variant with at least ``min_barcodes_per_variant``
barcodes, repeatedly holds out one of that variant's barcodes entirely, retrains an
OvWT-equivalent binary XGBoost classifier on the variant's remaining barcodes (via the
same :func:`.utils.xgbparams.train_binary_xgboost` used by :mod:`.ovwt`), and scores the
held-out barcode's cells. Does **not** recompute an in-distribution baseline -- normal
OvWT already produces that number as its own pipeline stage; this module's output is
joinable to it on ``variant`` (and ``barcode``) so a downstream step can compute the
generalization gap.

The wildtype reference pool is barcode-independent of which variant-barcode is held
out, so it is split into a fixed train/val/test partition once per batch (not
re-fit per fold) -- the held-out barcode's cells are paired with the held-out
wildtype test slice to form a genuine two-class test set (AUROC is undefined on a
single-class array).
"""

import dataclasses
import logging
import pathlib
import pickle
import traceback
from typing import Optional, Union

import hydra
import numpy as np
import polars as pl
import xgboost as xgb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .config import LabeledInputConfig
from .utils.batches import load_batches
from .utils.constants import META_BARCODE_COL
from .utils.filtering import (
    _exclude_blocked_features,
    downsample_group_to_target,
)
from .utils.log import setup_logging
from .utils.xgbparams import (
    XGBoostConfig,
    evaluate_binary,
    get_feature_cols,
    resolve_feature_importance,
    split_indices_stratified,
    train_binary_xgboost,
)


@dataclasses.dataclass
class OvwtLoboConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the leave-one-barcode-out entry point.

    Extends :class:`.config.LabeledInputConfig` with parameters controlling
    per-(variant, held-out-barcode) XGBoost training, mirroring
    :class:`.ovwt.OvwtConfig` wherever the same concept applies.

    Attributes
    ----------
    wt_label : str
        Label string identifying wildtype cells. Defaults to ``"WT"``.
    barcode_column : str
        Name of the column in ``input_file`` identifying each cell's barcode.
        Defaults to :data:`.utils.constants.META_BARCODE_COL` (``"meta_barcode"``).
    random_state : int
        Random seed for every split and downsampling step. Defaults to ``42``.
    feature_cols : list or None
        Explicit list of feature column names. If ``None``, columns are
        auto-detected by :func:`.xgbparams.get_feature_cols`. Defaults to
        ``None``.
    min_cells : int or None
        Minimum number of cells a variant's *other* (non-held-out) barcodes
        must have, combined, for a given held-out-barcode fold to be
        attempted -- the same check :mod:`.ovwt`'s ``filter_min_cells``
        applies to a variant's full cell count, evaluated here after removing
        one barcode's cells. ``None`` disables the check. Defaults to ``250``.
    min_cells_holdout : int
        Minimum number of cells the held-out barcode itself must have for its
        fold to be scored. Defaults to ``100``.
    min_barcodes_per_variant : int
        Minimum number of distinct barcodes a variant must have for LOBO to
        be applicable at all. Variants below this get a single
        ``status="skipped_single_barcode"`` output row. Defaults to ``2``.
    downsample_wt : bool or int
        If ``True``, downsample wildtype cells to the size of the largest
        variant group before splitting. If an integer, downsample to that
        exact count. ``False`` disables downsampling. Defaults to ``True``.
    max_cells_per_barcode_wt : int or None
        Maximum cells allowed for any single wildtype barcode. ``None``
        disables this cap. Defaults to ``None``.
    max_cells_per_barcode_variant : int or None
        Maximum cells allowed for any single non-wildtype barcode. ``None``
        disables this cap. Defaults to ``None``.
    save_models : bool
        If ``True``, write every trained model (keyed by ``(variant, barcode)``)
        to ``models.pkl``. LOBO trains many more models than :mod:`.ovwt`
        (one per variant-barcode, not one per variant); set ``False`` to skip
        this output on large runs. Defaults to ``True``.
    feature_block_list_file : str or None
        Optional path to a parquet file with at least ``feature`` (str) and
        ``feature_ok`` (bool) columns. Defaults to ``None`` (no features
        blocked).
    barcode_block_list_file : str or None
        Optional path to a parquet file with at least ``barcode`` (str) and
        ``barcode_ok`` (bool) columns. Defaults to ``None`` (no barcodes
        blocked).
    xgboost : XGBoostConfig
        XGBoost training configuration. Defaults to :class:`.xgbparams.XGBoostConfig`.
    """

    wt_label: str = "WT"
    barcode_column: str = META_BARCODE_COL
    random_state: int = 42
    feature_cols: Optional[list] = None
    min_cells: Optional[int] = 250
    min_cells_holdout: int = 100
    min_barcodes_per_variant: int = 2
    downsample_wt: Union[bool, int] = True
    max_cells_per_barcode_wt: Optional[int] = None
    max_cells_per_barcode_variant: Optional[int] = None
    save_models: bool = True
    feature_block_list_file: Optional[str] = None
    barcode_block_list_file: Optional[str] = None
    xgboost: XGBoostConfig = dataclasses.field(default_factory=XGBoostConfig)


_cs = ConfigStore.instance()
_cs.store(name="ovwtlobo_main", node=OvwtLoboConfig)


_RESULTS_SCHEMA = {
    "variant": pl.Utf8,
    "barcode": pl.Utf8,
    "status": pl.Utf8,
    "n_barcodes_train": pl.Int64,
    "n_cells_train": pl.Int64,
    "n_cells_test": pl.Int64,
    "n_wt_cells_test": pl.Int64,
    "train_auroc": pl.Float64,
    "train_accuracy": pl.Float64,
    "val_auroc": pl.Float64,
    "val_accuracy": pl.Float64,
    "test_auroc": pl.Float64,
    "test_accuracy": pl.Float64,
}


def _result_row(variant: str, barcode: Optional[str], status: str, **kwargs) -> dict:
    """Build a ``_RESULTS_SCHEMA``-shaped row, defaulting every unset field to
    ``None`` -- used for both the ``"ok"`` and every ``skipped_*``/``"failed"``
    row so :func:`main` can always build ``results_df`` against one explicit
    schema (see module docstring / :data:`_RESULTS_SCHEMA`)."""
    row = dict.fromkeys(_RESULTS_SCHEMA)
    row["variant"] = variant
    row["barcode"] = barcode
    row["status"] = status
    row.update(kwargs)
    return row


def _exclude_blocked_barcodes(
    data_df: pl.DataFrame,
    barcode_column: str,
    barcode_block_list_file: Optional[str],
) -> pl.DataFrame:
    """
    Drop cells whose barcode is blocked.

    Copied from :mod:`.ovwt` (not further factored into
    :mod:`.utils.filtering`, matching that module's documented "thin,
    same-named wrapper" convention).

    Parameters
    ----------
    data_df : pl.DataFrame
        Cell-level DataFrame containing ``barcode_column``.
    barcode_column : str
        Name of the column identifying each cell's barcode.
    barcode_block_list_file : str or None
        Path to a parquet file with ``barcode`` (str) and ``barcode_ok``
        (bool) columns, or ``None`` to skip filtering entirely.

    Returns
    -------
    pl.DataFrame
        ``data_df`` with rows whose ``barcode_column`` value is blocked
        removed. Unchanged if ``barcode_block_list_file`` is ``None``.
    """
    if barcode_block_list_file is None:
        return data_df
    bl_df = pl.read_parquet(barcode_block_list_file)
    blocked = set(bl_df.filter(~pl.col("barcode_ok"))["barcode"].to_list())
    if not blocked:
        return data_df
    return data_df.filter(~pl.col(barcode_column).is_in(blocked))


def downsample_per_barcode(
    data_df: pl.DataFrame,
    barcode_column: str,
    label_col: str,
    wt_label: str,
    seed: int,
    max_cells_wt: Optional[int] = None,
    max_cells_variant: Optional[int] = None,
) -> pl.DataFrame:
    """
    Cap each barcode's cell count independently, per wildtype/variant status.

    Copied from :mod:`.ovwt` (see that module for the full docstring) --
    identical behavior: wildtype and variant rows are capped independently.
    """
    if max_cells_wt is None and max_cells_variant is None:
        return data_df

    def _cap(df: pl.DataFrame, max_cells: Optional[int]) -> pl.DataFrame:
        if max_cells is None or len(df) == 0:
            return df
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)
        row_in_barcode = pl.int_range(pl.len()).over(barcode_column)
        return shuffled.filter(row_in_barcode < max_cells)

    wt_df = _cap(data_df.filter(pl.col(label_col) == wt_label), max_cells_wt)
    variant_df = _cap(data_df.filter(pl.col(label_col) != wt_label), max_cells_variant)
    return pl.concat([variant_df, wt_df])


def downsample_wildtype(
    data_df: pl.DataFrame,
    label_col: str,
    wt_label: str,
    seed: int,
    n: Optional[int] = None,
) -> pl.DataFrame:
    """Downsample wildtype rows to a target count -- copied from :mod:`.ovwt`."""
    return downsample_group_to_target(data_df, label_col, wt_label, seed, n=n)


def prepare_data(
    data_df: pl.DataFrame,
    cfg: DictConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Once-per-batch preprocessing, shared across every ``(variant, barcode)`` fold.

    Performs feature-column resolution and blocklist exclusion, barcode
    blocklist exclusion, the per-barcode cell cap, and wildtype downsampling
    -- everything :mod:`.ovwt`'s ``train_test_val_split`` does except its
    final split, since LOBO needs a different split per held-out barcode.

    The wildtype pool is barcode-independent of which variant-barcode will
    later be held out (LOBO never holds out a wildtype barcode -- that's
    :mod:`.wtvwt`'s job), so it is split into a fixed train/val/test
    partition here, once, rather than being re-fit per fold.

    Parameters
    ----------
    data_df : pl.DataFrame
        Full feature DataFrame containing feature columns, ``cfg.label_column``,
        and ``cfg.barcode_column``.
    cfg : DictConfig
        Hydra config supplying ``label_column``, ``wt_label``, ``feature_cols``,
        ``feature_block_list_file``, ``barcode_column``, ``barcode_block_list_file``,
        ``max_cells_per_barcode_wt``, ``max_cells_per_barcode_variant``,
        ``downsample_wt``, and ``random_state``.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]
        ``(non_wt_df, wt_train_df, wt_val_df, wt_test_df)``. ``non_wt_df``
        retains every surviving variant and barcode, un-split. Each contains
        feature columns, ``cfg.label_column``, and ``cfg.barcode_column``.
    """
    label_col = cfg.label_column
    barcode_col = cfg.barcode_column
    if cfg.feature_cols is not None:
        feature_cols = list(cfg.feature_cols)
    else:
        feature_cols = get_feature_cols(data_df)
    feature_cols = _exclude_blocked_features(feature_cols, cfg.feature_block_list_file)

    # Barcode blocklist + per-barcode cap applied before the column-narrowing
    # select() below, while barcode_column is guaranteed present -- same
    # ordering rationale as .ovwt.train_test_val_split.
    data_df = _exclude_blocked_barcodes(
        data_df, barcode_col, cfg.barcode_block_list_file
    )
    data_df = data_df.filter(pl.col(barcode_col).is_not_null())
    data_df = downsample_per_barcode(
        data_df,
        barcode_col,
        label_col,
        cfg.wt_label,
        cfg.random_state,
        max_cells_wt=cfg.max_cells_per_barcode_wt,
        max_cells_variant=cfg.max_cells_per_barcode_variant,
    )

    select_cols = feature_cols + [label_col, barcode_col]
    data_df = data_df.select(select_cols)
    data_df = data_df.filter(pl.col(label_col).is_not_null())

    if cfg.downsample_wt is not False and cfg.downsample_wt != 0:
        n = cfg.downsample_wt if not isinstance(cfg.downsample_wt, bool) else None
        data_df = downsample_wildtype(
            data_df, label_col, cfg.wt_label, cfg.random_state, n=n
        )

    wt_df = data_df.filter(pl.col(label_col) == cfg.wt_label)
    non_wt_df = data_df.filter(pl.col(label_col) != cfg.wt_label)

    wt_df = wt_df.with_row_index("__idx__")
    wt_labels = wt_df.get_column(label_col).to_numpy()
    wt_train_idx, wt_test_idx, wt_val_idx = split_indices_stratified(
        wt_labels, cfg.random_state
    )

    def select_wt_rows(idx: np.ndarray) -> pl.DataFrame:
        return wt_df.filter(pl.col("__idx__").is_in(idx)).drop("__idx__")

    wt_train_df = select_wt_rows(wt_train_idx)
    wt_val_df = select_wt_rows(wt_val_idx)
    wt_test_df = select_wt_rows(wt_test_idx)

    return non_wt_df, wt_train_df, wt_val_df, wt_test_df


def split_variant_holdout(
    variant_rows: pl.DataFrame,
    barcode: str,
    barcode_col: str,
    label_col: str,
    random_state: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split one variant's rows into held-in-train, held-in-val, and held-out.

    ``held_out_df`` (the barcode being tested) never contributes to
    ``held_in_train_df``/``held_in_val_df`` -- it is filtered out first, by
    construction. The held-in rows (every *other* barcode of this variant)
    are further split via :func:`.xgbparams.split_indices_stratified`, with
    the resulting train and test portions merged into one effective training
    set (LOBO does not need a separate in-distribution test split per fold --
    the held-out barcode itself is the test) and the val portion kept for
    early stopping only.

    Parameters
    ----------
    variant_rows : pl.DataFrame
        Every row belonging to one variant (every barcode).
    barcode : str
        The barcode to hold out.
    barcode_col : str
        Name of the column identifying each cell's barcode.
    label_col : str
        Name of the label column (used only as the stratification key for
        the held-in split; every row here already shares one variant label).
    random_state : int
        Random seed for the held-in split.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        ``(held_in_train_df, held_in_val_df, held_out_df)``.
    """
    held_out_df = variant_rows.filter(pl.col(barcode_col) == barcode)
    held_in_df = variant_rows.filter(pl.col(barcode_col) != barcode)

    if len(held_in_df) == 0:
        return held_in_df, held_in_df, held_out_df

    held_in_df = held_in_df.with_row_index("__idx__")
    labels = held_in_df.get_column(label_col).to_numpy()
    train_idx, test_idx, val_idx = split_indices_stratified(labels, random_state)
    held_in_train_idx = np.concatenate([train_idx, test_idx])

    held_in_train_df = held_in_df.filter(
        pl.col("__idx__").is_in(held_in_train_idx)
    ).drop("__idx__")
    held_in_val_df = held_in_df.filter(pl.col("__idx__").is_in(val_idx)).drop("__idx__")
    return held_in_train_df, held_in_val_df, held_out_df


def profile_holdout(
    variant: str,
    barcode: str,
    non_wt_df: pl.DataFrame,
    wt_train_df: pl.DataFrame,
    wt_val_df: pl.DataFrame,
    wt_test_df: pl.DataFrame,
    cfg: DictConfig,
) -> tuple[dict, Optional[xgb.Booster]]:
    """
    Train and evaluate an XGBoost model for one variant with one barcode held out.

    Gates on ``cfg.min_cells_holdout`` (the held-out barcode's own cell
    count) and ``cfg.min_cells`` (the variant's remaining, held-in cell
    count) *before* attempting any split, mirroring :mod:`.ovwt`'s own
    ``min_cells`` check. On success, trains via
    :func:`.xgbparams.train_binary_xgboost` on the held-in variant rows plus
    the shared wildtype train/val slices, and evaluates via
    :func:`.xgbparams.evaluate_binary` against a two-class test set: the
    held-out barcode's rows plus the shared wildtype test slice.

    Parameters
    ----------
    variant : str
        Variant label being profiled.
    barcode : str
        Barcode of this variant being held out.
    non_wt_df : pl.DataFrame
        Every surviving non-wildtype row (every variant, every barcode).
    wt_train_df, wt_val_df, wt_test_df : pl.DataFrame
        The fixed wildtype train/val/test partition from :func:`prepare_data`.
    cfg : DictConfig
        Hydra config supplying ``label_column``, ``barcode_column``,
        ``wt_label``, ``min_cells``, ``min_cells_holdout``, ``random_state``,
        and the ``xgboost`` sub-config.

    Returns
    -------
    tuple[dict, xgb.Booster or None]
        ``(result_row, model)``. ``model`` is ``None`` for every non-``"ok"``
        ``status``.
    """
    label_col = cfg.label_column
    barcode_col = cfg.barcode_column

    try:
        variant_rows = non_wt_df.filter(pl.col(label_col) == variant)
        held_out_df = variant_rows.filter(pl.col(barcode_col) == barcode)
        held_in_df = variant_rows.filter(pl.col(barcode_col) != barcode)
        n_cells_test = len(held_out_df)
        n_cells_train_pool = len(held_in_df)

        if n_cells_test < cfg.min_cells_holdout:
            logging.info(
                "Skipping variant '%s' barcode '%s': held-out cell count %d "
                "< min_cells_holdout=%d",
                variant,
                barcode,
                n_cells_test,
                cfg.min_cells_holdout,
            )
            return (
                _result_row(
                    variant,
                    barcode,
                    "skipped_min_cells_holdout",
                    n_cells_test=n_cells_test,
                ),
                None,
            )

        if cfg.min_cells is not None and n_cells_train_pool < cfg.min_cells:
            logging.info(
                "Skipping variant '%s' barcode '%s': remaining training-pool "
                "cell count %d < min_cells=%d",
                variant,
                barcode,
                n_cells_train_pool,
                cfg.min_cells,
            )
            return (
                _result_row(
                    variant,
                    barcode,
                    "skipped_min_cells_train",
                    n_cells_test=n_cells_test,
                ),
                None,
            )

        n_barcodes_train = held_in_df.get_column(barcode_col).n_unique()
        held_in_train, held_in_val, held_out_df = split_variant_holdout(
            variant_rows, barcode, barcode_col, label_col, cfg.random_state
        )

        # barcode_col has done its job (separating held-in from held-out); it
        # must be dropped before get_dmatrix/train_binary_xgboost/evaluate_binary,
        # which treat every non-label_col column as a feature.
        train = pl.concat([held_in_train, wt_train_df], how="vertical").drop(
            barcode_col
        )
        val = pl.concat([held_in_val, wt_val_df], how="vertical").drop(barcode_col)
        test = pl.concat([held_out_df, wt_test_df], how="vertical").drop(barcode_col)

        logging.info(
            "Training model for variant '%s' holding out barcode '%s' — "
            "train: %d, val: %d, test: %d (%d held-out + %d WT)",
            variant,
            barcode,
            len(train),
            len(val),
            len(test),
            n_cells_test,
            len(wt_test_df),
        )

        model = train_binary_xgboost(train, val, label_col, cfg.wt_label, cfg)
        train_auroc, train_accuracy = evaluate_binary(
            train, model, label_col, cfg.wt_label
        )
        val_auroc, val_accuracy = evaluate_binary(val, model, label_col, cfg.wt_label)
        test_auroc, test_accuracy = evaluate_binary(
            test, model, label_col, cfg.wt_label
        )

        logging.info(
            "Results for variant '%s' holding out barcode '%s': "
            "train_auroc=%.4f, val_auroc=%.4f, test_auroc=%.4f",
            variant,
            barcode,
            train_auroc,
            val_auroc,
            test_auroc,
        )

        row = _result_row(
            variant,
            barcode,
            "ok",
            n_barcodes_train=n_barcodes_train,
            n_cells_train=len(train),
            n_cells_test=n_cells_test,
            n_wt_cells_test=len(wt_test_df),
            train_auroc=train_auroc,
            train_accuracy=train_accuracy,
            val_auroc=val_auroc,
            val_accuracy=val_accuracy,
            test_auroc=test_auroc,
            test_accuracy=test_accuracy,
        )
        return row, model
    except Exception:
        logging.warning(
            "Failed to profile variant '%s' holding out barcode '%s':\n%s",
            variant,
            barcode,
            traceback.format_exc(),
        )
        return _result_row(variant, barcode, "failed"), None


@hydra.main(version_base=None, config_path=None, config_name="ovwtlobo_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: leave-one-barcode-out OvWT generalization testing.

    Steps
    -----
    1. Read the feature file at ``cfg.input_file``.
    2. Build the once-per-batch wildtype train/val/test partition and the
       full non-wildtype pool via :func:`prepare_data`.
    3. For each non-wildtype variant: if it has fewer than
       ``cfg.min_barcodes_per_variant`` barcodes, emit one
       ``status="skipped_single_barcode"`` row; otherwise, for each of its
       barcodes, train and evaluate a held-out model via
       :func:`profile_holdout`.
    4. Write per-``(variant, barcode)`` results to ``results.parquet``.
    5. If ``cfg.save_models``, write every successfully trained model (keyed
       by ``(variant, barcode)``) to ``models.pkl``.
    6. Write per-``(variant, barcode)`` gain-based feature importance to
       ``feature_importance.parquet``.

    Output files
    ------------
    - ``{output_dir}/results.parquet`` -- see :data:`_RESULTS_SCHEMA` for
      columns. Joinable to :mod:`.ovwt`'s ``results.parquet`` on ``variant``
      (and ``barcode``, where applicable) to compute a generalization gap
      downstream; this module does not compute that gap itself.
    - ``{output_dir}/models.pkl`` (when ``cfg.save_models``)
    - ``{output_dir}/feature_importance.parquet`` (one row per successfully
      trained ``(variant, barcode)``, plus ``variant``/``barcode`` columns)

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.ovwtlobo \\
            output_dir=./out \\
            input_file=data/features.parquet \\
            wt_label=WT
    """
    ovwtlobo_cfg: OvwtLoboConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(ovwtlobo_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ovwtlobo_cfg.output_dir = output_dir
    setup_logging(ovwtlobo_cfg, "ovwtlobo")

    logging.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    logging.info("Loading input from %s", cfg.input_file)
    feature_df = load_batches(cfg.input_file)[0].collect()

    non_wt_df, wt_train_df, wt_val_df, wt_test_df = prepare_data(feature_df, cfg)
    logging.info(
        "Wildtype pool split — train: %d, val: %d, test: %d",
        len(wt_train_df),
        len(wt_val_df),
        len(wt_test_df),
    )
    if len(wt_test_df) == 0:
        logging.warning(
            "Wildtype test slice is empty after prepare_data() -- every fold "
            "in this run will have an undefined (NaN) test AUROC. Check "
            "downsample_wt/min-cell settings and the input data's wildtype "
            "cell count."
        )

    label_col = cfg.label_column
    barcode_col = cfg.barcode_column
    variants = sorted(non_wt_df.get_column(label_col).unique().to_list())
    logging.info("Found %d variant(s)", len(variants))

    results = []
    models = {}

    for variant in variants:
        variant_rows = non_wt_df.filter(pl.col(label_col) == variant)
        barcodes = sorted(variant_rows.get_column(barcode_col).unique().to_list())

        if len(barcodes) < cfg.min_barcodes_per_variant:
            logging.info(
                "Variant '%s' has %d barcode(s) (< min_barcodes_per_variant=%d) "
                "-- LOBO not applicable, skipping",
                variant,
                len(barcodes),
                cfg.min_barcodes_per_variant,
            )
            sole_barcode = barcodes[0] if len(barcodes) == 1 else None
            results.append(_result_row(variant, sole_barcode, "skipped_single_barcode"))
            continue

        for barcode in barcodes:
            row, model = profile_holdout(
                variant, barcode, non_wt_df, wt_train_df, wt_val_df, wt_test_df, cfg
            )
            results.append(row)
            if model is not None:
                models[(variant, barcode)] = model

    results_df = pl.DataFrame(results, schema=_RESULTS_SCHEMA)

    results_path = output_dir / "results.parquet"
    results_df.write_parquet(results_path)
    logging.info("Results written to %s", results_path)

    if ovwtlobo_cfg.save_models:
        models_path = output_dir / "models.pkl"
        logging.info("Writing models to %s", models_path)
        with open(models_path, "wb") as f:
            pickle.dump(models, f)

    logging.info("Computing feature importance")
    feature_cols = [c for c in non_wt_df.columns if c not in (label_col, barcode_col)]
    importance_dicts = []
    for (variant, barcode), model in models.items():
        importance = resolve_feature_importance(model, feature_cols)
        importance["variant"] = variant
        importance["barcode"] = barcode
        importance_dicts.append(importance)
    importance_df = (
        pl.from_dicts(importance_dicts)
        if importance_dicts
        else pl.DataFrame({"variant": [], "barcode": []})
    )
    importance_path = output_dir / "feature_importance.parquet"
    importance_df.write_parquet(importance_path)
    logging.info("Feature importance written to %s", importance_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
