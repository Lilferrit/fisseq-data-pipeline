import dataclasses
import functools
import logging
import pathlib
import pickle
import traceback
from os import PathLike
from typing import Optional, Union

import hydra
import numpy as np
import polars as pl
import sklearn.metrics
import sklearn.utils
import xgboost as xgb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .config import LabeledInputConfig
from .utils.batches import load_batches
from .utils.log import setup_logging
from .utils.metadata import get_aggregate_meta_data
from .utils.xgbparams import (
    XGBoostConfig,
    get_dmatrix,
    get_feature_cols,
    split_indices_stratified,
)


@dataclasses.dataclass
class OvwtConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the one-vs-wildtype entry point.

    Extends :class:`.config.LabeledInputConfig` with parameters controlling
    XGBoost training and data handling.

    Attributes
    ----------
    wt_label : str
        Label string identifying wildtype cells. Defaults to ``"WT"``.
    random_state : int
        Random seed for train/test/val splitting and WT downsampling.
        Defaults to ``42``.
    feature_cols : list or None
        Explicit list of feature column names. If ``None``, columns are
        auto-detected by :func:`get_feature_cols`. Defaults to ``None``.
    min_cells : int or None
        Minimum number of cells required for a variant to be included.
        Variants with fewer cells are dropped before splitting. ``None``
        disables the filter. Defaults to ``250``.
    downsample_wt : bool or int
        If ``True``, downsample wildtype cells to the size of the largest
        variant group before splitting. If an integer, downsample to that
        exact count (no-op if wildtype count is already at or below the
        target). ``False`` disables downsampling. Defaults to ``True``.
    save_splits : bool
        If ``True``, write lightweight train/test/val index files to
        ``output_dir``. Each file records the original row position and source
        file path for each cell in the split rather than duplicating the full
        feature matrix. Defaults to ``True``.
    xgboost : XGBoostConfig
        XGBoost training configuration. Defaults to :class:`XGBoostConfig`.
    """

    wt_label: str = "WT"
    random_state: int = 42
    feature_cols: Optional[list] = None
    min_cells: Optional[int] = 250
    downsample_wt: Union[bool, int] = True
    save_splits: bool = True
    xgboost: XGBoostConfig = dataclasses.field(default_factory=XGBoostConfig)


_cs = ConfigStore.instance()
_cs.store(name="ovwt_main", node=OvwtConfig)


def train_xgboost(
    train: pl.DataFrame,
    val: pl.DataFrame,
    cfg: DictConfig,
) -> xgb.Booster:
    """
    Train an XGBoost binary classifier on a variant-vs-wildtype split.

    Uses ``binary:logistic`` objective with AUC as the eval metric. Sample
    weights are computed with :func:`sklearn.utils.compute_sample_weight`
    when ``cfg.xgboost.weigh_samples`` is ``True``. Early stopping is applied
    against the validation set.

    Parameters
    ----------
    train : pl.DataFrame
        Training split containing feature columns and ``cfg.label_column``.
    val : pl.DataFrame
        Validation split used for early stopping and eval logging.
    cfg : DictConfig
        Hydra config supplying ``label_column``, ``wt_label``, ``random_state``,
        and the ``xgboost`` sub-config.

    Returns
    -------
    xgb.Booster
        Trained XGBoost booster at the best iteration.
    """
    label_col = cfg.label_column
    wt_label = cfg.wt_label

    y_train = train.get_column(label_col).to_numpy() == wt_label
    sample_weight = (
        sklearn.utils.compute_sample_weight("balanced", y_train)
        if cfg.xgboost.weigh_samples
        else None
    )

    dtrain = get_dmatrix(train, label_col, wt_label, weight=sample_weight)
    deval = get_dmatrix(val, label_col, wt_label)

    params = dict(cfg.xgboost.params)
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "auc"
    params["seed"] = cfg.random_state

    return xgb.train(
        params,
        dtrain,
        num_boost_round=cfg.xgboost.num_boost_round,
        evals=[(dtrain, "train"), (deval, "eval")],
        early_stopping_rounds=cfg.xgboost.early_stopping_rounds,
        verbose_eval=True,
    )


def evaluate(
    df: pl.DataFrame, model: xgb.Booster, label_col: str, wt_label: str
) -> tuple[float, float]:
    """
    Compute AUROC and accuracy for a trained model on a DataFrame split.

    Parameters
    ----------
    df : pl.DataFrame
        Split to evaluate. Must contain ``label_col`` and the same feature
        columns used during training.
    model : xgb.Booster
        Trained XGBoost booster.
    label_col : str
        Name of the label column.
    wt_label : str
        Wildtype label string passed to :func:`get_dmatrix`.

    Returns
    -------
    tuple[float, float]
        ``(auroc, accuracy)`` where accuracy uses a 0.5 probability threshold.
    """
    dmatrix = get_dmatrix(df, label_col, wt_label)
    y_true = dmatrix.get_label()
    y_prob = model.predict(dmatrix)
    auroc = sklearn.metrics.roc_auc_score(y_true, y_prob)
    accuracy = sklearn.metrics.accuracy_score(y_true, y_prob >= 0.5)

    return auroc, accuracy


def test_xgboost(
    model: xgb.Booster,
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    cfg: DictConfig,
) -> dict:
    """
    Evaluate a trained model on train, validation, and test splits.

    Parameters
    ----------
    model : xgb.Booster
        Trained XGBoost booster.
    train : pl.DataFrame
        Training split.
    val : pl.DataFrame
        Validation split.
    test : pl.DataFrame
        Held-out test split.
    cfg : DictConfig
        Hydra config supplying ``label_column`` and ``wt_label``.

    Returns
    -------
    dict
        Dictionary with keys ``variant``, ``train_auroc``, ``train_accuracy``,
        ``val_auroc``, ``val_accuracy``, ``test_auroc``, ``test_accuracy``.
    """
    label_col = cfg.label_column
    wt_label = cfg.wt_label

    variant = next(
        v for v in train.get_column(label_col).unique().to_list() if v != wt_label
    )

    evaluate_wrapper = functools.partial(
        evaluate, model=model, label_col=label_col, wt_label=wt_label
    )

    train_auroc, train_accuracy = evaluate_wrapper(train)
    val_auroc, val_accuracy = evaluate_wrapper(val)
    test_auroc, test_accuracy = evaluate_wrapper(test)

    return {
        "variant": variant,
        "train_auroc": train_auroc,
        "train_accuracy": train_accuracy,
        "val_auroc": val_auroc,
        "val_accuracy": val_accuracy,
        "test_auroc": test_auroc,
        "test_accuracy": test_accuracy,
    }


def read_feature_file(file_path: PathLike) -> pl.DataFrame:
    """
    Read a feature file (Parquet or CSV) into a Polars DataFrame.

    Parameters
    ----------
    file_path : PathLike
        Path to the file. Supported extensions: ``.parquet``, ``.pq``, ``.csv``.

    Returns
    -------
    pl.DataFrame
        Contents of the file.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    path = pathlib.Path(file_path)
    suffix = path.suffix.lower()
    if suffix in [".parquet", ".pq"]:
        return pl.read_parquet(path)
    elif suffix == ".csv":
        return pl.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix!r}. Expected .parquet or .csv"
        )


def downsample_wildtype(
    data_df: pl.DataFrame,
    label_col: str,
    wt_label: str,
    seed: int,
    n: Optional[int] = None,
) -> pl.DataFrame:
    """
    Downsample wildtype rows to a target count.

    If the wildtype group is larger than the target, a random sample of
    wildtype rows is drawn without replacement.

    Parameters
    ----------
    data_df : pl.DataFrame
        DataFrame containing all variant and wildtype rows.
    label_col : str
        Name of the label column.
    wt_label : str
        Label string identifying wildtype rows.
    seed : int
        Random seed for sampling.
    n : int or None
        Target wildtype count. If ``None``, the target is the size of the
        largest non-wildtype variant group.

    Returns
    -------
    pl.DataFrame
        DataFrame with wildtype rows downsampled, or unchanged if already
        at or below the target.
    """
    if n is None:
        target = (
            data_df.filter(pl.col(label_col) != wt_label)
            .group_by(label_col)
            .len()
            .get_column("len")
            .max()
        )
    else:
        target = n
    wt_df = data_df.filter(pl.col(label_col) == wt_label)
    if target is not None and len(wt_df) > target:
        wt_df = wt_df.sample(n=target, seed=seed)
    return pl.concat([data_df.filter(pl.col(label_col) != wt_label), wt_df])


def filter_min_cells(
    data_df: pl.DataFrame,
    label_col: str,
    wt_label: str,
    min_cells: int,
) -> pl.DataFrame:
    """
    Remove variant groups with fewer than ``min_cells`` cells.

    Wildtype rows are always retained regardless of count.

    Parameters
    ----------
    data_df : pl.DataFrame
        DataFrame containing all variant and wildtype rows.
    label_col : str
        Name of the label column.
    wt_label : str
        Label string identifying wildtype rows (always kept).
    min_cells : int
        Minimum number of cells a variant must have to be retained.

    Returns
    -------
    pl.DataFrame
        DataFrame with small variant groups removed.
    """
    variant_counts = (
        data_df.filter(pl.col(label_col) != wt_label).group_by(label_col).len()
    )
    keep_labels = (
        variant_counts.filter(pl.col("len") >= min_cells)
        .get_column(label_col)
        .to_list()
    )
    return data_df.filter(
        (pl.col(label_col) == wt_label) | pl.col(label_col).is_in(keep_labels)
    )


def train_test_val_split(
    data_df: pl.DataFrame,
    cfg: DictConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split a feature DataFrame into train, test, and validation sets.

    Optionally filters small variant groups via :func:`filter_min_cells` and
    downsamples wildtype via :func:`downsample_wildtype` before splitting.
    The 80/10/10 split is stratified by label.

    Parameters
    ----------
    data_df : pl.DataFrame
        Full feature DataFrame containing feature columns and ``cfg.label_column``.
    cfg : DictConfig
        Hydra config supplying ``label_column``, ``wt_label``, ``feature_cols``,
        ``min_cells``, ``downsample_wt``, and ``random_state``.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        ``(train, test, val)`` DataFrames, each containing feature columns,
        the label column, and a ``__row_idx__`` column recording the 0-based
        row position of each cell in the original ``data_df`` argument (before
        any filtering or downsampling). Callers that do not need the index
        should drop ``__row_idx__`` before passing splits to model-training
        functions.
    """
    label_col = cfg.label_column
    if cfg.feature_cols is not None:
        feature_cols = list(cfg.feature_cols)
    else:
        feature_cols = get_feature_cols(data_df)

    data_df = data_df.with_row_index("__row_idx__")
    select_cols = feature_cols + [label_col]
    data_df = data_df.select(select_cols + ["__row_idx__"])
    data_df = data_df.filter(pl.col(label_col).is_not_null())

    if cfg.min_cells is not None:
        data_df = filter_min_cells(data_df, label_col, cfg.wt_label, cfg.min_cells)

    if cfg.downsample_wt is not False and cfg.downsample_wt != 0:
        n = cfg.downsample_wt if not isinstance(cfg.downsample_wt, bool) else None
        data_df = downsample_wildtype(
            data_df, label_col, cfg.wt_label, cfg.random_state, n=n
        )

    data_df = data_df.with_row_index("__idx__")
    labels = data_df.get_column(label_col).to_numpy()

    train_idx, test_idx, val_idx = split_indices_stratified(labels, cfg.random_state)

    def select_rows(idx: np.ndarray) -> pl.DataFrame:
        return data_df.filter(pl.col("__idx__").is_in(idx)).select(
            select_cols + ["__row_idx__"]
        )

    return select_rows(train_idx), select_rows(test_idx), select_rows(val_idx)


def profile_variant(
    v: str,
    train_all: pl.DataFrame,
    test_all: pl.DataFrame,
    val_all: pl.DataFrame,
    cfg: DictConfig,
) -> tuple[dict, xgb.Booster]:
    """
    Train and evaluate an XGBoost model for one variant vs. wildtype.

    Subsets ``train_all``, ``test_all``, and ``val_all`` to rows belonging to
    variant ``v`` or the wildtype label, trains a model via
    :func:`train_xgboost`, and evaluates it via :func:`test_xgboost`.

    Parameters
    ----------
    v : str
        Variant label to profile.
    train_all : pl.DataFrame
        Full training split (all variants).
    test_all : pl.DataFrame
        Full test split (all variants).
    val_all : pl.DataFrame
        Full validation split (all variants).
    cfg : DictConfig
        Hydra config supplying ``label_column``, ``wt_label``, and XGBoost
        settings.

    Returns
    -------
    tuple[dict, xgb.Booster]
        ``(result_dict, model)`` where ``result_dict`` contains the evaluation
        metrics from :func:`test_xgboost`.
    """
    keep = pl.col(cfg.label_column).is_in([v, cfg.wt_label])
    train, test, val = (
        train_all.filter(keep),
        test_all.filter(keep),
        val_all.filter(keep),
    )
    logging.info(
        "Subset sizes — train: %d, val: %d, test: %d",
        len(train),
        len(val),
        len(test),
    )
    model = train_xgboost(train, val, cfg)
    result = test_xgboost(model, train, val, test, cfg)
    logging.info(
        "Results for '%s': train_auroc=%.4f, val_auroc=%.4f, test_auroc=%.4f",
        v,
        result["train_auroc"],
        result["val_auroc"],
        result["test_auroc"],
    )
    return result, model


@hydra.main(version_base=None, config_path=None, config_name="ovwt_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: one-vs-wildtype XGBoost variant profiling.

    Steps
    -----
    1. Read the feature file at ``cfg.input_file``.
    2. Split into train/test/val via :func:`train_test_val_split`.
    3. For each non-wildtype variant, train and evaluate an XGBoost binary
       classifier via :func:`profile_variant`. Variants that raise an exception
       are skipped with a warning.
    4. Write per-variant evaluation metrics to ``results.csv`` and all trained
       models (keyed by variant label) to ``models.pkl``.

    Output files
    ------------
    - ``{output_dir}/results.parquet``
    - ``{output_dir}/models.pkl``
    - ``{output_dir}/{train,test,val}_index.parquet`` (when ``save_splits`` is ``True``,
      which is the default; each file has columns ``row_idx`` and ``origin_file``)

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.ovwt \\
            output_dir=./out \\
            input_file=data/features.parquet \\
            wt_label=WT
    """
    ovwt_cfg: OvwtConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(ovwt_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ovwt_cfg.output_dir = output_dir
    setup_logging(ovwt_cfg, "ovwt")

    logging.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    logging.info("Loading input from %s", cfg.input_file)
    feature_df = load_batches(cfg.input_file)[0].collect()
    train_all, test_all, val_all = train_test_val_split(feature_df, cfg)
    unique_vars = train_all.get_column(cfg.label_column).unique().to_list()
    variants = [v for v in unique_vars if v != cfg.wt_label]

    logging.info("Found %d variant(s) to profile", len(variants))
    logging.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_all),
        len(val_all),
        len(test_all),
    )

    if cfg.save_splits:
        origin_file = str(pathlib.Path(cfg.input_file).resolve())
        for name, split_df in (
            ("train", train_all),
            ("test", test_all),
            ("val", val_all),
        ):
            index_path = output_dir / f"{name}_index.parquet"
            pl.DataFrame(
                {
                    "row_idx": split_df["__row_idx__"],
                    "origin_file": pl.Series([origin_file] * len(split_df)),
                }
            ).write_parquet(index_path)
            logging.info("Wrote %s index to %s", name, index_path)

    train_all = train_all.drop("__row_idx__")
    test_all = test_all.drop("__row_idx__")
    val_all = val_all.drop("__row_idx__")

    results = []
    models = {}

    for v in variants:
        logging.info("Training model for variant '%s' vs. '%s'", v, cfg.wt_label)
        try:
            result, model = profile_variant(v, train_all, test_all, val_all, cfg)
        except Exception:
            logging.warning(
                "Failed to profile variant '%s', skipping:\n%s",
                v,
                traceback.format_exc(),
            )
            continue
        results.append(result)
        models[v] = model

    results_df = pl.DataFrame(results)

    logging.info("Joining results with per-variant metadata")
    meta_df = (
        get_aggregate_meta_data(feature_df.lazy(), cfg.label_column)
        .collect()
        .rename({cfg.label_column: "variant"})
    )
    results_df = results_df.join(meta_df, on="variant", how="left")

    results_path = output_dir / "results.parquet"
    results_df.write_parquet(results_path)
    logging.info("Results written to %s", results_path)

    models_path = output_dir / "models.pkl"
    logging.info("Writing models to %s", models_path)
    with open(models_path, "wb") as f:
        pickle.dump(models, f)

    logging.info("Done")


if __name__ == "__main__":
    main()
