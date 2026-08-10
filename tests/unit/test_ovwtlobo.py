from unittest.mock import patch

import numpy as np
import polars as pl
from omegaconf import OmegaConf

import fisseq_data_pipeline.ovwtlobo as m
from fisseq_data_pipeline.ovwtlobo import (
    OvwtLoboConfig,
    _exclude_blocked_barcodes,
    downsample_per_barcode,
    downsample_wildtype,
    prepare_data,
    profile_holdout,
    split_variant_holdout,
)
from fisseq_data_pipeline.utils.xgbparams import XGBoostConfig, XGBoostParams

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _make_barcode_df(
    wt_barcode_counts: dict[str, int],
    variant_barcode_counts: dict[str, dict[str, int]],
    lo_wt: float = 0.6,
    hi_wt: float = 1.0,
    lo_variant: float = 0.0,
    hi_variant: float = 0.4,
    seed: int = 0,
) -> pl.DataFrame:
    """Build a per-barcode DataFrame. ``variant_barcode_counts`` maps
    variant label -> {barcode: n_cells}. Intensity_Mean is linearly
    separable between WT (high) and every variant (low), so a well-formed
    LOBO fold should reach a high AUROC."""
    rng = np.random.default_rng(seed)
    barcodes: list[str] = []
    labels: list[str] = []
    for bc, n in wt_barcode_counts.items():
        barcodes.extend([bc] * n)
        labels.extend(["WT"] * n)
    for variant, bc_counts in variant_barcode_counts.items():
        for bc, n in bc_counts.items():
            barcodes.extend([bc] * n)
            labels.extend([variant] * n)
    n_total = len(labels)
    is_wt = np.array([label == "WT" for label in labels])
    intensity = np.where(
        is_wt,
        rng.uniform(lo_wt, hi_wt, n_total),
        rng.uniform(lo_variant, hi_variant, n_total),
    )
    return pl.DataFrame(
        {
            "Intensity_Mean": intensity.tolist(),
            "Texture_Var": rng.random(n_total).tolist(),
            "label": labels,
            "meta_barcode": barcodes,
        }
    )


def _split_cfg(**overrides) -> OmegaConf:
    base = {
        "label_column": "label",
        "wt_label": "WT",
        "barcode_column": "meta_barcode",
        "random_state": 0,
        "feature_cols": None,
        "min_cells": None,
        "min_cells_holdout": 0,
        "min_barcodes_per_variant": 2,
        "downsample_wt": False,
        "max_cells_per_barcode_wt": None,
        "max_cells_per_barcode_variant": None,
        "save_models": True,
        "feature_block_list_file": None,
        "barcode_block_list_file": None,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _make_xgb_cfg(weigh_samples: bool = True, **overrides) -> OmegaConf:
    cfg = _split_cfg(**overrides)
    return OmegaConf.merge(
        cfg,
        {
            "xgboost": {
                "num_boost_round": 5,
                "early_stopping_rounds": 3,
                "weigh_samples": weigh_samples,
                "params": {
                    "nthread": 1,
                    "max_depth": 2,
                    "colsample_bytree": 1.0,
                    "colsample_bylevel": 1.0,
                    "colsample_bynode": 1.0,
                    "subsample": 1.0,
                },
            }
        },
    )


def _default_df() -> pl.DataFrame:
    """3 WT barcodes (100 cells each), V1 with 3 barcodes (150 cells each),
    V2 with a single barcode (150 cells)."""
    return _make_barcode_df(
        wt_barcode_counts={"WTb0": 100, "WTb1": 100, "WTb2": 100},
        variant_barcode_counts={
            "V1": {"V1b0": 150, "V1b1": 150, "V1b2": 150},
            "V2": {"V2b0": 150},
        },
    )


# ---------------------------------------------------------------------------
# _exclude_blocked_barcodes / downsample_per_barcode / downsample_wildtype
# (copied from .ovwt -- smoke-test they still behave identically)
# ---------------------------------------------------------------------------


def test_exclude_blocked_barcodes_no_file_is_noop():
    df = _default_df()
    result = _exclude_blocked_barcodes(df, "meta_barcode", None)
    assert len(result) == len(df)


def test_exclude_blocked_barcodes_drops_blocked(tmp_path):
    df = _default_df()
    bl_path = tmp_path / "barcode_blocklist.parquet"
    pl.DataFrame(
        {"barcode": ["V1b0", "V1b1"], "barcode_ok": [False, True]}
    ).write_parquet(bl_path)
    result = _exclude_blocked_barcodes(df, "meta_barcode", str(bl_path))
    assert "V1b0" not in result.get_column("meta_barcode").to_list()
    assert "V1b1" in result.get_column("meta_barcode").to_list()


def test_downsample_per_barcode_caps_variant_barcode():
    df = _default_df()
    result = downsample_per_barcode(
        df, "meta_barcode", "label", "WT", seed=0, max_cells_variant=50
    )
    counts = result.filter(pl.col("label") == "V1").group_by("meta_barcode").len()
    assert counts["len"].max() == 50


def test_downsample_wildtype_targets_largest_variant_group():
    df = _default_df()  # WT: 300 total, V1: 450, V2: 150
    result = downsample_wildtype(df, "label", "WT", seed=0)
    # auto target = largest *other* group's count = 450 (V1)
    assert (result.get_column("label") == "WT").sum() == 300  # WT already below target


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------


def test_prepare_data_returns_non_wt_and_wt_splits():
    df = _default_df()
    cfg = _split_cfg()
    non_wt_df, wt_train, wt_val, wt_test = prepare_data(df, cfg)
    assert (non_wt_df.get_column("label") != "WT").all()
    for split in (wt_train, wt_val, wt_test):
        assert (split.get_column("label") == "WT").all()


def test_prepare_data_wt_splits_disjoint_and_cover_pool():
    df = _default_df()
    cfg = _split_cfg()
    _, wt_train, wt_val, wt_test = prepare_data(df, cfg)
    train_rows = set(wt_train.get_column("Intensity_Mean").to_list())
    val_rows = set(wt_val.get_column("Intensity_Mean").to_list())
    test_rows = set(wt_test.get_column("Intensity_Mean").to_list())
    assert train_rows.isdisjoint(val_rows)
    assert train_rows.isdisjoint(test_rows)
    assert val_rows.isdisjoint(test_rows)
    assert len(wt_train) + len(wt_val) + len(wt_test) == 300


def test_prepare_data_feature_block_list_excludes_feature(tmp_path):
    df = _default_df()
    bl_path = tmp_path / "feature_blocklist.parquet"
    pl.DataFrame({"feature": ["Texture_Var"], "feature_ok": [False]}).write_parquet(
        bl_path
    )
    cfg = _split_cfg(feature_block_list_file=str(bl_path))
    non_wt_df, wt_train, _, _ = prepare_data(df, cfg)
    assert "Texture_Var" not in non_wt_df.columns
    assert "Texture_Var" not in wt_train.columns
    assert "Intensity_Mean" in non_wt_df.columns


def test_prepare_data_barcode_block_list_excludes_barcode(tmp_path):
    df = _default_df()
    bl_path = tmp_path / "barcode_blocklist.parquet"
    pl.DataFrame({"barcode": ["V1b0"], "barcode_ok": [False]}).write_parquet(bl_path)
    cfg = _split_cfg(barcode_block_list_file=str(bl_path))
    non_wt_df, _, _, _ = prepare_data(df, cfg)
    assert "V1b0" not in non_wt_df.get_column("meta_barcode").to_list()


def test_prepare_data_max_cells_per_barcode_variant_caps_rows():
    df = _default_df()
    cfg = _split_cfg(max_cells_per_barcode_variant=20)
    non_wt_df, _, _, _ = prepare_data(df, cfg)
    counts = non_wt_df.group_by("meta_barcode").len()
    assert counts["len"].max() == 20


def test_prepare_data_downsample_wt_true_matches_largest_variant_group():
    df = _default_df()  # WT: 300, V1: 450 (largest), V2: 150
    cfg = _split_cfg(downsample_wt=True)
    _, wt_train, wt_val, wt_test = prepare_data(df, cfg)
    assert len(wt_train) + len(wt_val) + len(wt_test) == 300  # already <= 450, no-op


def test_prepare_data_downsample_wt_int_exact_count():
    df = _default_df()
    cfg = _split_cfg(downsample_wt=100)
    _, wt_train, wt_val, wt_test = prepare_data(df, cfg)
    assert len(wt_train) + len(wt_val) + len(wt_test) == 100


# ---------------------------------------------------------------------------
# split_variant_holdout
# ---------------------------------------------------------------------------


def test_split_variant_holdout_held_out_rows_excluded_from_train_and_val():
    df = _default_df()
    cfg = _split_cfg()
    non_wt_df, _, _, _ = prepare_data(df, cfg)
    variant_rows = non_wt_df.filter(pl.col("label") == "V1")
    train, val, held_out = split_variant_holdout(
        variant_rows, "V1b0", "meta_barcode", "label", random_state=0
    )
    assert "V1b0" not in train.get_column("meta_barcode").to_list()
    assert "V1b0" not in val.get_column("meta_barcode").to_list()
    assert (held_out.get_column("meta_barcode") == "V1b0").all()
    assert len(held_out) == 150


def test_split_variant_holdout_train_val_cover_held_in_rows():
    df = _default_df()
    cfg = _split_cfg()
    non_wt_df, _, _, _ = prepare_data(df, cfg)
    variant_rows = non_wt_df.filter(pl.col("label") == "V1")
    train, val, held_out = split_variant_holdout(
        variant_rows, "V1b0", "meta_barcode", "label", random_state=0
    )
    assert len(train) + len(val) == len(variant_rows) - len(held_out)


# ---------------------------------------------------------------------------
# profile_holdout
# ---------------------------------------------------------------------------


def test_profile_holdout_ok_status_and_nonnull_metrics():
    df = _default_df()
    cfg = _make_xgb_cfg(min_cells=50, min_cells_holdout=20)
    non_wt_df, wt_train, wt_val, wt_test = prepare_data(df, cfg)
    row, model = profile_holdout(
        "V1", "V1b0", non_wt_df, wt_train, wt_val, wt_test, cfg
    )
    assert row["status"] == "ok"
    assert model is not None
    for key in ("train_auroc", "val_auroc", "test_auroc"):
        assert row[key] is not None
        assert row[key] == row[key]  # not NaN
        assert 0.0 <= row[key] <= 1.0


def test_profile_holdout_separable_data_high_test_auroc():
    df = _default_df()
    cfg = _make_xgb_cfg(min_cells=50, min_cells_holdout=20)
    non_wt_df, wt_train, wt_val, wt_test = prepare_data(df, cfg)
    row, _ = profile_holdout("V1", "V1b0", non_wt_df, wt_train, wt_val, wt_test, cfg)
    assert row["test_auroc"] > 0.9


def test_profile_holdout_test_set_never_contains_other_barcodes_variant_rows():
    """Regression coverage for the held-out test set being exactly the held-out
    barcode's variant rows plus the shared WT test slice -- nothing from any
    other barcode of the same variant."""
    df = _default_df()
    cfg = _make_xgb_cfg(min_cells=50, min_cells_holdout=20)
    non_wt_df, wt_train, wt_val, wt_test = prepare_data(df, cfg)
    row, _ = profile_holdout("V1", "V1b0", non_wt_df, wt_train, wt_val, wt_test, cfg)
    assert row["n_cells_test"] == 150
    assert row["n_wt_cells_test"] == len(wt_test)


def test_profile_holdout_skips_below_min_cells_holdout():
    df = _default_df()
    cfg = _make_xgb_cfg(min_cells_holdout=1000)
    non_wt_df, wt_train, wt_val, wt_test = prepare_data(df, cfg)
    row, model = profile_holdout(
        "V1", "V1b0", non_wt_df, wt_train, wt_val, wt_test, cfg
    )
    assert row["status"] == "skipped_min_cells_holdout"
    assert row["n_cells_test"] == 150
    assert model is None
    assert row["train_auroc"] is None


def test_profile_holdout_skips_below_min_cells_train():
    df = _default_df()
    cfg = _make_xgb_cfg(min_cells=1000, min_cells_holdout=0)
    non_wt_df, wt_train, wt_val, wt_test = prepare_data(df, cfg)
    row, model = profile_holdout(
        "V1", "V1b0", non_wt_df, wt_train, wt_val, wt_test, cfg
    )
    assert row["status"] == "skipped_min_cells_train"
    assert model is None


def test_profile_holdout_exception_gives_failed_status():
    df = _default_df()
    cfg = _make_xgb_cfg(min_cells=50, min_cells_holdout=20)
    non_wt_df, wt_train, wt_val, wt_test = prepare_data(df, cfg)
    with patch(
        "fisseq_data_pipeline.ovwtlobo.train_binary_xgboost",
        side_effect=RuntimeError("boom"),
    ):
        row, model = profile_holdout(
            "V1", "V1b0", non_wt_df, wt_train, wt_val, wt_test, cfg
        )
    assert row["status"] == "failed"
    assert model is None
    assert row["train_auroc"] is None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _make_ovwtlobo_structured_cfg(tmp_path, input_path, **overrides) -> OmegaConf:
    xgb_params = XGBoostParams(
        nthread=1,
        max_depth=2,
        colsample_bytree=1.0,
        colsample_bylevel=1.0,
        colsample_bynode=1.0,
        subsample=1.0,
    )
    xgb_cfg = XGBoostConfig(
        num_boost_round=5,
        early_stopping_rounds=3,
        weigh_samples=False,
        params=xgb_params,
    )
    defaults = dict(
        output_dir=str(tmp_path),
        input_file=str(input_path),
        label_column="label",
        wt_label="WT",
        barcode_column="meta_barcode",
        random_state=0,
        min_cells=50,
        min_cells_holdout=20,
        min_barcodes_per_variant=2,
        downsample_wt=False,
        xgboost=xgb_cfg,
    )
    defaults.update(overrides)
    cfg = OvwtLoboConfig(**defaults)
    return OmegaConf.structured(cfg)


def _run_main(cfg):
    with patch("fisseq_data_pipeline.ovwtlobo.setup_logging"):
        m.main.__wrapped__(cfg)


def test_main_single_barcode_variant_flagged_not_dropped(tmp_path):
    df = _default_df()
    input_path = tmp_path / "input.parquet"
    df.write_parquet(input_path)
    cfg = _make_ovwtlobo_structured_cfg(tmp_path, input_path)
    _run_main(cfg)
    results = pl.read_parquet(tmp_path / "results.parquet")
    v2_rows = results.filter(pl.col("variant") == "V2")
    assert len(v2_rows) == 1
    assert v2_rows["status"].to_list() == ["skipped_single_barcode"]
    assert v2_rows["barcode"].to_list() == ["V2b0"]


def test_main_multi_barcode_variant_one_row_per_held_out_barcode(tmp_path):
    df = _default_df()
    input_path = tmp_path / "input.parquet"
    df.write_parquet(input_path)
    cfg = _make_ovwtlobo_structured_cfg(tmp_path, input_path)
    _run_main(cfg)
    results = pl.read_parquet(tmp_path / "results.parquet")
    v1_rows = results.filter(pl.col("variant") == "V1")
    assert set(v1_rows["barcode"].to_list()) == {"V1b0", "V1b1", "V1b2"}
    assert set(v1_rows["status"].to_list()) == {"ok"}
    assert (v1_rows["test_auroc"] > 0.9).all()


def test_main_writes_expected_output_files(tmp_path):
    df = _default_df()
    input_path = tmp_path / "input.parquet"
    df.write_parquet(input_path)
    cfg = _make_ovwtlobo_structured_cfg(tmp_path, input_path)
    _run_main(cfg)
    assert (tmp_path / "results.parquet").exists()
    assert (tmp_path / "models.pkl").exists()
    assert (tmp_path / "feature_importance.parquet").exists()
    importance = pl.read_parquet(tmp_path / "feature_importance.parquet")
    assert set(importance.columns) - {"variant", "barcode"} <= {
        "Intensity_Mean",
        "Texture_Var",
    }


def test_main_save_models_false_skips_models_pkl(tmp_path):
    df = _default_df()
    input_path = tmp_path / "input.parquet"
    df.write_parquet(input_path)
    cfg = _make_ovwtlobo_structured_cfg(tmp_path, input_path, save_models=False)
    _run_main(cfg)
    assert not (tmp_path / "models.pkl").exists()
    assert (tmp_path / "results.parquet").exists()


def test_main_all_skipped_results_schema_stays_float(tmp_path):
    """Regression coverage: when every fold is skipped/failed, results.parquet's
    metric columns must still be Float64, not Null (see _RESULTS_SCHEMA)."""
    df = _default_df()
    input_path = tmp_path / "input.parquet"
    df.write_parquet(input_path)
    cfg = _make_ovwtlobo_structured_cfg(tmp_path, input_path, min_cells_holdout=100000)
    _run_main(cfg)
    results = pl.read_parquet(tmp_path / "results.parquet")
    assert (results["status"] != "ok").all()
    for col in ("train_auroc", "val_auroc", "test_auroc"):
        assert results.schema[col] == pl.Float64


def test_main_test_auroc_never_nan(tmp_path):
    df = _default_df()
    input_path = tmp_path / "input.parquet"
    df.write_parquet(input_path)
    cfg = _make_ovwtlobo_structured_cfg(tmp_path, input_path)
    _run_main(cfg)
    results = pl.read_parquet(tmp_path / "results.parquet")
    ok_rows = results.filter(pl.col("status") == "ok")
    assert len(ok_rows) > 0
    assert not ok_rows["test_auroc"].is_nan().any()
