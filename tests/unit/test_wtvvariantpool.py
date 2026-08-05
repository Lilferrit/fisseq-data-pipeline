from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
import xgboost as xgb
from omegaconf import OmegaConf

import fisseq_data_pipeline.wtvvariantpool as m
from fisseq_data_pipeline.utils.xgbparams import XGBoostConfig, XGBoostParams
from fisseq_data_pipeline.wtvvariantpool import (
    _POOL_GROUP,
    WtvvariantpoolConfig,
    build_variant_pool,
    evaluate,
    evaluate_barcode,
    profile_barcode,
    train_test_val_split,
    train_xgboost,
)

_LABEL_COL = "label"
_BARCODE_COL = "meta_barcode"


def _make_cells_df(
    wt_barcode_counts: dict[str, int],
    variant_counts: dict[str, int],
    wt_label: str = "WT",
) -> pl.DataFrame:
    """Build a DataFrame with wildtype barcodes plus arbitrary variant labels."""
    rng = np.random.default_rng(0)
    labels: list[str] = []
    barcodes: list[str] = []
    for bc, n in wt_barcode_counts.items():
        labels.extend([wt_label] * n)
        barcodes.extend([bc] * n)
    for v, n in variant_counts.items():
        labels.extend([v] * n)
        barcodes.extend([f"bc_{v}"] * n)
    n_total = len(labels)
    return pl.DataFrame(
        {
            "Intensity_Mean": rng.random(n_total).tolist(),
            "Texture_Var": rng.random(n_total).tolist(),
            _LABEL_COL: labels,
            _BARCODE_COL: barcodes,
        }
    )


def _split_cfg(**overrides) -> OmegaConf:
    cfg = dict(
        label_column=_LABEL_COL,
        wt_label="WT",
        barcode_column=_BARCODE_COL,
        random_state=0,
        feature_cols=None,
        min_cells_per_barcode=0,
        variant_classes=["Synonymous"],
        downsample_variant_pool=None,
        feature_block_list_file=None,
    )
    cfg.update(overrides)
    return OmegaConf.create(cfg)


# ---------------------------------------------------------------------------
# build_variant_pool
# ---------------------------------------------------------------------------


def test_build_variant_pool_excludes_wt_rows():
    df = _make_cells_df({"bc1": 10}, {"A1A": 10, "M1K": 10})
    pool = build_variant_pool(df, _LABEL_COL, "WT", ["Synonymous", "Single Missense"])
    assert "WT" not in pool.get_column(_LABEL_COL).to_list()


def test_build_variant_pool_filters_to_requested_classes():
    # A1A -> Synonymous (A->A), M1K -> Single Missense.
    df = _make_cells_df({"bc1": 10}, {"A1A": 10, "M1K": 10})
    pool = build_variant_pool(df, _LABEL_COL, "WT", ["Synonymous"])
    assert set(pool.get_column(_LABEL_COL).to_list()) == {"A1A"}


def test_build_variant_pool_custom_class_selection():
    df = _make_cells_df({"bc1": 10}, {"A1A": 10, "M1K": 10})
    pool = build_variant_pool(df, _LABEL_COL, "WT", ["Single Missense"])
    assert set(pool.get_column(_LABEL_COL).to_list()) == {"M1K"}


def test_build_variant_pool_empty_when_no_classes_match():
    df = _make_cells_df({"bc1": 10}, {"A1A": 10})
    pool = build_variant_pool(df, _LABEL_COL, "WT", ["Nonsense"])
    assert len(pool) == 0


def test_build_variant_pool_ignores_configurable_wt_label():
    # wt_label need not be the literal string "WT"; only exact-match rows
    # are excluded, regardless of what classify_variant would say about them.
    df = _make_cells_df({"bc1": 10}, {"A1A": 10}, wt_label="WT_control")
    pool = build_variant_pool(df, _LABEL_COL, "WT_control", ["Synonymous"])
    assert "WT_control" not in pool.get_column(_LABEL_COL).to_list()
    assert set(pool.get_column(_LABEL_COL).to_list()) == {"A1A"}


# ---------------------------------------------------------------------------
# train_test_val_split
# ---------------------------------------------------------------------------


def test_train_test_val_split_pools_variant_rows():
    df = _make_cells_df({"bc1": 30, "bc2": 30}, {"A1A": 30})
    train, test, val = train_test_val_split(df, _split_cfg())
    for split in (train, test, val):
        assert _POOL_GROUP in split.get_column(_BARCODE_COL).to_list()


def test_train_test_val_split_drops_barcodes_below_min_cells():
    df = _make_cells_df({"bc1": 30, "bc2": 30, "bc3": 2}, {"A1A": 30})
    train, test, val = train_test_val_split(
        df, _split_cfg(min_cells_per_barcode=5)
    )
    for split in (train, test, val):
        assert "bc3" not in split.get_column(_BARCODE_COL).to_list()


def test_train_test_val_split_every_barcode_and_pool_present_in_every_split():
    df = _make_cells_df({f"bc{i}": 30 for i in range(4)}, {"A1A": 30})
    train, test, val = train_test_val_split(df, _split_cfg())
    for split in (train, test, val):
        groups = set(split.get_column(_BARCODE_COL).to_list())
        assert groups == {"bc0", "bc1", "bc2", "bc3", _POOL_GROUP}


def test_train_test_val_split_no_overlap():
    df = _make_cells_df({f"bc{i}": 30 for i in range(4)}, {"A1A": 30})
    train, test, val = train_test_val_split(df, _split_cfg())
    train_rows = set(map(tuple, train.select("Intensity_Mean", _BARCODE_COL).rows()))
    test_rows = set(map(tuple, test.select("Intensity_Mean", _BARCODE_COL).rows()))
    val_rows = set(map(tuple, val.select("Intensity_Mean", _BARCODE_COL).rows()))
    assert train_rows.isdisjoint(test_rows)
    assert train_rows.isdisjoint(val_rows)
    assert test_rows.isdisjoint(val_rows)


def test_train_test_val_split_excludes_non_feature_columns():
    df = _make_cells_df({"bc1": 30, "bc2": 30}, {"A1A": 30}).with_columns(
        pl.lit(0).alias("lowercase_extra")
    )
    train, test, val = train_test_val_split(df, _split_cfg())
    for split in (train, test, val):
        assert "lowercase_extra" not in split.columns
        assert set(split.columns) == {"Intensity_Mean", "Texture_Var", _BARCODE_COL}


def test_train_test_val_split_feature_block_list_file_excludes_feature(tmp_path):
    df = _make_cells_df({"bc1": 30, "bc2": 30}, {"A1A": 30})
    path = tmp_path / "feature_block_list.parquet"
    pl.DataFrame(
        {"feature": ["Intensity_Mean", "Texture_Var"], "feature_ok": [True, False]}
    ).write_parquet(path)
    cfg = _split_cfg(feature_block_list_file=str(path))
    train, test, val = train_test_val_split(df, cfg)
    for split in (train, test, val):
        assert "Texture_Var" not in split.columns
        assert "Intensity_Mean" in split.columns


def test_train_test_val_split_empty_pool_raises():
    df = _make_cells_df({"bc1": 30, "bc2": 30}, {"A1A": 30})
    with pytest.raises(ValueError, match="pool is empty"):
        train_test_val_split(df, _split_cfg(variant_classes=["Nonsense"]))


def test_train_test_val_split_sentinel_collision_raises():
    df = _make_cells_df({_POOL_GROUP: 30, "bc2": 30}, {"A1A": 30})
    with pytest.raises(ValueError, match="collides"):
        train_test_val_split(df, _split_cfg())


def test_train_test_val_split_downsample_true_matches_largest_barcode():
    df = _make_cells_df({"bc1": 40, "bc2": 20}, {"A1A": 200})
    train, test, val = train_test_val_split(
        df, _split_cfg(downsample_variant_pool=True)
    )
    all_rows = pl.concat([train, test, val])
    pool_count = all_rows.filter(pl.col(_BARCODE_COL) == _POOL_GROUP).height
    assert pool_count == 40


def test_train_test_val_split_downsample_int_exact_count():
    df = _make_cells_df({"bc1": 40, "bc2": 20}, {"A1A": 200})
    train, test, val = train_test_val_split(
        df, _split_cfg(downsample_variant_pool=50)
    )
    all_rows = pl.concat([train, test, val])
    pool_count = all_rows.filter(pl.col(_BARCODE_COL) == _POOL_GROUP).height
    assert pool_count == 50


def test_train_test_val_split_downsample_none_is_no_op():
    df = _make_cells_df({"bc1": 40, "bc2": 20}, {"A1A": 200})
    train, test, val = train_test_val_split(
        df, _split_cfg(downsample_variant_pool=None)
    )
    all_rows = pl.concat([train, test, val])
    pool_count = all_rows.filter(pl.col(_BARCODE_COL) == _POOL_GROUP).height
    assert pool_count == 200


def test_train_test_val_split_downsample_false_is_no_op():
    df = _make_cells_df({"bc1": 40, "bc2": 20}, {"A1A": 200})
    train, test, val = train_test_val_split(
        df, _split_cfg(downsample_variant_pool=False)
    )
    all_rows = pl.concat([train, test, val])
    pool_count = all_rows.filter(pl.col(_BARCODE_COL) == _POOL_GROUP).height
    assert pool_count == 200


# ---------------------------------------------------------------------------
# train_xgboost / evaluate / evaluate_barcode
# ---------------------------------------------------------------------------


def _make_xgb_cfg(weigh_samples: bool = True) -> OmegaConf:
    return OmegaConf.create(
        {
            "random_state": 0,
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
            },
        }
    )


def _make_separable_barcode_vs_pool_df(n: int = 60) -> pl.DataFrame:
    """Linearly separable: barcode has high Intensity_Mean, pool has low."""
    rng = np.random.default_rng(42)
    half = n // 2
    return pl.DataFrame(
        {
            "Intensity_Mean": np.concatenate(
                [
                    rng.uniform(0.6, 1.0, half),  # bc_a
                    rng.uniform(0.0, 0.4, half),  # pool
                ]
            ).tolist(),
            "Texture_Var": rng.random(n).tolist(),
            _BARCODE_COL: ["bc_a"] * half + [_POOL_GROUP] * half,
        }
    )


@pytest.fixture()
def trained_model_and_splits():
    df = _make_separable_barcode_vs_pool_df(n=60)
    cfg = _make_xgb_cfg()
    a = df.filter(pl.col(_BARCODE_COL) == "bc_a")
    pool = df.filter(pl.col(_BARCODE_COL) == _POOL_GROUP)
    train = pl.concat([a[:20], pool[:20]])
    val = pl.concat([a[20:25], pool[20:25]])
    test = pl.concat([a[25:], pool[25:]])
    model = train_xgboost(train, val, _BARCODE_COL, "bc_a", cfg)
    return model, train, val, test, cfg


def test_train_xgboost_returns_booster():
    df = _make_separable_barcode_vs_pool_df(n=60)
    cfg = _make_xgb_cfg()
    half = len(df) // 2
    model = train_xgboost(df[:half], df[half:], _BARCODE_COL, "bc_a", cfg)
    assert isinstance(model, xgb.Booster)


def test_train_xgboost_without_sample_weights():
    df = _make_separable_barcode_vs_pool_df(n=60)
    cfg = _make_xgb_cfg(weigh_samples=False)
    half = len(df) // 2
    model = train_xgboost(df[:half], df[half:], _BARCODE_COL, "bc_a", cfg)
    assert isinstance(model, xgb.Booster)


def test_evaluate_auroc_in_range(trained_model_and_splits):
    model, train, val, test, _ = trained_model_and_splits
    for split in (train, val, test):
        auroc, accuracy = evaluate(split, model, _BARCODE_COL, "bc_a")
        assert 0.0 <= auroc <= 1.0
        assert 0.0 <= accuracy <= 1.0


def test_evaluate_high_auroc_on_separable_data(trained_model_and_splits):
    model, train, _, _, _ = trained_model_and_splits
    auroc, _ = evaluate(train, model, _BARCODE_COL, "bc_a")
    assert auroc > 0.9


def test_evaluate_barcode_result_keys(trained_model_and_splits):
    model, train, val, test, _ = trained_model_and_splits
    result = evaluate_barcode(model, train, val, test, _BARCODE_COL, "bc_a")
    expected_keys = {
        "barcode",
        "train_auroc",
        "train_accuracy",
        "val_auroc",
        "val_accuracy",
        "test_auroc",
        "test_accuracy",
        "n_cells_barcode",
        "n_cells_pool",
    }
    assert set(result.keys()) == expected_keys


def test_evaluate_barcode_name(trained_model_and_splits):
    model, train, val, test, _ = trained_model_and_splits
    result = evaluate_barcode(model, train, val, test, _BARCODE_COL, "bc_a")
    assert result["barcode"] == "bc_a"


def test_evaluate_barcode_n_cells(trained_model_and_splits):
    model, train, val, test, _ = trained_model_and_splits
    result = evaluate_barcode(model, train, val, test, _BARCODE_COL, "bc_a")
    assert result["n_cells_barcode"] == 30
    assert result["n_cells_pool"] == 30


# ---------------------------------------------------------------------------
# profile_barcode
# ---------------------------------------------------------------------------


def _make_multi_barcode_vs_pool_split(n_each: int = 30):
    """
    bc_a is linearly separable from the pool (high vs low Intensity_Mean).
    bc_b shares the pool's distribution (not separable from it).
    """
    rng = np.random.default_rng(42)
    half = n_each // 2

    def _block(low: float, high: float, barcode: str, n: int) -> dict:
        return {
            "Intensity_Mean": rng.uniform(low, high, n).tolist(),
            _BARCODE_COL: [barcode] * n,
        }

    def _split(rows: list[dict]) -> pl.DataFrame:
        return pl.concat([pl.DataFrame(r) for r in rows])

    train = _split(
        [
            _block(0.6, 1.0, "bc_a", n_each),
            _block(0.0, 0.4, "bc_b", n_each),
            _block(0.0, 0.4, _POOL_GROUP, n_each),
        ]
    )
    val = _split(
        [
            _block(0.6, 1.0, "bc_a", half),
            _block(0.0, 0.4, "bc_b", half),
            _block(0.0, 0.4, _POOL_GROUP, half),
        ]
    )
    test = _split(
        [
            _block(0.6, 1.0, "bc_a", half),
            _block(0.0, 0.4, "bc_b", half),
            _block(0.0, 0.4, _POOL_GROUP, half),
        ]
    )
    return train, val, test


@pytest.fixture()
def profile_barcode_splits():
    train, val, test = _make_multi_barcode_vs_pool_split()
    cfg = _make_xgb_cfg()
    cfg.barcode_column = _BARCODE_COL
    return train, val, test, cfg


def test_profile_barcode_returns_booster(profile_barcode_splits):
    train, val, test, cfg = profile_barcode_splits
    result, model = profile_barcode("bc_a", train, test, val, cfg)
    assert isinstance(model, xgb.Booster)


def test_profile_barcode_high_auroc_on_separable_barcode(profile_barcode_splits):
    train, val, test, cfg = profile_barcode_splits
    result, _ = profile_barcode("bc_a", train, test, val, cfg)
    assert result["train_auroc"] > 0.9


def test_profile_barcode_low_auroc_on_indistinguishable_barcode(
    profile_barcode_splits,
):
    """bc_b shares the pool's distribution, so AUROC should be near chance."""
    train, val, test, cfg = profile_barcode_splits
    result, _ = profile_barcode("bc_b", train, test, val, cfg)
    assert result["train_auroc"] < 0.8


def test_profile_barcode_filters_to_target_barcode_and_pool_only(
    profile_barcode_splits,
):
    # n_cells_barcode/n_cells_pool count rows across train+val+test
    # (30 + 15 + 15 = 60, per _make_multi_barcode_vs_pool_split's
    # n_each=30/half=15), with no leakage from the other WT barcode.
    train, val, test, cfg = profile_barcode_splits
    result, _ = profile_barcode("bc_a", train, test, val, cfg)
    assert result["n_cells_barcode"] == 60
    assert result["n_cells_pool"] == 60


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _make_wtvvariantpool_structured_cfg(tmp_path, input_path) -> OmegaConf:
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
    cfg = WtvvariantpoolConfig(
        output_dir=str(tmp_path),
        input_file=str(input_path),
        label_column=_LABEL_COL,
        wt_label="WT",
        barcode_column=_BARCODE_COL,
        random_state=0,
        min_cells_per_barcode=5,
        variant_classes=["Synonymous"],
        xgboost=xgb_cfg,
    )
    return OmegaConf.structured(cfg)


def test_main_writes_feature_importance(tmp_path):
    df = _make_cells_df({"bc1": 30, "bc2": 30}, {"A1A": 30})
    input_path = tmp_path / "input.parquet"
    df.write_parquet(input_path)
    cfg = _make_wtvvariantpool_structured_cfg(tmp_path, input_path)
    with patch("fisseq_data_pipeline.wtvvariantpool.setup_logging"):
        m.main.__wrapped__(cfg)
    importance = pl.read_parquet(tmp_path / "feature_importance.parquet")
    assert set(importance.get_column("barcode").to_list()) == {"bc1", "bc2"}
    assert set(importance.columns) - {"barcode"} <= {"Intensity_Mean", "Texture_Var"}
