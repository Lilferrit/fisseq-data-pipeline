import numpy as np
import polars as pl
import pytest
import xgboost as xgb
from omegaconf import OmegaConf

from fisseq_data_pipeline.wtvwt import (
    _exclude_blocked_features,
    evaluate,
    evaluate_pair,
    filter_min_cells_per_barcode,
    profile_pair,
    select_barcodes,
    train_test_val_split,
    train_xgboost,
)

_LABEL_COL = "label"
_BARCODE_COL = "meta_barcode"


def _make_barcode_df(
    barcode_counts: dict[str, int], wt_label: str = "WT"
) -> pl.DataFrame:
    """Build a WT-only DataFrame with explicit per-barcode cell counts."""
    rng = np.random.default_rng(0)
    barcodes: list[str] = []
    for bc, n in barcode_counts.items():
        barcodes.extend([bc] * n)
    n_total = len(barcodes)
    return pl.DataFrame(
        {
            "Intensity_Mean": rng.random(n_total).tolist(),
            "Texture_Var": rng.random(n_total).tolist(),
            _LABEL_COL: [wt_label] * n_total,
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
        max_barcodes=None,
        barcode_downsample_mode="top",
        feature_block_list_file=None,
    )
    cfg.update(overrides)
    return OmegaConf.create(cfg)


# ---------------------------------------------------------------------------
# filter_min_cells_per_barcode
# ---------------------------------------------------------------------------


def test_filter_min_cells_per_barcode_removes_small_barcodes():
    df = _make_barcode_df({"bc1": 10, "bc2": 3})
    result = filter_min_cells_per_barcode(df, _BARCODE_COL, min_cells=5)
    assert "bc2" not in result.get_column(_BARCODE_COL).to_list()
    assert "bc1" in result.get_column(_BARCODE_COL).to_list()


def test_filter_min_cells_per_barcode_retains_at_threshold():
    df = _make_barcode_df({"bc1": 5, "bc2": 4})
    result = filter_min_cells_per_barcode(df, _BARCODE_COL, min_cells=5)
    assert set(result.get_column(_BARCODE_COL).to_list()) == {"bc1"}


def test_filter_min_cells_per_barcode_no_op_when_all_pass():
    df = _make_barcode_df({"bc1": 10, "bc2": 8})
    result = filter_min_cells_per_barcode(df, _BARCODE_COL, min_cells=5)
    assert len(result) == len(df)


def test_filter_min_cells_per_barcode_row_count():
    df = _make_barcode_df({"bc1": 10, "bc2": 3, "bc3": 7})
    result = filter_min_cells_per_barcode(df, _BARCODE_COL, min_cells=5)
    assert len(result) == 10 + 7


# ---------------------------------------------------------------------------
# select_barcodes
# ---------------------------------------------------------------------------


class TestSelectBarcodes:
    def test_top_mode_keeps_highest_count_barcodes(self):
        df = _make_barcode_df({"bc1": 3, "bc2": 5, "bc3": 1})
        result = select_barcodes(df, _BARCODE_COL, max_barcodes=2, mode="top", seed=0)
        assert set(result.get_column(_BARCODE_COL).to_list()) == {"bc1", "bc2"}

    def test_top_mode_tie_break_is_alphabetical(self):
        df = _make_barcode_df({"bc2": 5, "bc1": 5})
        result = select_barcodes(df, _BARCODE_COL, max_barcodes=1, mode="top", seed=0)
        assert set(result.get_column(_BARCODE_COL).to_list()) == {"bc1"}

    def test_random_mode_is_deterministic_across_calls(self):
        df = _make_barcode_df({f"bc{i}": 5 for i in range(5)})
        result1 = select_barcodes(
            df, _BARCODE_COL, max_barcodes=2, mode="random", seed=7
        )
        result2 = select_barcodes(
            df, _BARCODE_COL, max_barcodes=2, mode="random", seed=7
        )
        assert set(result1.get_column(_BARCODE_COL).to_list()) == set(
            result2.get_column(_BARCODE_COL).to_list()
        )
        assert result1.get_column(_BARCODE_COL).n_unique() == 2

    def test_random_mode_different_seed_can_change_selection(self):
        df = _make_barcode_df({f"bc{i}": 5 for i in range(5)})
        result_a = select_barcodes(
            df, _BARCODE_COL, max_barcodes=2, mode="random", seed=0
        )
        result_b = select_barcodes(
            df, _BARCODE_COL, max_barcodes=2, mode="random", seed=1
        )
        assert (
            result_a.get_column(_BARCODE_COL).n_unique()
            == result_b.get_column(_BARCODE_COL).n_unique()
            == 2
        )

    def test_invalid_mode_raises(self):
        df = _make_barcode_df({"bc1": 5})
        with pytest.raises(ValueError):
            select_barcodes(df, _BARCODE_COL, max_barcodes=1, mode="bogus", seed=0)


# ---------------------------------------------------------------------------
# _exclude_blocked_features
# ---------------------------------------------------------------------------


def _write_feature_block_list(tmp_path, feature_ok: dict[str, bool]):
    path = tmp_path / "feature_block_list.parquet"
    pl.DataFrame(
        {
            "feature": list(feature_ok.keys()),
            "feature_ok": list(feature_ok.values()),
        }
    ).write_parquet(path)
    return path


def test_exclude_blocked_features_none_is_no_op():
    assert _exclude_blocked_features(["Intensity_Mean", "Texture_Var"], None) == [
        "Intensity_Mean",
        "Texture_Var",
    ]


def test_exclude_blocked_features_drops_blocked(tmp_path):
    path = _write_feature_block_list(
        tmp_path, {"Intensity_Mean": True, "Texture_Var": False}
    )
    result = _exclude_blocked_features(["Intensity_Mean", "Texture_Var"], str(path))
    assert result == ["Intensity_Mean"]


# ---------------------------------------------------------------------------
# train_test_val_split
# ---------------------------------------------------------------------------


def test_train_test_val_split_restricts_to_wt_label():
    # label_column is dropped from the returned splits (it's not a feature
    # column and isn't needed once WT-filtering has happened), so the only
    # observable effect is that the non-WT barcode's rows are absent.
    wt = _make_barcode_df({"bc1": 30, "bc2": 30})
    variant = _make_barcode_df({"bc_v1": 30}, wt_label="V1")
    df = pl.concat([wt, variant])
    train, test, val = train_test_val_split(df, _split_cfg())
    for split in (train, test, val):
        assert _LABEL_COL not in split.columns
        assert "bc_v1" not in split.get_column(_BARCODE_COL).to_list()
    total = len(train) + len(test) + len(val)
    assert total == 60


def test_train_test_val_split_sizes():
    df = _make_barcode_df({f"bc{i}": 30 for i in range(4)})
    train, test, val = train_test_val_split(df, _split_cfg())
    total = len(df)
    assert len(train) + len(test) + len(val) == total
    assert len(train) > len(test)
    assert len(train) > len(val)


def test_train_test_val_split_no_overlap():
    df = _make_barcode_df({f"bc{i}": 30 for i in range(4)})
    train, test, val = train_test_val_split(df, _split_cfg())
    train_rows = set(map(tuple, train.select("Intensity_Mean", _BARCODE_COL).rows()))
    test_rows = set(map(tuple, test.select("Intensity_Mean", _BARCODE_COL).rows()))
    val_rows = set(map(tuple, val.select("Intensity_Mean", _BARCODE_COL).rows()))
    assert train_rows.isdisjoint(test_rows)
    assert train_rows.isdisjoint(val_rows)
    assert test_rows.isdisjoint(val_rows)


def test_train_test_val_split_every_barcode_present_in_every_split():
    df = _make_barcode_df({f"bc{i}": 30 for i in range(4)})
    train, test, val = train_test_val_split(df, _split_cfg())
    for split in (train, test, val):
        assert set(split.get_column(_BARCODE_COL).to_list()) == {
            "bc0",
            "bc1",
            "bc2",
            "bc3",
        }


def test_train_test_val_split_drops_barcodes_below_min_cells():
    df = _make_barcode_df({"bc1": 30, "bc2": 30, "bc3": 2})
    train, test, val = train_test_val_split(df, _split_cfg(min_cells_per_barcode=5))
    for split in (train, test, val):
        assert "bc3" not in split.get_column(_BARCODE_COL).to_list()


def test_train_test_val_split_max_barcodes_applies_after_min_cells():
    # bc3 has too few cells to survive min_cells_per_barcode, so it must
    # never count toward max_barcodes's budget: with min_cells_per_barcode=5
    # and max_barcodes=2, only bc1/bc2 are eligible at all, so both should
    # survive even though max_barcodes=2 would otherwise be a real cap.
    df = _make_barcode_df({"bc1": 30, "bc2": 20, "bc3": 2})
    train, test, val = train_test_val_split(
        df,
        _split_cfg(
            min_cells_per_barcode=5, max_barcodes=2, barcode_downsample_mode="top"
        ),
    )
    all_barcodes = set()
    for split in (train, test, val):
        all_barcodes |= set(split.get_column(_BARCODE_COL).to_list())
    assert all_barcodes == {"bc1", "bc2"}


def test_train_test_val_split_max_barcodes_caps_surviving_barcodes():
    df = _make_barcode_df({"bc1": 30, "bc2": 20, "bc3": 10})
    train, test, val = train_test_val_split(
        df,
        _split_cfg(
            min_cells_per_barcode=5, max_barcodes=2, barcode_downsample_mode="top"
        ),
    )
    all_barcodes = set()
    for split in (train, test, val):
        all_barcodes |= set(split.get_column(_BARCODE_COL).to_list())
    # bc1 (30) and bc2 (20) have the highest cell counts; bc3 (10) is capped out.
    assert all_barcodes == {"bc1", "bc2"}


def test_train_test_val_split_excludes_non_feature_columns():
    df = _make_barcode_df({"bc1": 20, "bc2": 20}).with_columns(
        pl.lit(0).alias("lowercase_extra")
    )
    train, test, val = train_test_val_split(df, _split_cfg())
    for split in (train, test, val):
        assert "lowercase_extra" not in split.columns
        assert set(split.columns) == {"Intensity_Mean", "Texture_Var", _BARCODE_COL}


def test_train_test_val_split_feature_block_list_file_excludes_feature(tmp_path):
    df = _make_barcode_df({"bc1": 20, "bc2": 20})
    path = _write_feature_block_list(
        tmp_path, {"Intensity_Mean": True, "Texture_Var": False}
    )
    cfg = _split_cfg(feature_block_list_file=str(path))
    train, test, val = train_test_val_split(df, cfg)
    for split in (train, test, val):
        assert "Texture_Var" not in split.columns
        assert "Intensity_Mean" in split.columns


# ---------------------------------------------------------------------------
# train_xgboost / evaluate / evaluate_pair
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


def _make_separable_pair_df(n: int = 60) -> pl.DataFrame:
    """
    Linearly separable pair: barcode A has high Intensity_Mean, B has low.

    No label column -- mirrors what train_xgboost/evaluate/evaluate_pair
    actually receive: post-train_test_val_split splits contain only feature
    columns plus barcode_column (label_column is dropped during WT
    filtering), and get_dmatrix treats every non-barcode-column as a feature.
    """
    rng = np.random.default_rng(42)
    half = n // 2
    return pl.DataFrame(
        {
            "Intensity_Mean": np.concatenate(
                [
                    rng.uniform(0.6, 1.0, half),  # bc_a
                    rng.uniform(0.0, 0.4, half),  # bc_b
                ]
            ).tolist(),
            "Texture_Var": rng.random(n).tolist(),
            _BARCODE_COL: ["bc_a"] * half + ["bc_b"] * half,
        }
    )


@pytest.fixture()
def trained_model_and_splits():
    df = _make_separable_pair_df(n=60)
    cfg = _make_xgb_cfg()
    a = df.filter(pl.col(_BARCODE_COL) == "bc_a")
    b = df.filter(pl.col(_BARCODE_COL) == "bc_b")
    train = pl.concat([a[:20], b[:20]])
    val = pl.concat([a[20:25], b[20:25]])
    test = pl.concat([a[25:], b[25:]])
    model = train_xgboost(train, val, _BARCODE_COL, "bc_a", cfg)
    return model, train, val, test, cfg


def test_train_xgboost_returns_booster():
    df = _make_separable_pair_df(n=60)
    cfg = _make_xgb_cfg()
    half = len(df) // 2
    model = train_xgboost(df[:half], df[half:], _BARCODE_COL, "bc_a", cfg)
    assert isinstance(model, xgb.Booster)


def test_train_xgboost_without_sample_weights():
    df = _make_separable_pair_df(n=60)
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


def test_evaluate_pair_result_keys(trained_model_and_splits):
    model, train, val, test, _ = trained_model_and_splits
    result = evaluate_pair(model, train, val, test, _BARCODE_COL, "bc_a", "bc_b")
    expected_keys = {
        "barcode_a",
        "barcode_b",
        "train_auroc",
        "train_accuracy",
        "val_auroc",
        "val_accuracy",
        "test_auroc",
        "test_accuracy",
        "n_cells_a",
        "n_cells_b",
    }
    assert set(result.keys()) == expected_keys


def test_evaluate_pair_barcode_names(trained_model_and_splits):
    model, train, val, test, _ = trained_model_and_splits
    result = evaluate_pair(model, train, val, test, _BARCODE_COL, "bc_a", "bc_b")
    assert result["barcode_a"] == "bc_a"
    assert result["barcode_b"] == "bc_b"


def test_evaluate_pair_n_cells(trained_model_and_splits):
    model, train, val, test, _ = trained_model_and_splits
    result = evaluate_pair(model, train, val, test, _BARCODE_COL, "bc_a", "bc_b")
    assert result["n_cells_a"] == 30
    assert result["n_cells_b"] == 30


# ---------------------------------------------------------------------------
# profile_pair
# ---------------------------------------------------------------------------


def _make_multi_barcode_split(n_each: int = 30):
    """
    Build train/test/val splits containing three WT barcodes.

    bc_a is linearly separable from bc_b (high vs low Intensity_Mean).
    bc_c shares bc_b's distribution (not separable from it).
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
            _block(0.0, 0.4, "bc_c", n_each),
        ]
    )
    val = _split(
        [
            _block(0.6, 1.0, "bc_a", half),
            _block(0.0, 0.4, "bc_b", half),
            _block(0.0, 0.4, "bc_c", half),
        ]
    )
    test = _split(
        [
            _block(0.6, 1.0, "bc_a", half),
            _block(0.0, 0.4, "bc_b", half),
            _block(0.0, 0.4, "bc_c", half),
        ]
    )
    return train, val, test


@pytest.fixture()
def profile_pair_splits():
    train, val, test = _make_multi_barcode_split()
    cfg = _make_xgb_cfg()
    cfg.barcode_column = _BARCODE_COL
    return train, val, test, cfg


def test_profile_pair_returns_booster(profile_pair_splits):
    train, val, test, cfg = profile_pair_splits
    result, model = profile_pair("bc_a", "bc_b", train, test, val, cfg)
    assert isinstance(model, xgb.Booster)


def test_profile_pair_high_auroc_on_separable_barcodes(profile_pair_splits):
    train, val, test, cfg = profile_pair_splits
    result, _ = profile_pair("bc_a", "bc_b", train, test, val, cfg)
    assert result["train_auroc"] > 0.9


def test_profile_pair_low_auroc_on_indistinguishable_barcodes(profile_pair_splits):
    """bc_b and bc_c share the same distribution, so AUROC should be near chance."""
    train, val, test, cfg = profile_pair_splits
    result, _ = profile_pair("bc_b", "bc_c", train, test, val, cfg)
    assert result["train_auroc"] < 0.8


def test_profile_pair_filters_to_target_pair_only(profile_pair_splits):
    # n_cells_a/b count rows across train+val+test (30 + 15 + 15 = 60, per
    # _make_multi_barcode_split's n_each=30/half=15).
    train, val, test, cfg = profile_pair_splits
    result, _ = profile_pair("bc_a", "bc_b", train, test, val, cfg)
    assert result["n_cells_a"] == 60
    assert result["n_cells_b"] == 60
