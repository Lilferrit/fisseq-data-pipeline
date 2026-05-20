import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from fisseq_data_pipeline.ovwt import (
    convert_labels_to_boolean,
    downsample_wildtype,
    filter_min_cells,
    get_dmatrix,
    get_feature_cols,
    profile_variant,
    read_feature_file,
)
from fisseq_data_pipeline.ovwt import test_xgboost as evaluate_splits
from fisseq_data_pipeline.ovwt import train_test_val_split, train_xgboost


def _make_df(
    n: int = 20,
    label_column: str = "label",
    wt_label: str = "WT",
    variant_label: str = "V1",
) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    labels = [wt_label] * (n // 2) + [variant_label] * (n // 2)
    return pl.DataFrame(
        {
            "Intensity_Mean": rng.random(n).tolist(),
            "Texture_Var": rng.random(n).tolist(),
            label_column: labels,
        }
    )


def _split_cfg(label_column: str = "label") -> OmegaConf:
    return OmegaConf.create(
        {
            "label_column": label_column,
            "wt_label": "WT",
            "random_state": 0,
            "feature_cols": None,
            "min_cells": None,
            "downsample_wt": False,
        }
    )


# ---------------------------------------------------------------------------
# get_feature_cols
# ---------------------------------------------------------------------------


def test_get_feature_cols_returns_cellprofiler_columns():
    df = pl.DataFrame({"Intensity_Mean": [1.0], "Texture_Var": [2.0], "label": ["WT"]})
    assert get_feature_cols(df) == ["Intensity_Mean", "Texture_Var"]


def test_get_feature_cols_excludes_lowercase_columns():
    df = pl.DataFrame({"Intensity_Mean": [1.0], "metadata": ["foo"]})
    assert get_feature_cols(df) == ["Intensity_Mean"]


def test_get_feature_cols_excludes_uppercase_without_underscore():
    df = pl.DataFrame({"Intensity_Mean": [1.0], "Intensity": [2.0]})
    assert get_feature_cols(df) == ["Intensity_Mean"]


def test_get_feature_cols_empty_dataframe():
    df = pl.DataFrame({"label": []})
    assert get_feature_cols(df) == []


def test_get_feature_cols_no_matching_columns():
    df = pl.DataFrame({"label": ["WT"], "metadata": ["foo"]})
    assert get_feature_cols(df) == []


def test_get_feature_cols_all_columns_match():
    df = pl.DataFrame({"Intensity_Mean": [1.0], "Texture_Var": [2.0]})
    assert set(get_feature_cols(df)) == {"Intensity_Mean", "Texture_Var"}


# ---------------------------------------------------------------------------
# convert_labels_to_boolean
# ---------------------------------------------------------------------------


def test_convert_labels_to_boolean_wt_maps_to_true():
    labels = np.array(["WT", "V1", "WT", "V1"])
    result = convert_labels_to_boolean(labels, "WT")
    np.testing.assert_array_equal(result, [True, False, True, False])


def test_convert_labels_to_boolean_all_wt():
    labels = np.array(["WT", "WT", "WT"])
    assert convert_labels_to_boolean(labels, "WT").all()


def test_convert_labels_to_boolean_none_wt():
    labels = np.array(["V1", "V2", "V3"])
    assert not convert_labels_to_boolean(labels, "WT").any()


# ---------------------------------------------------------------------------
# get_dmatrix
# ---------------------------------------------------------------------------


def test_get_dmatrix_label_values():
    df = _make_df(n=10)
    dm = get_dmatrix(df, "label", "WT")
    assert set(dm.get_label()) == {0.0, 1.0}


def test_get_dmatrix_wt_label_is_true():
    df = _make_df(n=10)
    dm = get_dmatrix(df, "label", "WT")
    # 5 WT rows → 5 True (1.0) labels
    assert dm.get_label().sum() == 5.0


def test_get_dmatrix_shape():
    df = _make_df(n=20)
    dm = get_dmatrix(df, "label", "WT")
    assert dm.num_row() == 20
    assert dm.num_col() == 2  # Intensity_Mean, Texture_Var


def test_get_dmatrix_with_weights():
    df = _make_df(n=10)
    weights = np.full(10, 2.0)
    dm = get_dmatrix(df, "label", "WT", weight=weights)
    np.testing.assert_array_equal(dm.get_weight(), weights)


def test_get_dmatrix_no_weights_by_default():
    df = _make_df(n=10)
    dm = get_dmatrix(df, "label", "WT")
    assert len(dm.get_weight()) == 0


def test_get_dmatrix_inf_replaced_with_nan():
    df = _make_df(n=10)
    df = df.with_columns(pl.lit(float("inf")).alias("Inf_Feature"))
    # Should not raise; inf values are coerced to nan (missing)
    dm = get_dmatrix(df, "label", "WT")
    assert dm.num_row() == 10


def test_get_dmatrix_neg_inf_replaced_with_nan():
    df = _make_df(n=10)
    df = df.with_columns(pl.lit(float("-inf")).alias("NegInf_Feature"))
    dm = get_dmatrix(df, "label", "WT")
    assert dm.num_row() == 10


def test_get_dmatrix_mixed_inf_and_valid_values():
    rng = np.random.default_rng(0)
    values = rng.random(10).tolist()
    values[0] = float("inf")
    values[5] = float("-inf")
    df = pl.DataFrame({"Intensity_Mean": values, "label": ["WT"] * 5 + ["V1"] * 5})
    dm = get_dmatrix(df, "label", "WT")
    assert dm.num_row() == 10


# ---------------------------------------------------------------------------
# read_feature_file
# ---------------------------------------------------------------------------


def test_read_feature_file_parquet(tmp_path):
    df = pl.DataFrame({"Intensity_Mean": [1.0, 2.0], "label": ["WT", "V1"]})
    path = tmp_path / "data.parquet"
    df.write_parquet(path)
    assert read_feature_file(path).equals(df)


def test_read_feature_file_pq_extension(tmp_path):
    df = pl.DataFrame({"Intensity_Mean": [1.0, 2.0], "label": ["WT", "V1"]})
    path = tmp_path / "data.pq"
    df.write_parquet(path)
    assert read_feature_file(path).equals(df)


def test_read_feature_file_csv(tmp_path):
    df = pl.DataFrame({"Intensity_Mean": [1.0, 2.0], "label": ["WT", "V1"]})
    path = tmp_path / "data.csv"
    df.write_csv(path)
    assert read_feature_file(path).equals(df)


def test_read_feature_file_unsupported_extension(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported file format"):
        read_feature_file(path)


# ---------------------------------------------------------------------------
# downsample_wildtype
# ---------------------------------------------------------------------------


def _make_multilabel_df(wt_count: int, variant_counts: dict[str, int]) -> pl.DataFrame:
    """Build a DataFrame with WT rows and multiple named variants."""
    rng = np.random.default_rng(0)
    rows: dict[str, list] = {"Intensity_Mean": [], "label": []}
    for _ in range(wt_count):
        rows["Intensity_Mean"].append(rng.random())
        rows["label"].append("WT")
    for label, n in variant_counts.items():
        for _ in range(n):
            rows["Intensity_Mean"].append(rng.random())
            rows["label"].append(label)
    return pl.DataFrame(rows)


def test_downsample_wildtype_reduces_wt_to_max_variant():
    df = _make_multilabel_df(wt_count=100, variant_counts={"V1": 30, "V2": 20})
    result = downsample_wildtype(df, "label", "WT", seed=0)
    wt_count = (result.get_column("label") == "WT").sum()
    assert wt_count == 30


def test_downsample_wildtype_preserves_all_variant_rows():
    df = _make_multilabel_df(wt_count=100, variant_counts={"V1": 30, "V2": 20})
    result = downsample_wildtype(df, "label", "WT", seed=0)
    assert (result.get_column("label") == "V1").sum() == 30
    assert (result.get_column("label") == "V2").sum() == 20


def test_downsample_wildtype_no_op_when_wt_already_smaller():
    df = _make_multilabel_df(wt_count=10, variant_counts={"V1": 30})
    result = downsample_wildtype(df, "label", "WT", seed=0)
    assert (result.get_column("label") == "WT").sum() == 10


def test_downsample_wildtype_no_op_when_wt_equals_max_variant():
    df = _make_multilabel_df(wt_count=30, variant_counts={"V1": 30})
    result = downsample_wildtype(df, "label", "WT", seed=0)
    assert (result.get_column("label") == "WT").sum() == 30


def test_downsample_wildtype_reproducible_with_same_seed():
    df = _make_multilabel_df(wt_count=100, variant_counts={"V1": 40})
    r1 = downsample_wildtype(df, "label", "WT", seed=7)
    r2 = downsample_wildtype(df, "label", "WT", seed=7)
    wt1 = r1.filter(pl.col("label") == "WT").sort("Intensity_Mean")
    wt2 = r2.filter(pl.col("label") == "WT").sort("Intensity_Mean")
    assert wt1.equals(wt2)


def test_downsample_wildtype_total_row_count():
    df = _make_multilabel_df(wt_count=100, variant_counts={"V1": 30, "V2": 20})
    result = downsample_wildtype(df, "label", "WT", seed=0)
    assert len(result) == 30 + 30 + 20


# ---------------------------------------------------------------------------
# filter_min_cells
# ---------------------------------------------------------------------------


def test_filter_min_cells_removes_variants_below_threshold():
    df = _make_multilabel_df(wt_count=50, variant_counts={"V1": 10, "V2": 3})
    result = filter_min_cells(df, "label", "WT", min_cells=5)
    assert "V2" not in result.get_column("label").to_list()
    assert "V1" in result.get_column("label").to_list()


def test_filter_min_cells_retains_variants_at_threshold():
    df = _make_multilabel_df(wt_count=50, variant_counts={"V1": 5, "V2": 4})
    result = filter_min_cells(df, "label", "WT", min_cells=5)
    assert "V1" in result.get_column("label").to_list()
    assert "V2" not in result.get_column("label").to_list()


def test_filter_min_cells_always_retains_wt():
    df = _make_multilabel_df(wt_count=3, variant_counts={"V1": 10})
    result = filter_min_cells(df, "label", "WT", min_cells=5)
    assert (result.get_column("label") == "WT").sum() == 3


def test_filter_min_cells_no_op_when_all_pass():
    df = _make_multilabel_df(wt_count=20, variant_counts={"V1": 10, "V2": 8})
    result = filter_min_cells(df, "label", "WT", min_cells=5)
    assert len(result) == len(df)


def test_filter_min_cells_removes_all_variants_when_none_pass():
    df = _make_multilabel_df(wt_count=20, variant_counts={"V1": 2, "V2": 3})
    result = filter_min_cells(df, "label", "WT", min_cells=10)
    assert set(result.get_column("label").to_list()) == {"WT"}


def test_filter_min_cells_row_count():
    df = _make_multilabel_df(wt_count=20, variant_counts={"V1": 10, "V2": 3})
    result = filter_min_cells(df, "label", "WT", min_cells=5)
    assert len(result) == 20 + 10


# ---------------------------------------------------------------------------
# train_test_val_split
# ---------------------------------------------------------------------------


def test_train_test_val_split_sizes():
    # 100 rows → 80 train, 10 test, 10 val
    df = _make_df(n=100)
    train, test, val = train_test_val_split(df, _split_cfg())
    assert len(train) == 80
    assert len(test) == 10
    assert len(val) == 10


def test_train_test_val_split_no_overlap():
    df = _make_df(n=100)
    # Add a unique id column to track rows across splits
    df = df.with_row_index("__row__")
    train, test, val = train_test_val_split(df, _split_cfg())
    # __row__ is lowercase so not selected as a feature; total rows should sum to 100
    assert len(train) + len(test) + len(val) == 100


def test_train_test_val_split_excludes_non_feature_columns():
    df = _make_df(n=100).with_columns(pl.lit(0).alias("lowercase_extra"))
    train, test, val = train_test_val_split(df, _split_cfg())
    for split in (train, test, val):
        assert "lowercase_extra" not in split.columns
        assert set(split.columns) == {"Intensity_Mean", "Texture_Var", "label"}


def test_train_test_val_split_preserves_class_ratio():
    df = _make_df(n=100)
    train, test, val = train_test_val_split(df, _split_cfg())
    for split in (train, test, val):
        counts = split.get_column("label").value_counts()
        n_wt = counts.filter(pl.col("label") == "WT")["count"][0]
        n_v1 = counts.filter(pl.col("label") == "V1")["count"][0]
        assert n_wt == n_v1


# ---------------------------------------------------------------------------
# train_xgboost / test_xgboost
# ---------------------------------------------------------------------------


def _make_xgb_cfg(weigh_samples: bool = True) -> OmegaConf:
    return OmegaConf.create(
        {
            "label_column": "label",
            "wt_label": "WT",
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


def _make_separable_df(n: int = 60) -> pl.DataFrame:
    """Linearly separable dataset: WT has high Intensity_Mean, V1 has low."""
    rng = np.random.default_rng(42)
    half = n // 2
    return pl.DataFrame(
        {
            "Intensity_Mean": np.concatenate(
                [
                    rng.uniform(0.6, 1.0, half),  # WT
                    rng.uniform(0.0, 0.4, half),  # V1
                ]
            ).tolist(),
            "Texture_Var": rng.random(n).tolist(),
            "label": ["WT"] * half + ["V1"] * half,
        }
    )


@pytest.fixture()
def trained_model_and_splits():
    df = _make_separable_df(n=60)
    cfg = _make_xgb_cfg()
    wt = df.filter(pl.col("label") == "WT")
    v1 = df.filter(pl.col("label") == "V1")
    # 20 + 20 train, 5 + 5 val, 5 + 5 test
    train = pl.concat([wt[:20], v1[:20]])
    val = pl.concat([wt[20:25], v1[20:25]])
    test = pl.concat([wt[25:], v1[25:]])
    model = train_xgboost(train, val, cfg)
    return model, train, val, test, cfg


def test_train_xgboost_returns_booster():
    import xgboost as xgb

    df = _make_separable_df(n=60)
    cfg = _make_xgb_cfg()
    half = len(df) // 2
    model = train_xgboost(df[:half], df[half:], cfg)
    assert isinstance(model, xgb.Booster)


def test_train_xgboost_without_sample_weights():
    import xgboost as xgb

    df = _make_separable_df(n=60)
    cfg = _make_xgb_cfg(weigh_samples=False)
    half = len(df) // 2
    model = train_xgboost(df[:half], df[half:], cfg)
    assert isinstance(model, xgb.Booster)


def test_test_xgboost_result_keys(trained_model_and_splits):
    model, train, val, test, cfg = trained_model_and_splits
    result = evaluate_splits(model, train, val, test, cfg)
    expected_keys = {
        "variant",
        "train_auroc",
        "train_accuracy",
        "val_auroc",
        "val_accuracy",
        "test_auroc",
        "test_accuracy",
    }
    assert set(result.keys()) == expected_keys


def test_test_xgboost_variant_name(trained_model_and_splits):
    model, train, val, test, cfg = trained_model_and_splits
    result = evaluate_splits(model, train, val, test, cfg)
    assert result["variant"] == "V1"


def test_test_xgboost_auroc_in_range(trained_model_and_splits):
    model, train, val, test, cfg = trained_model_and_splits
    result = evaluate_splits(model, train, val, test, cfg)
    for key in ("train_auroc", "val_auroc", "test_auroc"):
        assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"


def test_test_xgboost_accuracy_in_range(trained_model_and_splits):
    model, train, val, test, cfg = trained_model_and_splits
    result = evaluate_splits(model, train, val, test, cfg)
    for key in ("train_accuracy", "val_accuracy", "test_accuracy"):
        assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"


def test_test_xgboost_separable_data_high_auroc(trained_model_and_splits):
    model, train, val, test, cfg = trained_model_and_splits
    result = evaluate_splits(model, train, val, test, cfg)
    assert result["train_auroc"] > 0.9


# ---------------------------------------------------------------------------
# profile_variant
# ---------------------------------------------------------------------------


def _make_multisplit(
    variant: str = "V1",
    other_variant: str = "V2",
    n_each: int = 30,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Build train/test/val splits containing two variants plus WT.

    V1 is linearly separable from WT (high Intensity_Mean vs low).
    V2 has the same distribution as WT (not separable).
    """
    rng = np.random.default_rng(42)
    half = n_each // 2

    def _block(low: float, high: float, label: str, n: int) -> dict:
        return {
            "Intensity_Mean": rng.uniform(low, high, n).tolist(),
            "label": [label] * n,
        }

    def _split(rows: list[dict]) -> pl.DataFrame:
        return pl.concat([pl.DataFrame(r) for r in rows])

    wt_train = _block(0.6, 1.0, "WT", n_each)
    v1_train = _block(0.0, 0.4, variant, n_each)
    v2_train = _block(0.6, 1.0, other_variant, n_each)

    wt_val = _block(0.6, 1.0, "WT", half)
    v1_val = _block(0.0, 0.4, variant, half)
    v2_val = _block(0.6, 1.0, other_variant, half)

    wt_test = _block(0.6, 1.0, "WT", half)
    v1_test = _block(0.0, 0.4, variant, half)
    v2_test = _block(0.6, 1.0, other_variant, half)

    train = _split([wt_train, v1_train, v2_train])
    val = _split([wt_val, v1_val, v2_val])
    test = _split([wt_test, v1_test, v2_test])
    return train, val, test


@pytest.fixture()
def profile_variant_splits():
    train, val, test = _make_multisplit()
    cfg = _make_xgb_cfg()
    return train, val, test, cfg


def test_profile_variant_returns_booster(profile_variant_splits):
    import xgboost as xgb

    train, val, test, cfg = profile_variant_splits
    result, model = profile_variant("V1", train, test, val, cfg)
    assert isinstance(model, xgb.Booster)


def test_profile_variant_returns_result_dict(profile_variant_splits):
    train, val, test, cfg = profile_variant_splits
    result, _ = profile_variant("V1", train, test, val, cfg)
    assert isinstance(result, dict)


def test_profile_variant_result_keys(profile_variant_splits):
    train, val, test, cfg = profile_variant_splits
    result, _ = profile_variant("V1", train, test, val, cfg)
    expected_keys = {
        "variant",
        "train_auroc",
        "train_accuracy",
        "val_auroc",
        "val_accuracy",
        "test_auroc",
        "test_accuracy",
    }
    assert set(result.keys()) == expected_keys


def test_profile_variant_result_variant_name(profile_variant_splits):
    train, val, test, cfg = profile_variant_splits
    result, _ = profile_variant("V1", train, test, val, cfg)
    assert result["variant"] == "V1"


def test_profile_variant_auroc_in_range(profile_variant_splits):
    train, val, test, cfg = profile_variant_splits
    result, _ = profile_variant("V1", train, test, val, cfg)
    for key in ("train_auroc", "val_auroc", "test_auroc"):
        assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"


def test_profile_variant_high_auroc_on_separable_data(profile_variant_splits):
    train, val, test, cfg = profile_variant_splits
    result, _ = profile_variant("V1", train, test, val, cfg)
    assert result["train_auroc"] > 0.9


def test_profile_variant_filters_to_target_variant_only(profile_variant_splits):
    """V2 has the same distribution as WT, so it should be near-chance AUROC."""
    train, val, test, cfg = profile_variant_splits
    result, _ = profile_variant("V2", train, test, val, cfg)
    # V2 is indistinguishable from WT, so the model should not achieve high AUROC
    assert result["train_auroc"] < 0.75
