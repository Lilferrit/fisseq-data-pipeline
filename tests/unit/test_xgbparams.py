import numpy as np
import polars as pl
import pytest
import xgboost as xgb

from fisseq_data_pipeline.xgbparams import (
    XGBoostConfig,
    XGBoostParams,
    get_dmatrix,
    get_dmatrix_multiclass,
    get_feature_cols,
    split_indices_stratified,
)


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


def _make_multiclass_df(
    n_per_class: int = 20, classes: list[str] | None = None
) -> pl.DataFrame:
    if classes is None:
        classes = ["batch_a", "batch_b", "batch_c"]
    rng = np.random.default_rng(1)
    n = n_per_class * len(classes)
    labels = []
    for c in classes:
        labels.extend([c] * n_per_class)
    return pl.DataFrame(
        {
            "Intensity_Mean": rng.random(n).tolist(),
            "Texture_Var": rng.random(n).tolist(),
            "batch": labels,
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


# ---------------------------------------------------------------------------
# get_dmatrix (binary)
# ---------------------------------------------------------------------------


def test_get_dmatrix_label_values():
    df = _make_df(n=10)
    dm = get_dmatrix(df, "label", "WT")
    assert set(dm.get_label()) == {0.0, 1.0}


def test_get_dmatrix_wt_label_is_true():
    df = _make_df(n=10)
    dm = get_dmatrix(df, "label", "WT")
    assert dm.get_label().sum() == 5.0


def test_get_dmatrix_shape():
    df = _make_df(n=20)
    dm = get_dmatrix(df, "label", "WT")
    assert dm.num_row() == 20
    assert dm.num_col() == 2


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
    df = _make_df(n=10).with_columns(pl.lit(float("inf")).alias("Inf_Feature"))
    dm = get_dmatrix(df, "label", "WT")
    assert dm.num_row() == 10


def test_get_dmatrix_neg_inf_replaced_with_nan():
    df = _make_df(n=10).with_columns(pl.lit(float("-inf")).alias("NegInf_Feature"))
    dm = get_dmatrix(df, "label", "WT")
    assert dm.num_row() == 10


# ---------------------------------------------------------------------------
# get_dmatrix_multiclass
# ---------------------------------------------------------------------------


def test_get_dmatrix_multiclass_shape():
    df = _make_multiclass_df(n_per_class=20, classes=["a", "b", "c"])
    feature_cols = ["Intensity_Mean", "Texture_Var"]
    dm, classes = get_dmatrix_multiclass(df, feature_cols, "batch")
    assert dm.num_row() == 60
    assert dm.num_col() == 2


def test_get_dmatrix_multiclass_classes_sorted():
    df = _make_multiclass_df(classes=["gamma", "alpha", "beta"])
    feature_cols = ["Intensity_Mean", "Texture_Var"]
    _, classes = get_dmatrix_multiclass(df, feature_cols, "batch")
    assert classes == sorted(classes)


def test_get_dmatrix_multiclass_labels_are_integers_in_range():
    df = _make_multiclass_df(n_per_class=10, classes=["a", "b", "c"])
    feature_cols = ["Intensity_Mean", "Texture_Var"]
    dm, classes = get_dmatrix_multiclass(df, feature_cols, "batch")
    labels = dm.get_label().astype(int)
    assert set(labels) == set(range(len(classes)))


def test_get_dmatrix_multiclass_inf_replaced_with_nan():
    df = _make_multiclass_df(n_per_class=10).with_columns(
        pl.lit(float("inf")).alias("Inf_Feature")
    )
    feature_cols = ["Intensity_Mean", "Texture_Var", "Inf_Feature"]
    dm, _ = get_dmatrix_multiclass(df, feature_cols, "batch")
    assert dm.num_row() == len(df)


# ---------------------------------------------------------------------------
# split_indices_stratified
# ---------------------------------------------------------------------------


def test_split_indices_stratified_sizes():
    labels = np.array(["A"] * 50 + ["B"] * 50)
    train_idx, test_idx, val_idx = split_indices_stratified(labels, random_state=0)
    assert len(train_idx) == 80
    assert len(test_idx) == 10
    assert len(val_idx) == 10


def test_split_indices_stratified_no_overlap():
    labels = np.array(["A"] * 50 + ["B"] * 50)
    train_idx, test_idx, val_idx = split_indices_stratified(labels, random_state=0)
    assert set(train_idx).isdisjoint(set(test_idx))
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(test_idx).isdisjoint(set(val_idx))


def test_split_indices_stratified_union_is_all():
    n = 100
    labels = np.array(["A"] * 50 + ["B"] * 50)
    train_idx, test_idx, val_idx = split_indices_stratified(labels, random_state=0)
    all_idx = set(train_idx) | set(test_idx) | set(val_idx)
    assert all_idx == set(range(n))


def test_split_indices_stratified_preserves_class_ratio():
    labels = np.array(["A"] * 50 + ["B"] * 50)
    train_idx, test_idx, val_idx = split_indices_stratified(labels, random_state=0)
    for idx in (train_idx, test_idx, val_idx):
        split_labels = labels[idx]
        n_a = (split_labels == "A").sum()
        n_b = (split_labels == "B").sum()
        assert n_a == n_b


# ---------------------------------------------------------------------------
# XGBoostParams / XGBoostConfig dataclasses
# ---------------------------------------------------------------------------


def test_xgboost_params_defaults():
    p = XGBoostParams()
    assert p.nthread == -1
    assert p.max_depth == 3
    assert p.subsample == 0.5


def test_xgboost_config_defaults():
    c = XGBoostConfig()
    assert c.num_boost_round == 100
    assert c.early_stopping_rounds == 5
    assert c.weigh_samples is True
    assert isinstance(c.params, XGBoostParams)
