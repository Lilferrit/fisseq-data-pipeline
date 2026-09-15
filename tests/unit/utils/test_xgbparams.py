import dataclasses
import logging

import numpy as np
import polars as pl
import pytest
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf

from fisseq_data_pipeline.utils.xgbparams import (
    XGBoostConfig,
    XGBoostParams,
    get_dmatrix,
    get_feature_cols,
    split_indices_stratified,
    train_binary_xgboost,
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
# split_indices_stratified
# ---------------------------------------------------------------------------


def test_split_indices_stratified_sizes():
    labels = np.array(["A"] * 50 + ["B"] * 50)
    train_idx, holdout_idx = split_indices_stratified(labels, random_state=0)
    assert len(train_idx) == 80
    assert len(holdout_idx) == 20


def test_split_indices_stratified_no_overlap():
    labels = np.array(["A"] * 50 + ["B"] * 50)
    train_idx, holdout_idx = split_indices_stratified(labels, random_state=0)
    assert set(train_idx).isdisjoint(set(holdout_idx))


def test_split_indices_stratified_union_is_all():
    n = 100
    labels = np.array(["A"] * 50 + ["B"] * 50)
    train_idx, holdout_idx = split_indices_stratified(labels, random_state=0)
    assert set(train_idx) | set(holdout_idx) == set(range(n))


def test_split_indices_stratified_preserves_class_ratio():
    labels = np.array(["A"] * 50 + ["B"] * 50)
    train_idx, holdout_idx = split_indices_stratified(labels, random_state=0)
    for idx in (train_idx, holdout_idx):
        split_labels = labels[idx]
        assert (split_labels == "A").sum() == (split_labels == "B").sum()


def test_split_indices_stratified_respects_test_size():
    labels = np.array(["A"] * 50 + ["B"] * 50)
    train_idx, holdout_idx = split_indices_stratified(
        labels, random_state=0, test_size=0.5
    )
    assert len(train_idx) == 50
    assert len(holdout_idx) == 50


def test_split_indices_stratified_singleton_stratum_does_not_raise():
    """The production failure: a lone member of its stratum used to raise."""
    labels = np.array(["A"] * 50 + ["B"] * 49 + ["solo"])
    train_idx, holdout_idx = split_indices_stratified(labels, random_state=0)
    assert len(train_idx) + len(holdout_idx) == len(labels)


def test_split_indices_stratified_singleton_goes_to_train():
    labels = np.array(["A"] * 50 + ["B"] * 49 + ["solo"])
    solo = len(labels) - 1
    train_idx, holdout_idx = split_indices_stratified(labels, random_state=0)
    assert solo in set(train_idx)
    assert solo not in set(holdout_idx)


def test_split_indices_stratified_discards_no_rows_with_singletons():
    labels = np.array(["A"] * 40 + ["B"] * 40 + ["solo1", "solo2", "solo3"])
    train_idx, holdout_idx = split_indices_stratified(labels, random_state=0)
    assert set(train_idx) | set(holdout_idx) == set(range(len(labels)))
    assert set(train_idx).isdisjoint(set(holdout_idx))


def test_split_indices_stratified_warns_on_singleton_stratum(caplog):
    labels = np.array(["A"] * 40 + ["B"] * 40 + ["solo1", "solo2"])
    with caplog.at_level(logging.WARNING):
        split_indices_stratified(labels, random_state=0)
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "2 row(s)" in message
    assert "2 singleton" in message


def test_split_indices_stratified_no_warning_when_all_strata_populated(caplog):
    labels = np.array(["A"] * 50 + ["B"] * 50)
    with caplog.at_level(logging.WARNING):
        split_indices_stratified(labels, random_state=0)
    assert caplog.records == []


def test_split_indices_stratified_two_member_stratum_survives():
    """A 2-member stratum splitting 1/1 broke the old nested second split."""
    labels = np.array(["A"] * 49 + ["B"] * 49 + ["pair", "pair"])
    train_idx, holdout_idx = split_indices_stratified(labels, random_state=0)
    assert set(train_idx) | set(holdout_idx) == set(range(len(labels)))


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


# ---------------------------------------------------------------------------
# train_binary_xgboost
# ---------------------------------------------------------------------------


def _train_cfg(**overrides) -> DictConfig:
    """A DictConfig shaped like OvwtConfig, which is what the real caller passes."""
    base = {
        "label_column": "label",
        "wt_label": "WT",
        "random_seed": 0,
        "xgboost": OmegaConf.structured(XGBoostConfig()),
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _separable_df(n: int = 60) -> pl.DataFrame:
    """WT and V1 cleanly separated on Intensity_Mean, noise on Texture_Var."""
    rng = np.random.default_rng(0)
    half = n // 2
    return pl.DataFrame(
        {
            "Intensity_Mean": np.concatenate(
                [rng.normal(0.0, 0.1, half), rng.normal(5.0, 0.1, half)]
            ).tolist(),
            "Texture_Var": rng.random(n).tolist(),
            "label": ["WT"] * half + ["V1"] * half,
        }
    )


def test_train_binary_xgboost_returns_booster():
    df = _separable_df()
    model = train_binary_xgboost(df, df, _train_cfg())
    assert isinstance(model, xgb.Booster)


def test_train_binary_xgboost_learns_separable_signal():
    df = _separable_df()
    model = train_binary_xgboost(df, df, _train_cfg())
    scores = model.predict(get_dmatrix(df, "label", "WT"))
    is_wt = df.get_column("label").to_numpy() == "WT"
    # Predicts P(wildtype), so WT rows must score above variant rows.
    assert scores[is_wt].mean() > scores[~is_wt].mean()


def test_train_binary_xgboost_is_seeded_by_random_seed():
    df = _separable_df()
    a = train_binary_xgboost(df, df, _train_cfg(random_seed=7))
    b = train_binary_xgboost(df, df, _train_cfg(random_seed=7))
    dm = get_dmatrix(df, "label", "WT")
    np.testing.assert_allclose(a.predict(dm), b.predict(dm))


def test_train_binary_xgboost_rejects_plain_dataclass_config():
    """dict(cfg.xgboost.params) needs a DictConfig, not a bare dataclass."""
    df = _separable_df()
    with pytest.raises(Exception):
        train_binary_xgboost(df, df, dataclasses.make_dataclass("C", [])())
