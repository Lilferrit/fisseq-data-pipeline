from __future__ import annotations

import pathlib
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
import xgboost as xgb
from omegaconf import OmegaConf

import fisseq_data_pipeline.batchvsbatch as m
from fisseq_data_pipeline.batchvsbatch import BvbConfig
from fisseq_data_pipeline.utils.xgbparams import XGBoostConfig, XGBoostParams

_LABEL_COL = "label"
_BATCH_COL = "meta_batch"
_FEATURE_COLS = ["Intensity_Mean", "Texture_Var"]


def _make_batch_df(
    n_batches: int = 3,
    n_cells_per_batch: int = 40,
    n_variants: int = 2,
) -> pl.DataFrame:
    """
    Synthetic DataFrame with two feature columns, a variant label, and a batch label.

    batch_0 has high Intensity_Mean (separable from others), remaining batches
    share the same low range so the model can distinguish at least one batch.
    """
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    for v_idx in range(n_variants):
        variant = f"V{v_idx}"
        for b_idx in range(n_batches):
            batch = f"batch_{b_idx}"
            low, high = (0.7, 1.0) if b_idx == 0 else (0.0, 0.4)
            intensity = rng.uniform(low, high, n_cells_per_batch).tolist()
            texture = rng.random(n_cells_per_batch).tolist()
            for i in range(n_cells_per_batch):
                rows.append(
                    {
                        "Intensity_Mean": intensity[i],
                        "Texture_Var": texture[i],
                        _LABEL_COL: variant,
                        _BATCH_COL: batch,
                    }
                )
    return pl.DataFrame(rows)


def _make_cfg(**overrides) -> OmegaConf:
    cfg = dict(
        output_dir="/tmp/bvb_test",
        input_file="/tmp/bvb_test/input.parquet",
        label_column=_LABEL_COL,
        batch_column=_BATCH_COL,
        random_state=0,
        feature_cols=None,
        min_cells=10,
        min_batches=2,
        xgboost=dict(
            num_boost_round=5,
            early_stopping_rounds=3,
            weigh_samples=False,
            params=dict(
                nthread=1,
                max_depth=2,
                colsample_bytree=1.0,
                colsample_bylevel=1.0,
                colsample_bynode=1.0,
                subsample=1.0,
            ),
        ),
    )
    cfg.update(overrides)
    return OmegaConf.create(cfg)


# ---------------------------------------------------------------------------
# train_test_val_split
# ---------------------------------------------------------------------------


def test_train_test_val_split_no_overlap():
    df = _make_batch_df(n_batches=3, n_cells_per_batch=30, n_variants=2)
    cfg = _make_cfg()
    train, test, val = m.train_test_val_split(df, cfg)

    # Identify rows by a combined key
    def row_keys(part: pl.DataFrame) -> set[tuple]:
        return set(
            zip(
                part[_LABEL_COL].to_list(),
                part[_BATCH_COL].to_list(),
                part["Intensity_Mean"].to_list(),
            )
        )

    train_keys = row_keys(train)
    test_keys = row_keys(test)
    val_keys = row_keys(val)
    assert train_keys.isdisjoint(test_keys)
    assert train_keys.isdisjoint(val_keys)
    assert test_keys.isdisjoint(val_keys)


def test_train_test_val_split_sizes():
    df = _make_batch_df(n_batches=3, n_cells_per_batch=50, n_variants=2)
    cfg = _make_cfg()
    train, test, val = m.train_test_val_split(df, cfg)
    total = len(train) + len(test) + len(val)
    assert total == len(df)
    assert len(train) == pytest.approx(len(df) * 0.8, abs=5)


def test_train_test_val_split_all_batches_in_train():
    df = _make_batch_df(n_batches=3, n_cells_per_batch=40, n_variants=2)
    cfg = _make_cfg()
    train, _, _ = m.train_test_val_split(df, cfg)
    for variant in df[_LABEL_COL].unique().to_list():
        variant_train = train.filter(pl.col(_LABEL_COL) == variant)
        batches_in_train = set(variant_train[_BATCH_COL].unique().to_list())
        assert len(batches_in_train) == 3, (
            f"Variant {variant} should have all 3 batches in train, got {batches_in_train}"
        )


# ---------------------------------------------------------------------------
# train_batch_classifier
# ---------------------------------------------------------------------------


@pytest.fixture
def splits_and_cfg():
    df = _make_batch_df(n_batches=3, n_cells_per_batch=40, n_variants=1)
    cfg = _make_cfg()
    train, test, val = m.train_test_val_split(df, cfg)
    train_v = train.filter(pl.col(_LABEL_COL) == "V0")
    test_v = test.filter(pl.col(_LABEL_COL) == "V0")
    val_v = val.filter(pl.col(_LABEL_COL) == "V0")
    classes = sorted(df[_BATCH_COL].unique().to_list())
    return train_v, test_v, val_v, classes, cfg


def test_train_batch_classifier_returns_booster(splits_and_cfg):
    train, _, val, classes, cfg = splits_and_cfg
    model = m.train_batch_classifier(
        train, val, _FEATURE_COLS, _BATCH_COL, classes, cfg
    )
    assert isinstance(model, xgb.Booster)


# ---------------------------------------------------------------------------
# extract_ovr_stats
# ---------------------------------------------------------------------------


@pytest.fixture
def model_and_test(splits_and_cfg):
    train, test, val, classes, cfg = splits_and_cfg
    model = m.train_batch_classifier(
        train, val, _FEATURE_COLS, _BATCH_COL, classes, cfg
    )
    return model, test, classes, cfg


def test_extract_ovr_stats_length(model_and_test):
    model, test, classes, _ = model_and_test
    stats = m.extract_ovr_stats(model, test, _FEATURE_COLS, _BATCH_COL, classes)
    assert len(stats) == len(classes)


def test_extract_ovr_stats_keys(model_and_test):
    model, test, classes, _ = model_and_test
    stats = m.extract_ovr_stats(model, test, _FEATURE_COLS, _BATCH_COL, classes)
    expected_keys = {"batch", "auroc", "mw_pvalue", "n_batch_cells", "n_cells"}
    for row in stats:
        assert set(row.keys()) == expected_keys


def test_extract_ovr_stats_auroc_in_range(model_and_test):
    model, test, classes, _ = model_and_test
    stats = m.extract_ovr_stats(model, test, _FEATURE_COLS, _BATCH_COL, classes)
    for row in stats:
        assert 0.0 <= row["auroc"] <= 1.0, f"AUROC out of range: {row}"


def test_extract_ovr_stats_pvalue_in_range(model_and_test):
    model, test, classes, _ = model_and_test
    stats = m.extract_ovr_stats(model, test, _FEATURE_COLS, _BATCH_COL, classes)
    for row in stats:
        assert 0.0 <= row["mw_pvalue"] <= 1.0, f"p-value out of range: {row}"


def test_extract_ovr_stats_separable_batch_high_auroc(model_and_test):
    model, test, classes, _ = model_and_test
    stats = m.extract_ovr_stats(model, test, _FEATURE_COLS, _BATCH_COL, classes)
    batch0_row = next(r for r in stats if r["batch"] == "batch_0")
    assert batch0_row["auroc"] > 0.8, (
        f"Expected high AUROC for separable batch_0, got {batch0_row['auroc']}"
    )


# ---------------------------------------------------------------------------
# profile_variant
# ---------------------------------------------------------------------------


@pytest.fixture
def global_splits():
    df = _make_batch_df(n_batches=3, n_cells_per_batch=40, n_variants=2)
    cfg = _make_cfg()
    train, test, val = m.train_test_val_split(df, cfg)
    return train, test, val, cfg


def test_profile_variant_returns_list(global_splits):
    train, test, val, cfg = global_splits
    result = m.profile_variant("V0", train, test, val, _FEATURE_COLS, cfg)
    assert isinstance(result, list)


def test_profile_variant_one_entry_per_batch(global_splits):
    train, test, val, cfg = global_splits
    result = m.profile_variant("V0", train, test, val, _FEATURE_COLS, cfg)
    assert len(result) == 3


def test_profile_variant_has_variant_key(global_splits):
    train, test, val, cfg = global_splits
    result = m.profile_variant("V0", train, test, val, _FEATURE_COLS, cfg)
    for row in result:
        assert row["variant"] == "V0"


def test_profile_variant_too_few_cells_returns_empty():
    df = _make_batch_df(n_batches=2, n_cells_per_batch=30, n_variants=1)
    cfg = _make_cfg(min_cells=100_000)
    train, test, val = m.train_test_val_split(df, cfg)
    result = m.profile_variant("V0", train, test, val, _FEATURE_COLS, cfg)
    assert result == []


def test_profile_variant_single_batch_returns_empty():
    rng = np.random.default_rng(0)
    n = 60
    df = pl.DataFrame(
        {
            "Intensity_Mean": rng.random(n).tolist(),
            "Texture_Var": rng.random(n).tolist(),
            _LABEL_COL: ["V0"] * n,
            _BATCH_COL: ["only_batch"] * n,
        }
    )
    cfg = _make_cfg(min_batches=2)
    # Can't split with only one batch value — profile_variant receives empty splits
    # Build splits manually to avoid stratification error on single class
    train = df[:48]
    test = df[48:54]
    val = df[54:]
    result = m.profile_variant("V0", train, test, val, _FEATURE_COLS, cfg)
    assert result == []


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _make_structured_cfg(tmp_path: pathlib.Path, input_path: pathlib.Path) -> OmegaConf:
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
    bvb = BvbConfig(
        output_dir=str(tmp_path),
        input_file=str(input_path),
        label_column=_LABEL_COL,
        batch_column=_BATCH_COL,
        random_state=0,
        min_cells=10,
        min_batches=2,
        xgboost=xgb_cfg,
    )
    return OmegaConf.structured(bvb)


def test_main_creates_output_file(tmp_path):
    df = _make_batch_df(n_batches=3, n_cells_per_batch=40, n_variants=2)
    input_path = tmp_path / "input.parquet"
    df.write_parquet(input_path)
    cfg = _make_structured_cfg(tmp_path, input_path)
    with patch("fisseq_data_pipeline.batchvsbatch.setup_logging"):
        m.main.__wrapped__(cfg)
    assert (tmp_path / "results.parquet").exists()


def test_main_output_schema(tmp_path):
    df = _make_batch_df(n_batches=3, n_cells_per_batch=40, n_variants=2)
    input_path = tmp_path / "input.parquet"
    df.write_parquet(input_path)
    cfg = _make_structured_cfg(tmp_path, input_path)
    with patch("fisseq_data_pipeline.batchvsbatch.setup_logging"):
        m.main.__wrapped__(cfg)
    result = pl.read_parquet(tmp_path / "results.parquet")
    expected_cols = {
        "variant",
        "batch",
        "auroc",
        "mw_pvalue",
        "n_batch_cells",
        "n_cells",
    }
    assert expected_cols.issubset(set(result.columns))
