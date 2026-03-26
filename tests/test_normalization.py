import pickle

import numpy as np
import polars as pl
import pytest

from fisseq_data_pipeline.normalize import Normalizer, fit_normalizer, normalize

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def data_lf():
    """
    LazyFrame with two real features, one zero-variance feature, and meta cols.
    Batch A: rows 0-2, Batch B: rows 3-5.
    """
    return pl.DataFrame(
        {
            "_meta_batch": ["A", "A", "A", "B", "B", "B"],
            "_meta_is_control": [True, False, True, False, True, False],
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "f_zero": [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
        }
    ).lazy()


@pytest.fixture
def data_lf_no_zero(data_lf):
    return data_lf.select(["_meta_batch", "_meta_is_control", "f1", "f2"])


# ---------------------------------------------------------------------------
# fit_normalizer — basic structure
# ---------------------------------------------------------------------------


def test_fit_normalizer_returns_normalizer(data_lf_no_zero):
    norm = fit_normalizer(data_lf_no_zero, fit_batch_wise=False)
    assert isinstance(norm, Normalizer)


def test_fit_normalizer_is_batch_wise_flag(data_lf_no_zero):
    assert fit_normalizer(data_lf_no_zero, fit_batch_wise=False).is_batch_wise is False
    assert fit_normalizer(data_lf_no_zero, fit_batch_wise=True).is_batch_wise is True


def test_fit_normalizer_drops_zero_variance(data_lf):
    norm = fit_normalizer(data_lf, fit_batch_wise=False)
    assert "f_zero" not in norm.means.columns
    assert "f_zero" not in norm.stds.columns
    assert "f1" in norm.means.columns
    assert "f2" in norm.means.columns


# ---------------------------------------------------------------------------
# fit_normalizer — global (fit_batch_wise=False)
# ---------------------------------------------------------------------------


def test_fit_normalizer_global_shape(data_lf_no_zero):
    norm = fit_normalizer(data_lf_no_zero, fit_batch_wise=False)
    assert norm.means.shape[0] == 1
    assert norm.stds.shape[0] == 1


def test_fit_normalizer_global_values(data_lf_no_zero):
    norm = fit_normalizer(data_lf_no_zero, fit_batch_wise=False)

    f1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    f2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    assert np.isclose(norm.means["f1"][0], np.mean(f1))
    assert np.isclose(norm.means["f2"][0], np.mean(f2))
    assert np.isclose(norm.stds["f1"][0], np.std(f1, ddof=1))
    assert np.isclose(norm.stds["f2"][0], np.std(f2, ddof=1))


# ---------------------------------------------------------------------------
# fit_normalizer — batch-wise (fit_batch_wise=True)
# ---------------------------------------------------------------------------


def test_fit_normalizer_batchwise_shape(data_lf_no_zero):
    norm = fit_normalizer(data_lf_no_zero, fit_batch_wise=True)
    assert norm.means.shape[0] == 2
    assert norm.stds.shape[0] == 2
    assert "_meta_batch" in norm.means.columns
    assert "_meta_batch" in norm.stds.columns


def test_fit_normalizer_batchwise_values(data_lf_no_zero):
    norm = fit_normalizer(data_lf_no_zero, fit_batch_wise=True)

    f1_a = np.array([1.0, 2.0, 3.0])
    f2_b = np.array([40.0, 50.0, 60.0])

    a_means = norm.means.filter(pl.col("_meta_batch") == "A")
    b_stds = norm.stds.filter(pl.col("_meta_batch") == "B")

    assert np.isclose(a_means["f1"][0], np.mean(f1_a))
    assert np.isclose(b_stds["f2"][0], np.std(f2_b, ddof=1))


# ---------------------------------------------------------------------------
# fit_normalizer — control-only (fit_only_on_control=True)
# ---------------------------------------------------------------------------


def test_fit_normalizer_control_only_values(data_lf_no_zero):
    # control rows (is_control=True): rows 0, 2, 4 → f1=[1,3,5], f2=[10,30,50]
    norm = fit_normalizer(
        data_lf_no_zero, fit_batch_wise=False, fit_only_on_control=True
    )

    f1_ctrl = np.array([1.0, 3.0, 5.0])
    f2_ctrl = np.array([10.0, 30.0, 50.0])

    assert np.isclose(norm.means["f1"][0], np.mean(f1_ctrl))
    assert np.isclose(norm.means["f2"][0], np.mean(f2_ctrl))
    assert np.isclose(norm.stds["f1"][0], np.std(f1_ctrl, ddof=1))
    assert np.isclose(norm.stds["f2"][0], np.std(f2_ctrl, ddof=1))


# ---------------------------------------------------------------------------
# normalize — return type and structure
# ---------------------------------------------------------------------------


def test_normalize_returns_lazy_frame(data_lf_no_zero):
    norm = fit_normalizer(data_lf_no_zero, fit_batch_wise=False)
    result = normalize(data_lf_no_zero, norm)
    assert isinstance(result, pl.LazyFrame)


def test_normalize_output_has_feature_and_meta_columns(data_lf_no_zero):
    norm = fit_normalizer(data_lf_no_zero, fit_batch_wise=False)
    out = normalize(data_lf_no_zero, norm).collect()
    assert "f1" in out.columns
    assert "f2" in out.columns
    assert "_meta_batch" in out.columns


# ---------------------------------------------------------------------------
# normalize — global standardization
# ---------------------------------------------------------------------------


def test_normalize_global_standardizes(data_lf_no_zero):
    norm = fit_normalizer(data_lf_no_zero, fit_batch_wise=False)
    out = normalize(data_lf_no_zero, norm).collect()

    for col in ["f1", "f2"]:
        arr = out[col].to_numpy()
        assert np.isclose(
            np.mean(arr), 0.0, atol=1e-6
        ), f"{col} mean off: {np.mean(arr)}"
        assert np.isclose(np.std(arr, ddof=1), 1.0, atol=1e-6), f"{col} std off"


# ---------------------------------------------------------------------------
# normalize — batch-wise standardization
# ---------------------------------------------------------------------------


def test_normalize_batchwise_standardizes(data_lf_no_zero):
    norm = fit_normalizer(data_lf_no_zero, fit_batch_wise=True)
    out = normalize(data_lf_no_zero, norm).collect()

    for batch in ["A", "B"]:
        batch_out = out.filter(pl.col("_meta_batch") == batch)
        for col in ["f1", "f2"]:
            arr = batch_out[col].to_numpy()
            assert np.isclose(
                np.mean(arr), 0.0, atol=1e-6
            ), f"batch {batch} {col} mean off"
            assert np.isclose(
                np.std(arr, ddof=1), 1.0, atol=1e-6
            ), f"batch {batch} {col} std off"


# ---------------------------------------------------------------------------
# normalize — column handling
# ---------------------------------------------------------------------------


def test_normalize_drops_columns_not_in_normalizer(data_lf_no_zero, caplog):
    norm = fit_normalizer(data_lf_no_zero, fit_batch_wise=False)
    lf_extra = data_lf_no_zero.with_columns(pl.lit(99.0).alias("extra"))

    with caplog.at_level("WARNING"):
        out = normalize(lf_extra, norm).collect()

    assert "extra" not in out.columns
    assert "f1" in out.columns
    assert any("Dropped" in rec.message for rec in caplog.records)


def test_normalize_zero_var_column_absent_from_output(data_lf):
    # f_zero is in data_lf but gets dropped by fit_normalizer
    norm = fit_normalizer(data_lf, fit_batch_wise=False)
    out = normalize(data_lf, norm).collect()
    assert "f_zero" not in out.columns


# ---------------------------------------------------------------------------
# normalize — round-trip
# ---------------------------------------------------------------------------


def test_normalize_roundtrip(data_lf_no_zero):
    norm = fit_normalizer(data_lf_no_zero, fit_batch_wise=True)
    normed = normalize(data_lf_no_zero, norm).collect()

    recovered_parts = []
    for batch in ["A", "B"]:
        mu = (
            norm.means.filter(pl.col("_meta_batch") == batch)
            .select(["f1", "f2"])
            .to_numpy()
        )
        sigma = (
            norm.stds.filter(pl.col("_meta_batch") == batch)
            .select(["f1", "f2"])
            .to_numpy()
        )
        batch_normed = (
            normed.filter(pl.col("_meta_batch") == batch)
            .select(["f1", "f2"])
            .to_numpy()
        )
        recovered_parts.append(batch_normed * sigma + mu)

    recovered = np.vstack(recovered_parts)
    original = data_lf_no_zero.select(["f1", "f2"]).collect().to_numpy()
    assert np.allclose(recovered, original, atol=1e-5)


# ---------------------------------------------------------------------------
# Normalizer.save / reload
# ---------------------------------------------------------------------------


def test_normalizer_save_and_reload(tmp_path, data_lf_no_zero):
    norm = fit_normalizer(data_lf_no_zero, fit_batch_wise=True)
    path = tmp_path / "normalizer.pkl"
    norm.save(path)

    with open(path, "rb") as f:
        loaded = pickle.load(f)

    assert isinstance(loaded, Normalizer)
    assert loaded.is_batch_wise == norm.is_batch_wise
    assert loaded.means.equals(norm.means)
    assert loaded.stds.equals(norm.stds)
