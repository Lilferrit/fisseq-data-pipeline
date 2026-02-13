# test_medmad.py
import pickle

import numpy as np
import polars as pl
import pytest

from fisseq_data_pipeline.medmad import MedMadNormalizer, fit_normalizer, normalize


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation (unscaled)."""
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def data_df():
    """
    Simple feature matrix + embedded metadata columns.

    Includes a constant feature ('f_zero_mad') that should be dropped because MAD==0.
    """
    return pl.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "f_zero_mad": [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
            "_meta_batch": ["A", "A", "A", "B", "B", "B"],
            "_meta_is_control": [True, False, True, False, True, False],
        }
    )


@pytest.fixture
def expected_global():
    """Expected global medians and MADs (computed manually)."""
    f1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    f2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    return {
        "median_f1": float(np.median(f1)),
        "median_f2": float(np.median(f2)),
        "mad_f1": _mad(f1),
        "mad_f2": _mad(f2),
    }


@pytest.fixture
def expected_global_control():
    """Expected global medians and MADs computed on control rows only."""
    f1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    f2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    is_control = np.array([True, False, True, False, True, False])

    f1c = f1[is_control]
    f2c = f2[is_control]
    return {
        "median_f1": float(np.median(f1c)),
        "median_f2": float(np.median(f2c)),
        "mad_f1": _mad(f1c),
        "mad_f2": _mad(f2c),
    }


@pytest.fixture
def expected_batchwise():
    """Expected per-batch medians and MADs (computed manually)."""
    f1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    f2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    # Batch A: first three rows
    a_f1, a_f2 = f1[:3], f2[:3]
    # Batch B: last three rows
    b_f1, b_f2 = f1[3:], f2[3:]

    return {
        "A": {
            "median_f1": float(np.median(a_f1)),
            "median_f2": float(np.median(a_f2)),
            "mad_f1": _mad(a_f1),
            "mad_f2": _mad(a_f2),
        },
        "B": {
            "median_f1": float(np.median(b_f1)),
            "median_f2": float(np.median(b_f2)),
            "mad_f1": _mad(b_f1),
            "mad_f2": _mad(b_f2),
        },
    }


# ---------------------------------------------------------------------
# fit_normalizer tests
# ---------------------------------------------------------------------
def test_fit_medmad_global(data_df, expected_global):
    """Should compute global median/MAD across all samples."""
    norm = fit_normalizer(data_df.lazy(), fit_only_on_control=False, fit_batch_wise=False)
    assert isinstance(norm, MedMadNormalizer)
    assert norm.is_batch_wise is False

    # Constant feature should be dropped
    assert "f_zero_mad" not in norm.medians.columns
    assert "f_zero_mad" not in norm.mads.columns

    # Should have pseudo-batch key
    assert "_meta_batch" in norm.medians.columns
    assert "_meta_batch" in norm.mads.columns
    assert norm.medians.shape[0] == 1
    assert norm.mads.shape[0] == 1

    # Check numerical accuracy
    row_med = norm.medians.row(0)
    row_mad = norm.mads.row(0)
    assert np.isclose(row_med[norm.medians.columns.index("f1")], expected_global["median_f1"])
    assert np.isclose(row_med[norm.medians.columns.index("f2")], expected_global["median_f2"])
    assert np.isclose(row_mad[norm.mads.columns.index("f1")], expected_global["mad_f1"])
    assert np.isclose(row_mad[norm.mads.columns.index("f2")], expected_global["mad_f2"])


def test_fit_medmad_batchwise(data_df, expected_batchwise):
    """Should compute correct per-batch medians and MADs."""
    norm = fit_normalizer(data_df.lazy(), fit_only_on_control=False, fit_batch_wise=True)
    assert norm.is_batch_wise is True

    # Two batches
    assert norm.medians.shape[0] == 2
    assert norm.mads.shape[0] == 2
    assert "_meta_batch" in norm.medians.columns
    assert "_meta_batch" in norm.mads.columns

    # Constant feature should be dropped
    assert "f_zero_mad" not in norm.medians.columns
    assert "f_zero_mad" not in norm.mads.columns

    # Validate batch A
    a_med = norm.medians.filter(pl.col("_meta_batch") == "A").row(0)
    a_mad = norm.mads.filter(pl.col("_meta_batch") == "A").row(0)
    assert np.isclose(a_med[norm.medians.columns.index("f1")], expected_batchwise["A"]["median_f1"])
    assert np.isclose(a_med[norm.medians.columns.index("f2")], expected_batchwise["A"]["median_f2"])
    assert np.isclose(a_mad[norm.mads.columns.index("f1")], expected_batchwise["A"]["mad_f1"])
    assert np.isclose(a_mad[norm.mads.columns.index("f2")], expected_batchwise["A"]["mad_f2"])

    # Validate batch B
    b_med = norm.medians.filter(pl.col("_meta_batch") == "B").row(0)
    b_mad = norm.mads.filter(pl.col("_meta_batch") == "B").row(0)
    assert np.isclose(b_med[norm.medians.columns.index("f1")], expected_batchwise["B"]["median_f1"])
    assert np.isclose(b_med[norm.medians.columns.index("f2")], expected_batchwise["B"]["median_f2"])
    assert np.isclose(b_mad[norm.mads.columns.index("f1")], expected_batchwise["B"]["mad_f1"])
    assert np.isclose(b_mad[norm.mads.columns.index("f2")], expected_batchwise["B"]["mad_f2"])


def test_fit_medmad_control_only_global(data_df, expected_global_control):
    """Should filter to _meta_is_control == True before computing stats."""
    norm = fit_normalizer(data_df.lazy(), fit_only_on_control=True, fit_batch_wise=False)
    assert isinstance(norm, MedMadNormalizer)
    assert norm.is_batch_wise is False

    # Constant feature should be dropped
    assert "f_zero_mad" not in norm.medians.columns
    assert "f_zero_mad" not in norm.mads.columns

    # Check numerical accuracy
    row_med = norm.medians.row(0)
    row_mad = norm.mads.row(0)
    assert np.isclose(row_med[norm.medians.columns.index("f1")], expected_global_control["median_f1"])
    assert np.isclose(row_med[norm.medians.columns.index("f2")], expected_global_control["median_f2"])
    assert np.isclose(row_mad[norm.mads.columns.index("f1")], expected_global_control["mad_f1"])
    assert np.isclose(row_mad[norm.mads.columns.index("f2")], expected_global_control["mad_f2"])


# ---------------------------------------------------------------------
# normalize tests
# ---------------------------------------------------------------------
def test_normalize_global_has_median_0_mad_1(data_df):
    """
    After global normalization, each feature should have median≈0 and MAD≈1
    (robust standardization), not necessarily mean≈0/std≈1.
    """
    norm = fit_normalizer(data_df.lazy(), fit_batch_wise=False)
    out = normalize(data_df.lazy(), norm).collect()

    # Only retained feature columns
    feat_cols = [c for c in out.columns if not c.startswith("_meta_")]
    assert "f_zero_mad" not in feat_cols  # dropped at fit time

    for c in feat_cols:
        x = out[c].to_numpy()
        assert np.isclose(np.median(x), 0.0, atol=1e-8), f"{c} median={np.median(x)}"
        assert np.isclose(_mad(x), 1.0, atol=1e-8), f"{c} mad={_mad(x)}"


def test_normalize_batchwise_has_median_0_mad_1_per_batch(data_df):
    """After batch-wise normalization, each batch should have median≈0 and MAD≈1."""
    norm = fit_normalizer(data_df.lazy(), fit_batch_wise=True)
    out = normalize(data_df.lazy(), norm).collect()

    feat_cols = [c for c in out.columns if not c.startswith("_meta_")]
    assert "f_zero_mad" not in feat_cols

    for batch in ["A", "B"]:
        batch_df = out.filter(pl.col("_meta_batch") == batch)
        for c in feat_cols:
            x = batch_df[c].to_numpy()
            assert np.isclose(np.median(x), 0.0, atol=1e-8), f"{batch}:{c} median={np.median(x)}"
            assert np.isclose(_mad(x), 1.0, atol=1e-8), f"{batch}:{c} mad={_mad(x)}"


def test_normalize_drops_missing_columns(data_df, caplog):
    """Should drop feature columns not present in the normalizer and log a warning."""
    norm = fit_normalizer(data_df.lazy(), fit_batch_wise=True)

    df_extra = data_df.with_columns(pl.Series("extra", [1, 2, 3, 4, 5, 6]))
    with caplog.at_level("WARNING"):
        out = normalize(df_extra.lazy(), norm).collect()

    assert "extra" not in out.columns
    assert any("Dropped" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------
# Round-trip sanity (normalize then "denormalize")
# ---------------------------------------------------------------------
def test_normalize_and_denormalize_reversibility_batchwise(data_df):
    """
    Normalize then de-normalize should recover original values (for retained features).
    """
    norm = fit_normalizer(data_df.lazy(), fit_batch_wise=True)
    normed = normalize(data_df.lazy(), norm).collect()

    retained = [c for c in ["f1", "f2"] if c in norm.medians.columns]  # excludes dropped
    assert retained == ["f1", "f2"]

    recovered_parts = []
    for batch in ["A", "B"]:
        # batch stats
        med = (
            norm.medians.filter(pl.col("_meta_batch") == batch)
            .select(retained)
            .to_numpy()
        )  # shape (1, nfeat)
        mad = (
            norm.mads.filter(pl.col("_meta_batch") == batch)
            .select(retained)
            .to_numpy()
        )  # shape (1, nfeat)

        # normalized values for batch
        xb = normed.filter(pl.col("_meta_batch") == batch).select(retained).to_numpy()

        # invert: x = x_norm * mad + median
        recovered_parts.append(xb * mad + med)

    recovered = np.vstack(recovered_parts)
    original = data_df.select(retained).to_numpy()

    assert np.allclose(recovered, original, atol=1e-8)


# ---------------------------------------------------------------------
# Persistence sanity
# ---------------------------------------------------------------------
def test_normalizer_pickle_roundtrip(tmp_path, data_df):
    """save()/pickle roundtrip should preserve statistics."""
    norm = fit_normalizer(data_df.lazy(), fit_batch_wise=True)
    p = tmp_path / "medmad.pkl"
    norm.save(p)

    with open(p, "rb") as f:
        loaded = pickle.load(f)

    assert isinstance(loaded, MedMadNormalizer)
    assert loaded.is_batch_wise == norm.is_batch_wise
    assert loaded.medians.frame_equal(norm.medians)
    assert loaded.mads.frame_equal(norm.mads)
