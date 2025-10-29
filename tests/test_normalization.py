# test_normalizer.py
import numpy as np
import polars as pl
import pytest

from fisseq_data_pipeline.normalize import Normalizer, fit_normalizer, normalize


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def feature_df():
    """Simple feature matrix with a zero-variance column."""
    return pl.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "f_zero": [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
        }
    )


@pytest.fixture
def meta_df():
    """Metadata defining batches and control flags."""
    return pl.DataFrame(
        {
            "_batch": ["A", "A", "A", "B", "B", "B"],
            "_is_control": [True, False, True, False, True, False],
        }
    )


@pytest.fixture
def expected_global():
    """Expected global statistics computed manually."""
    f1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    f2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    return {
        "mean_f1": np.mean(f1),
        "mean_f2": np.mean(f2),
        "std_f1": np.std(f1, ddof=1),
        "std_f2": np.std(f2, ddof=1),
    }


@pytest.fixture
def expected_global_control():
    f1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    f2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    is_control = np.array([True, False, True, False, True, False])

    return {
        "mean_f1": np.mean(f1[is_control]),
        "mean_f2": np.mean(f2[is_control]),
        "std_f1": np.std(f1[is_control], ddof=1),
        "std_f2": np.std(f2[is_control], ddof=1),
    }


@pytest.fixture
def expected_batchwise():
    """Expected per-batch means and stds (computed manually)."""
    f1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    f2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    # Batch A: first three rows
    batch_a = {
        "mean_f1": np.mean(f1[:3]),
        "std_f1": np.std(f1[:3], ddof=1),
        "mean_f2": np.mean(f2[:3]),
        "std_f2": np.std(f2[:3], ddof=1),
    }
    # Batch B: last three rows
    batch_b = {
        "mean_f1": np.mean(f1[3:]),
        "std_f1": np.std(f1[3:], ddof=1),
        "mean_f2": np.mean(f2[3:]),
        "std_f2": np.std(f2[3:], ddof=1),
    }
    return {"A": batch_a, "B": batch_b}


# ---------------------------------------------------------------------
# fit_normalizer tests
# ---------------------------------------------------------------------
def test_fit_normalizer_global(feature_df, expected_global):
    """Should compute global mean/std across all samples."""
    norm = fit_normalizer(feature_df.lazy(), meta_data_df=None, fit_batch_wise=False)
    assert isinstance(norm, Normalizer)
    assert not norm.is_batch_wise

    # f_zero should be dropped (zero variance)
    assert "f_zero" not in norm.means.columns

    # Check numerical accuracy
    assert np.isclose(norm.means["f1"][0], expected_global["mean_f1"])
    assert np.isclose(norm.means["f2"][0], expected_global["mean_f2"])
    assert np.isclose(norm.stds["f1"][0], expected_global["std_f1"])
    assert np.isclose(norm.stds["f2"][0], expected_global["std_f2"])


def test_fit_normalizer_batchwise(feature_df, meta_df, expected_batchwise):
    """Should compute correct batch-wise statistics with per-batch rows."""
    norm = fit_normalizer(feature_df.lazy(), meta_df.lazy(), fit_batch_wise=True)

    # still batch-wise
    assert norm.is_batch_wise
    assert norm.means.shape[0] == 2
    assert norm.stds.shape[0] == 2
    assert "_batch" in norm.means.columns
    assert "_batch" in norm.stds.columns

    # look up per-batch stats by the _batch column directly
    a_row = norm.means.filter(pl.col("_batch") == "A").row(0)
    b_row = norm.stds.filter(pl.col("_batch") == "B").row(0)

    # Check mean and std values against expected ground truth
    assert np.isclose(
        a_row[norm.means.columns.index("f1")], expected_batchwise["A"]["mean_f1"]
    )
    assert np.isclose(
        b_row[norm.stds.columns.index("f2")], expected_batchwise["B"]["std_f2"]
    )


def test_fit_normalizer_control_only(feature_df, meta_df, expected_global_control):
    """Should filter to _is_control == True before computing stats."""
    norm = fit_normalizer(
        feature_df.lazy(),
        meta_df.lazy(),
        fit_only_on_control=True,
        fit_batch_wise=False,
    )

    # Verify Normalizer object structure
    assert isinstance(norm, Normalizer)
    assert norm.is_batch_wise is False
    assert "_batch" in norm.means.columns
    assert "_batch" in norm.stds.columns

    # f_zero should be dropped (zero variance)
    assert "f_zero" not in norm.means.columns
    assert "f_zero" not in norm.stds.columns

    # Since fit_batch_wise=False, there’s only one row (global stats)
    means_row = norm.means.row(0)
    stds_row = norm.stds.row(0)

    # Check numerical accuracy against expected control-only stats
    assert np.isclose(
        means_row[norm.means.columns.index("f1")], expected_global_control["mean_f1"]
    )
    assert np.isclose(
        means_row[norm.means.columns.index("f2")], expected_global_control["mean_f2"]
    )
    assert np.isclose(
        stds_row[norm.stds.columns.index("f1")], expected_global_control["std_f1"]
    )
    assert np.isclose(
        stds_row[norm.stds.columns.index("f2")], expected_global_control["std_f2"]
    )


def test_fit_normalizer_requires_metadata(feature_df):
    """Should raise ValueError when meta_data_df missing but required."""
    with pytest.raises(ValueError):
        fit_normalizer(feature_df, None, fit_batch_wise=True)
    with pytest.raises(ValueError):
        fit_normalizer(feature_df, None, fit_only_on_control=True)


# ---------------------------------------------------------------------
# normalize tests
# ---------------------------------------------------------------------
def test_normalize_global(feature_df):
    """Should normalize without metadata when mapping=None."""
    norm = fit_normalizer(feature_df.lazy(), fit_batch_wise=False)
    out = normalize(feature_df.lazy(), norm)
    out = out.collect()
    # Each column now mean≈0, std≈1
    for col in out.columns:
        arr = out[col].to_numpy()
        assert np.isclose(np.mean(arr), 0.0, atol=1e-8)
        assert np.isclose(np.std(arr, ddof=1), 1.0, atol=1e-8)


def test_normalize_batchwise(feature_df, meta_df):
    """Should normalize per batch so each batch has mean≈0 and std≈1."""
    norm = fit_normalizer(feature_df.lazy(), meta_df.lazy(), fit_batch_wise=True)
    out = normalize(feature_df.lazy(), norm, meta_df.lazy()).collect()

    # Split back into batch A (rows 0–2) and batch B (rows 3–5)
    out_a = out.slice(0, 3)
    out_b = out.slice(3, 3)

    # Verify each batch is approximately standardized
    for batch_df in (out_a, out_b):
        for col in batch_df.columns:
            arr = batch_df[col].to_numpy()
            mean = np.mean(arr)
            std = np.std(arr, ddof=1)
            assert np.isclose(mean, 0.0, atol=1e-8), f"{col} mean={mean}"
            assert np.isclose(std, 1.0, atol=1e-8), f"{col} std={std}"


def test_normalize_requires_metadata(feature_df, meta_df):
    """Should raise ValueError if mapping exists but no meta_data_df given."""
    norm = fit_normalizer(feature_df.lazy(), meta_df.lazy(), fit_batch_wise=True)
    with pytest.raises(ValueError):
        normalize(feature_df, norm, meta_data_df=None)


def test_normalize_drops_missing_columns(feature_df, meta_df, caplog):
    """Should drop columns not in normalizer and log a warning."""
    norm = fit_normalizer(feature_df.lazy(), meta_df.lazy(), fit_batch_wise=True)
    df_extra = feature_df.with_columns(pl.Series("extra", [1, 2, 3, 4, 5, 6]))
    with caplog.at_level("WARNING"):
        out = normalize(df_extra.lazy(), norm, meta_df.lazy())
    assert "extra" not in out.columns
    assert any("Dropped" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------
# Round-trip sanity
# ---------------------------------------------------------------------
def test_normalize_and_denormalize_reversibility(feature_df, meta_df):
    """Normalize then de-normalize should roughly recover original values."""
    # Fit and normalize (lazy → eager)
    norm = fit_normalizer(feature_df.lazy(), meta_df.lazy(), fit_batch_wise=True)
    normed = normalize(feature_df.lazy(), norm, meta_df.lazy()).collect()

    # Recover each batch's stats from Normalizer
    recovered_batches = []
    for batch in meta_df["_batch"].unique():
        # Get mask for this batch
        batch_mask = meta_df["_batch"] == batch

        # Extract batch statistics
        mu = (
            norm.means.filter(pl.col("_batch") == batch)
            .select(pl.exclude("_batch"))
            .to_numpy()
        )
        sigma = (
            norm.stds.filter(pl.col("_batch") == batch)
            .select(pl.exclude("_batch"))
            .to_numpy()
        )

        # Get normalized features for this batch
        normed_batch = normed.filter(pl.lit(batch_mask)).to_numpy()

        # Undo z-score normalization
        recovered_batches.append(normed_batch * sigma + mu)

    recovered = np.vstack(recovered_batches)

    # Compare to original feature values (excluding dropped zero-var columns)
    retained_cols = [c for c in feature_df.columns if c in norm.means.columns]
    original = feature_df.select(retained_cols).to_numpy()

    assert np.allclose(recovered, original, atol=1e-8)
