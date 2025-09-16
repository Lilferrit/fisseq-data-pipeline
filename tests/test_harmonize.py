from pathlib import Path

import numpy as np
import pytest

from fisseq_data_pipeline.harmonize import fit_harmonizer, harmonize


def _make_toy_data():
    """
    Create a tiny dataset with two batches and a few features.
    Returns X (n_samples, n_features) and batch_idx (n_samples,).
    """
    # Two batches with different means so ComBat has something to learn.
    X_a = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.5, 1.5, 2.5],
            [1.0, 2.0, 3.0],
        ],
        dtype=float,
    )
    X_b = np.array(
        [
            [5.0, 6.0, 7.0],
            [5.5, 6.5, 7.5],
            [6.0, 7.0, 8.0],
        ],
        dtype=float,
    )
    X = np.vstack([X_a, X_b])  # (6, 3)
    batch_idx = np.array(["A"] * 3 + ["B"] * 3, dtype=object)
    return X, batch_idx


@pytest.fixture
def cfg_path(tmp_path: Path) -> Path:
    """
    Provide a real PathLike for the `config` argument. The implementation
    of Config(config) in your code doesn't need to read it for these tests,
    but passing a real file avoids surprises.
    """
    p = tmp_path / "dummy.yaml"
    p.write_text("batch_col_name: SITE\n")
    return p


def test_fit_harmonizer_and_harmonize_shapes(cfg_path):
    """Fit a real harmonizer and apply it; verify shapes are preserved."""
    X, batch_idx = _make_toy_data()

    model = fit_harmonizer(
        feature_matrix=X,
        batch_idx=batch_idx,
        config=cfg_path,
        is_control=None,  # use all rows
    )
    assert isinstance(model, dict)

    X_h = harmonize(
        feature_matrix=X,
        batch_idx=batch_idx,
        config=cfg_path,
        harmonizer=model,
    )

    # Same shape, finite numbers
    assert X_h.shape == X.shape
    assert np.isfinite(X_h).all()


def test_fit_harmonizer_with_control_mask_and_apply(cfg_path):
    """
    Fit using only a control subset, then apply to the full dataset.
    Ensures model trains and application preserves shape.
    """
    X, batch_idx = _make_toy_data()
    # Mark the middle four samples as "control"
    is_control = np.array([False, True, True, True, True, False])

    model = fit_harmonizer(
        feature_matrix=X,
        batch_idx=batch_idx,
        config=cfg_path,
        is_control=is_control,
    )
    assert isinstance(model, dict)

    X_h = harmonize(
        feature_matrix=X,
        batch_idx=batch_idx,
        config=cfg_path,
        harmonizer=model,
    )
    assert X_h.shape == X.shape


def test_harmonize_on_new_data_with_same_features(cfg_path):
    """
    Train on one dataset and apply to a different dataset with the same
    number of features but different number of samples.
    """
    # Train on 6x3
    X_train, batch_train = _make_toy_data()
    model = fit_harmonizer(
        feature_matrix=X_train,
        batch_idx=batch_train,
        config=cfg_path,
        is_control=None,
    )

    # New data: 4x3, two from A-like range, two from B-like range
    X_new = np.array(
        [
            [0.25, 1.25, 2.25],
            [0.75, 1.75, 2.75],
            [5.25, 6.25, 7.25],
            [5.75, 6.75, 7.75],
        ],
        dtype=float,
    )
    batch_new = np.array(["A", "A", "B", "B"], dtype=object)

    X_new_h = harmonize(
        feature_matrix=X_new,
        batch_idx=batch_new,
        config=cfg_path,
        harmonizer=model,
    )

    assert X_new_h.shape == X_new.shape
    assert np.isfinite(X_new_h).all()
