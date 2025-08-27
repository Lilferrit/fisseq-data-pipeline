import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

import fisseq_data_pipeline.normalize as mod


class StubConfig:
    """Minimal stub that matches the constructor signature used in the module."""

    def __init__(self, cfg):
        # store anything you want to assert if needed
        self.cfg = cfg
        # attributes are not used by our stubs, so we keep it minimal


def test_fit_normalizer_uses_control_features_and_fits(monkeypatch):
    # Arrange
    # Replace Config with a stub to avoid touching real config logic
    monkeypatch.setattr(mod, "Config", StubConfig, raising=True)

    # Build a dummy LazyFrame (not actually used by our stubs, but passed through)
    lf = pl.DataFrame({"x": [0]}).lazy()

    # Control feature matrix we expect the scaler to fit on
    control_mat = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=float)

    # Stub get_control_samples: pass-through or any transform; we just return the same lf
    calls = {"get_control_samples": 0, "get_feature_matrix": 0}

    def fake_get_control_samples(data_lf, config):
        calls["get_control_samples"] += 1
        return data_lf

    def fake_get_feature_matrix(data_lf, config):
        calls["get_feature_matrix"] += 1
        return (["f1", "f2"], control_mat)

    monkeypatch.setattr(
        mod, "get_control_samples", fake_get_control_samples, raising=True
    )
    monkeypatch.setattr(
        mod, "get_feature_matrix", fake_get_feature_matrix, raising=True
    )

    # Act
    scaler = mod.fit_normalizer(lf, config={"anything": "ok"})

    # Assert
    assert isinstance(scaler, StandardScaler)
    assert calls["get_control_samples"] == 1
    assert calls["get_feature_matrix"] == 1

    # Check fitted statistics
    # Means: [2.0, 20.0]; std (ddof=0) from StandardScaler: [~0.81649658, ~8.1649658]
    np.testing.assert_allclose(scaler.mean_, np.array([2.0, 20.0]))
    np.testing.assert_allclose(
        scaler.scale_, np.array([np.std(control_mat[:, 0]), np.std(control_mat[:, 1])])
    )


def test_normalize_transforms_features_and_sets_columns(monkeypatch):
    # Arrange
    monkeypatch.setattr(mod, "Config", StubConfig, raising=True)

    # Make a simple dataset (lazy)
    data = pl.DataFrame(
        {"f1": [1.0, 2.0, 3.0], "f2": [10.0, 20.0, 30.0], "other": [7, 8, 9]}
    )
    data_lf = data.lazy()

    # Prepare a scaler fitted on the same features so we know the transform
    scaler = StandardScaler()
    X = np.column_stack([data["f1"].to_numpy(), data["f2"].to_numpy()])
    scaler.fit(X)

    # get_feature_matrix should return (feature_cols, matrix) from the input lf
    feature_cols = ["f1", "f2"]

    def fake_get_feature_matrix(data_lf_arg, config):
        # Return original features as numpy
        return feature_cols, X.copy()

    # Capture what set_feature_matrix is called with, and
    # return a sentinel LazyFrame to ensure normalize returns what set_feature_matrix returns.
    captured = {}

    def fake_set_feature_matrix(data_lf_arg, feature_cols_arg, new_features_arg):
        captured["data_lf_arg"] = data_lf_arg
        captured["feature_cols_arg"] = feature_cols_arg
        captured["new_features_arg"] = new_features_arg
        # Build a small DF to act as the "result"
        return pl.DataFrame(
            {
                "f1": new_features_arg[:, 0],
                "f2": new_features_arg[:, 1],
                "other": [7, 8, 9],
            }
        ).lazy()

    monkeypatch.setattr(
        mod, "get_feature_matrix", fake_get_feature_matrix, raising=True
    )
    monkeypatch.setattr(
        mod, "set_feature_matrix", fake_set_feature_matrix, raising=True
    )

    # Act
    out_lf = mod.normalize(data_lf, config={"whatever": 1}, normalizer=scaler)

    # Assert: normalize should have provided transformed features to set_feature_matrix
    assert captured["feature_cols_arg"] == feature_cols
    transformed_expected = scaler.transform(X)
    np.testing.assert_allclose(captured["new_features_arg"], transformed_expected)

    # And the function returns what set_feature_matrix returned
    out = out_lf.collect()
    np.testing.assert_allclose(out["f1"].to_numpy(), transformed_expected[:, 0])
    np.testing.assert_allclose(out["f2"].to_numpy(), transformed_expected[:, 1])
    assert out["other"].to_list() == [7, 8, 9]


def test_normalize_works_with_nontrivial_feature_order(monkeypatch):
    """Ensure the (cols, matrix) order from get_feature_matrix is passed through unchanged."""
    monkeypatch.setattr(mod, "Config", StubConfig, raising=True)

    data = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]}).lazy()

    cols = ["b", "a"]
    mat = np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]], dtype=float)

    # Fit scaler on this order
    scaler = StandardScaler().fit(mat)

    def fake_get_feature_matrix(_lf, _cfg):
        return cols, mat.copy()

    called = {"set_called": False}

    def fake_set_feature_matrix(_lf, feature_cols, new_features):
        called["set_called"] = True
        assert feature_cols == cols  # order preserved
        # Return something collectable
        return pl.DataFrame({"b": new_features[:, 0], "a": new_features[:, 1]}).lazy()

    monkeypatch.setattr(
        mod, "get_feature_matrix", fake_get_feature_matrix, raising=True
    )
    monkeypatch.setattr(
        mod, "set_feature_matrix", fake_set_feature_matrix, raising=True
    )

    out_lf = mod.normalize(data, config={}, normalizer=scaler)
    out = out_lf.collect()

    assert called["set_called"] is True
    # Output should be standardized (mean ~0, std ~1)
    np.testing.assert_allclose(out["b"].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(out["a"].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(out["b"].std(ddof=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(out["a"].std(ddof=0), 1.0, atol=1e-12)
