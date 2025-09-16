from pathlib import Path

import numpy as np
import pytest

import fisseq_data_pipeline.utils.config as mod_under_test
from fisseq_data_pipeline.normalize import fit_normalizer, normalize


class _DummyConfig:
    """Minimal stand-in for Config(config_arg)."""

    def __init__(self, cfg):
        self._raw = cfg


@pytest.fixture(autouse=True)
def _patch_config(monkeypatch):
    """Auto-apply a lightweight Config shim for all tests."""
    monkeypatch.setattr(mod_under_test, "Config", _DummyConfig)
    yield


def _col_mean_var(X):
    """Return (mean, var) with population variance (ddof=0), like StandardScaler."""
    return X.mean(axis=0), X.var(axis=0)


def test_fit_normalizer_uses_all_samples_when_no_mask():
    # 3x2 toy matrix
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 3.0],
            [2.0, 5.0],
        ],
        dtype=float,
    )

    norm = fit_normalizer(X, config=None, is_control=None)

    mu_np, var_np = _col_mean_var(X)
    # StandardScaler stores population mean/var
    np.testing.assert_allclose(norm.mean_, mu_np, rtol=0, atol=1e-12)
    np.testing.assert_allclose(norm.var_, var_np, rtol=0, atol=1e-12)


def test_fit_normalizer_uses_only_control_rows_when_mask_provided(tmp_path):
    # Control rows are the last two; their means/vars differ from full set
    X = np.array(
        [
            [0.0, 1.0],  # non-control
            [10.0, 5.0],  # control
            [14.0, 9.0],  # control
        ],
        dtype=float,
    )
    is_control = np.array([False, True, True])

    # Create a temporary config file so Config(PathLike) doesn't fail
    cfg_path = tmp_path / "dummy.yaml"
    cfg_path.write_text("dummy: true\n")

    norm = fit_normalizer(X, config=cfg_path, is_control=is_control)

    X_ctrl = X[is_control]
    mu_ctrl, var_ctrl = _col_mean_var(X_ctrl)
    np.testing.assert_allclose(norm.mean_, mu_ctrl, rtol=0, atol=1e-12)
    np.testing.assert_allclose(norm.var_, var_ctrl, rtol=0, atol=1e-12)

    # Sanity: confirm differs from full-data stats
    mu_all, var_all = _col_mean_var(X)
    assert not np.allclose(mu_all, mu_ctrl)
    assert not np.allclose(var_all, var_ctrl)


def test_normalize_matches_manual_standardization_no_mask():
    X = np.array(
        [
            [0.0, 1.0, 2.0],
            [2.0, 3.0, 6.0],
            [4.0, 5.0, 10.0],
            [6.0, 7.0, 14.0],
        ],
        dtype=float,
    )

    norm = fit_normalizer(X, config=None)
    X_hat = normalize(X, norm)

    # Columns should be ~zero-mean and unit-variance
    np.testing.assert_allclose(X_hat.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(X_hat.var(axis=0), 1.0, atol=1e-12)

    # Also check against manual formula (x - mean) / sqrt(var)
    mu, var = _col_mean_var(X)
    manual = (X - mu) / np.sqrt(var)
    np.testing.assert_allclose(X_hat, manual, rtol=0, atol=1e-12)


def test_normalize_with_control_fitting_standardizes_controls_as_expected():
    # Fit on controls only; when applied to all rows,
    # control rows should be ~0 mean / unit var (within controls),
    # non-control rows need not be centered at 0.
    X = np.array(
        [
            [0.0, 0.0],  # non-control
            [10.0, 5.0],  # control
            [12.0, 7.0],  # control
            [14.0, 9.0],  # control
        ],
        dtype=float,
    )
    is_control = np.array([False, True, True, True])

    norm = fit_normalizer(X, config=None, is_control=is_control)
    X_hat = normalize(X, norm)

    # Evaluate only control rows
    Xc_hat = X_hat[is_control]
    np.testing.assert_allclose(Xc_hat.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Xc_hat.var(axis=0), 1.0, atol=1e-12)

    # Non-control row shouldn't necessarily be near 0
    assert not np.allclose(X_hat[~is_control], np.zeros_like(X_hat[~is_control]))


def test_config_argument_accepts_pathlike_and_none(monkeypatch, tmp_path):
    # Ensure passing a PathLike or None doesn't raise (Config shim handles it).
    X = np.random.RandomState(0).randn(8, 3)

    # PathLike
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("dummy: true")
    norm1 = fit_normalizer(X, config=cfg_path)
    assert hasattr(norm1, "mean_") and hasattr(norm1, "var_")

    # None
    norm2 = fit_normalizer(X, config=None)
    assert hasattr(norm2, "mean_") and hasattr(norm2, "var_")
