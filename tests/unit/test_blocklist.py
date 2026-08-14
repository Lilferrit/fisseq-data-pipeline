from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_data_pipeline.blocklist as m

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPS = 1e-6


def make_bl_cfg(
    tmp_path,
    correlation_files,
    *,
    minimum_correlation: float = 0.5,
    max_se_z: float = 0.0884,
) -> OmegaConf:
    return OmegaConf.structured(
        m.BlocklistConfig(
            output_dir=str(tmp_path / "bl_out"),
            correlation_files=correlation_files,
            minimum_correlation=minimum_correlation,
            max_se_z=max_se_z,
        )
    )


def fisher_z_stats(r_values: list[float]) -> tuple[float, float | None, int]:
    """Manual numpy ground truth for the Fisher-z aggregation, used to
    cross-check ``blocklist.py``'s Polars implementation."""
    z = np.arctanh(np.clip(np.array(r_values, dtype=float), -1 + _EPS, 1 - _EPS))
    n = len(z)
    r_est = float(np.tanh(z.mean()))
    se_z = float(z.std(ddof=1) / math.sqrt(n)) if n > 1 else None
    return r_est, se_z, n


def write_replicates(corr_dir, feature: str, r_values: list[float | None]) -> None:
    """Write one bootstrap-replicate correlations.parquet per entry in
    ``r_values`` (``None`` produces a null ``r`` for that replicate)."""
    for i, r in enumerate(r_values, start=1):
        pl.DataFrame(
            {
                "feature": [feature],
                "r": pl.Series([r], dtype=pl.Float64),
                "r_squared": pl.Series(
                    [None if r is None else r**2], dtype=pl.Float64
                ),
            }
        ).write_parquet(corr_dir / f"bootstrap_{feature}_{i}.parquet")


def get_row(result: pl.DataFrame, feature: str) -> dict:
    return result.filter(pl.col("feature") == feature).to_dicts().pop()


def run_main(tmp_path, cfg) -> pl.DataFrame:
    with patch("fisseq_data_pipeline.blocklist.setup_logging"):
        m.main.__wrapped__(cfg)
    return pl.read_parquet(tmp_path / "bl_out" / "blocklist.parquet")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_fisher_z_averages_r_across_bootstraps(tmp_path) -> None:
    corr_dir = tmp_path / "corr"
    corr_dir.mkdir()
    r_values = [0.5, 0.6, 0.7]
    write_replicates(corr_dir, "f1_mean", r_values)

    result = run_main(tmp_path, make_bl_cfg(tmp_path, str(corr_dir / "*.parquet")))

    row = get_row(result, "f1_mean")
    expected_r_est, expected_se_z, expected_n = fisher_z_stats(r_values)
    assert row["r_est"] == pytest.approx(expected_r_est)
    assert row["se_z"] == pytest.approx(expected_se_z)
    assert row["n_replicates"] == expected_n


def test_main_feature_ok_requires_both_gates(tmp_path) -> None:
    """Truth table over the two independent gates: a feature must clear both
    the magnitude gate (r_est >= minimum_correlation) and the quality gate
    (se_z <= max_se_z) to be marked feature_ok."""
    corr_dir = tmp_path / "corr"
    corr_dir.mkdir()

    # r values hand-picked (and verified against fisher_z_stats) so each
    # feature lands unambiguously on one side of both gates at
    # minimum_correlation=0.5, max_se_z=0.0884.
    cases = {
        "mag_pass_qual_pass": [0.68, 0.70, 0.71, 0.69, 0.72],
        "mag_pass_qual_fail": [0.95, 0.2, 0.9, 0.1, 0.85],
        "mag_fail_qual_pass": [0.28, 0.30, 0.29, 0.31, 0.27],
        "mag_fail_qual_fail": [0.1, -0.3, 0.5, -0.1, 0.4],
    }
    for feature, r_values in cases.items():
        write_replicates(corr_dir, feature, r_values)
        r_est, se_z, _ = fisher_z_stats(r_values)
        assert se_z is not None
        # Sanity-check the fixture actually lands where the test name claims.
        mag_pass = r_est >= 0.5
        qual_pass = se_z <= 0.0884
        assert feature == (
            f"mag_{'pass' if mag_pass else 'fail'}_qual_{'pass' if qual_pass else 'fail'}"
        )

    result = run_main(tmp_path, make_bl_cfg(tmp_path, str(corr_dir / "*.parquet")))

    ok = dict(zip(result["feature"].to_list(), result["feature_ok"].to_list()))
    assert ok["mag_pass_qual_pass"] is True
    assert ok["mag_pass_qual_fail"] is False
    assert ok["mag_fail_qual_pass"] is False
    assert ok["mag_fail_qual_fail"] is False


def test_main_raises_on_empty_glob(tmp_path) -> None:
    with patch("fisseq_data_pipeline.blocklist.setup_logging"):
        with pytest.raises(ValueError):
            m.main.__wrapped__(
                make_bl_cfg(tmp_path, str(tmp_path / "nonexistent" / "*.parquet"))
            )


def test_main_k_equals_one_forces_feature_ok_false(tmp_path) -> None:
    """A single replicate can't support a precision claim: se_z is undefined
    (null) and feature_ok is False regardless of how high r_est is."""
    corr_dir = tmp_path / "corr"
    corr_dir.mkdir()
    write_replicates(corr_dir, "f1_mean", [0.99])

    result = run_main(tmp_path, make_bl_cfg(tmp_path, str(corr_dir / "*.parquet")))

    row = get_row(result, "f1_mean")
    assert row["n_replicates"] == 1
    assert row["se_z"] is None
    assert row["feature_ok"] is False


def test_main_extreme_r_values_stay_finite(tmp_path) -> None:
    """r at/near +/-1 must not produce inf/nan Fisher z after clipping."""
    corr_dir = tmp_path / "corr"
    corr_dir.mkdir()
    write_replicates(corr_dir, "f_hi", [1.0, 1.0, 0.99, 1.0, 0.98])
    write_replicates(corr_dir, "f_lo", [-1.0, -0.99, -1.0, -0.98, -1.0])

    result = run_main(tmp_path, make_bl_cfg(tmp_path, str(corr_dir / "*.parquet")))

    for feature in ["f_hi", "f_lo"]:
        row = get_row(result, feature)
        assert math.isfinite(row["r_est"])
        assert math.isfinite(row["se_z"])


def test_main_null_r_in_some_replicates_excluded_from_average(tmp_path) -> None:
    """A feature with a null r in some (not all) replicate files -- e.g. from
    CORRELATE_FEATURES's pl.corr returning null for a degenerate half -- is
    excluded from that replicate's contribution rather than crashing or
    silently dropping the feature row."""
    corr_dir = tmp_path / "corr"
    corr_dir.mkdir()
    non_null = [0.6, 0.65, 0.62, 0.58]
    write_replicates(corr_dir, "f1_mean", [*non_null, None])

    result = run_main(tmp_path, make_bl_cfg(tmp_path, str(corr_dir / "*.parquet")))

    assert result.filter(pl.col("feature") == "f1_mean").height == 1
    row = get_row(result, "f1_mean")
    expected_r_est, expected_se_z, expected_n = fisher_z_stats(non_null)
    assert row["n_replicates"] == expected_n == 4
    assert row["r_est"] == pytest.approx(expected_r_est)
    assert row["se_z"] == pytest.approx(expected_se_z)


def test_main_end_to_end_discriminates_noisy_vs_stable_feature(tmp_path) -> None:
    """Full k=10 fixture (matching the pipeline's default bootstrap_reps=10)
    exercising the default max_se_z=0.0884 exactly as it's calibrated: a
    tightly-clustered feature passes both gates, and a high-mean-but-noisy
    feature clears the magnitude gate but is excluded by the quality gate."""
    corr_dir = tmp_path / "corr"
    corr_dir.mkdir()
    stable = [0.70, 0.68, 0.71, 0.69, 0.72, 0.70, 0.69, 0.71, 0.68, 0.72]
    noisy = [0.9, 0.3, 0.85, 0.2, 0.8, 0.25, 0.88, 0.15, 0.82, 0.35]
    write_replicates(corr_dir, "feature_stable", stable)
    write_replicates(corr_dir, "feature_noisy", noisy)

    result = run_main(tmp_path, make_bl_cfg(tmp_path, str(corr_dir / "*.parquet")))

    assert result.height == 2
    assert set(result.columns) == {"feature", "r_est", "se_z", "n_replicates", "feature_ok"}

    stable_row = get_row(result, "feature_stable")
    expected_r_est, expected_se_z, expected_n = fisher_z_stats(stable)
    assert stable_row["n_replicates"] == expected_n == 10
    assert stable_row["r_est"] == pytest.approx(expected_r_est)
    assert stable_row["se_z"] == pytest.approx(expected_se_z)
    assert stable_row["feature_ok"] is True

    noisy_row = get_row(result, "feature_noisy")
    assert noisy_row["n_replicates"] == 10
    assert noisy_row["r_est"] >= 0.5  # clears the magnitude gate...
    assert noisy_row["se_z"] > 0.0884  # ...but fails the quality gate
    assert noisy_row["feature_ok"] is False
