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
    se_multiplier: float | None = 1.0,
) -> OmegaConf:
    return OmegaConf.structured(
        m.BlocklistConfig(
            output_dir=str(tmp_path / "bl_out"),
            correlation_files=correlation_files,
            minimum_correlation=minimum_correlation,
            se_multiplier=se_multiplier,
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


def expected_adjusted_r(r_values: list[float], se_multiplier: float | None) -> float | None:
    """Manual numpy ground truth for the se_multiplier-adjusted
    lower-confidence-bound estimate, mirroring blocklist.py's Fisher-z-space
    computation exactly (see BlocklistConfig.se_multiplier's docstring)."""
    z = np.arctanh(np.clip(np.array(r_values, dtype=float), -1 + _EPS, 1 - _EPS))
    n = len(z)
    if se_multiplier is None:
        return float(np.tanh(z.mean()))
    if n <= 1:
        return None
    se_z = float(z.std(ddof=1) / math.sqrt(n))
    return float(np.tanh(z.mean() - se_multiplier * se_z))


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


def test_main_feature_ok_truth_table_across_se_multiplier(tmp_path) -> None:
    """Truth table over se_multiplier=1.0 scenarios: a feature must clear
    minimum_correlation on its se_multiplier-adjusted (lower-confidence-
    bound) estimate, not on the raw r_est, to be marked feature_ok."""
    corr_dir = tmp_path / "corr"
    corr_dir.mkdir()

    # r values hand-picked (and verified against expected_adjusted_r) so each
    # feature lands unambiguously on one side of both r_est's raw magnitude
    # gate and the se_multiplier=1.0-adjusted gate.
    cases = {
        "mag_pass_adjusted_pass": [0.68, 0.70, 0.71, 0.69, 0.72],
        "mag_pass_adjusted_fail": [0.9, 0.15, 0.85, 0.1, 0.8, 0.2, 0.88, 0.05, 0.75, 0.25],
        "mag_fail_adjusted_fail": [0.28, 0.30, 0.29, 0.31, 0.27],
    }
    for feature, r_values in cases.items():
        write_replicates(corr_dir, feature, r_values)
        r_est, _se_z, _ = fisher_z_stats(r_values)
        adj = expected_adjusted_r(r_values, 1.0)
        assert adj is not None
        # Sanity-check the fixture actually lands where the test name claims.
        mag_pass = r_est >= 0.5
        adjusted_pass = adj >= 0.5
        assert feature == (
            f"mag_{'pass' if mag_pass else 'fail'}_adjusted_"
            f"{'pass' if adjusted_pass else 'fail'}"
        )

    result = run_main(
        tmp_path, make_bl_cfg(tmp_path, str(corr_dir / "*.parquet"), se_multiplier=1.0)
    )

    ok = dict(zip(result["feature"].to_list(), result["feature_ok"].to_list()))
    assert ok["mag_pass_adjusted_pass"] is True
    assert ok["mag_pass_adjusted_fail"] is False
    assert ok["mag_fail_adjusted_fail"] is False


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
    assert row["adjusted_r"] is None
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
        assert math.isfinite(row["adjusted_r"])


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
    exercising the default se_multiplier=1.0 lower-confidence-bound
    criterion: a tightly-clustered feature passes (small se_z barely
    penalizes r_est), and a high-mean-but-noisy feature's r_est clears
    minimum_correlation on its own but its adjusted_r (r_est penalized by
    1 se_z in Fisher-z space) does not."""
    corr_dir = tmp_path / "corr"
    corr_dir.mkdir()
    stable = [0.70, 0.68, 0.71, 0.69, 0.72, 0.70, 0.69, 0.71, 0.68, 0.72]
    noisy = [0.9, 0.15, 0.85, 0.1, 0.8, 0.2, 0.88, 0.05, 0.75, 0.25]
    write_replicates(corr_dir, "feature_stable", stable)
    write_replicates(corr_dir, "feature_noisy", noisy)

    result = run_main(tmp_path, make_bl_cfg(tmp_path, str(corr_dir / "*.parquet")))

    assert result.height == 2
    assert set(result.columns) == {
        "feature",
        "r_est",
        "se_z",
        "n_replicates",
        "adjusted_r",
        "feature_ok",
    }

    stable_row = get_row(result, "feature_stable")
    expected_r_est, expected_se_z, expected_n = fisher_z_stats(stable)
    assert stable_row["n_replicates"] == expected_n == 10
    assert stable_row["r_est"] == pytest.approx(expected_r_est)
    assert stable_row["se_z"] == pytest.approx(expected_se_z)
    assert stable_row["adjusted_r"] == pytest.approx(expected_adjusted_r(stable, 1.0))
    assert stable_row["feature_ok"] is True

    noisy_row = get_row(result, "feature_noisy")
    assert noisy_row["n_replicates"] == 10
    assert noisy_row["r_est"] >= 0.5  # clears the raw magnitude gate...
    assert noisy_row["adjusted_r"] < 0.5  # ...but the lower-confidence bound doesn't
    assert noisy_row["feature_ok"] is False


def test_main_se_multiplier_none_passthrough_and_reduces_gate(tmp_path) -> None:
    """se_multiplier=None disables the precision adjustment entirely:
    adjusted_r == r_est exactly, and feature_ok reduces to the raw
    r_est >= minimum_correlation comparison -- even for a k=1 feature, which
    fails automatically whenever se_multiplier is set (since se_z, and thus
    adjusted_r, is null)."""
    corr_dir = tmp_path / "corr"
    corr_dir.mkdir()
    write_replicates(corr_dir, "f1_mean", [0.5, 0.6, 0.7])
    write_replicates(corr_dir, "f2_mean", [0.1, 0.2, 0.3])
    write_replicates(corr_dir, "k1_mean", [0.9])

    result = run_main(
        tmp_path,
        make_bl_cfg(tmp_path, str(corr_dir / "*.parquet"), se_multiplier=None),
    )

    for feature in ["f1_mean", "f2_mean", "k1_mean"]:
        row = get_row(result, feature)
        assert row["adjusted_r"] == pytest.approx(row["r_est"])
        assert row["feature_ok"] == (row["r_est"] >= 0.5)

    k1_row = get_row(result, "k1_mean")
    assert k1_row["se_z"] is None
    assert k1_row["adjusted_r"] == pytest.approx(0.9)
    assert k1_row["feature_ok"] is True


def test_main_se_multiplier_monotonicity(tmp_path) -> None:
    """se_multiplier=2.0 is strictly more conservative than 1.0: every
    feature ok at 2.0 is also ok at 1.0, but not necessarily vice versa."""
    corr_dir = tmp_path / "corr"
    corr_dir.mkdir()
    cases = {
        "tight_high": [0.70, 0.71, 0.69, 0.70, 0.72, 0.69],
        "tight_low": [0.30, 0.31, 0.29, 0.30, 0.28, 0.31],
        "spread_high": [0.9, 0.2, 0.85, 0.15, 0.8, 0.6],
        "spread_borderline": [0.65, 0.3, 0.6, 0.35, 0.55, 0.45],
    }
    for feature, r_values in cases.items():
        write_replicates(corr_dir, feature, r_values)

    result_1 = run_main(
        tmp_path, make_bl_cfg(tmp_path, str(corr_dir / "*.parquet"), se_multiplier=1.0)
    )
    result_2 = run_main(
        tmp_path, make_bl_cfg(tmp_path, str(corr_dir / "*.parquet"), se_multiplier=2.0)
    )
    ok_1 = dict(zip(result_1["feature"].to_list(), result_1["feature_ok"].to_list()))
    ok_2 = dict(zip(result_2["feature"].to_list(), result_2["feature_ok"].to_list()))

    ok_at_2 = {f for f, ok in ok_2.items() if ok}
    ok_at_1 = {f for f, ok in ok_1.items() if ok}
    assert ok_at_2 <= ok_at_1
    # Fixture actually exercises the strict-subset property, not a
    # coincidental equality.
    assert ok_at_2 < ok_at_1


def test_main_se_multiplier_truth_table(tmp_path) -> None:
    """Explicit hand-picked (r_values) fixtures at se_multiplier 1.0 and 2.0,
    asserting exact adjusted_r (via expected_adjusted_r) and the resulting
    feature_ok in each cell."""
    corr_dir = tmp_path / "corr"
    corr_dir.mkdir()
    r_values = [0.75, 0.2, 0.7, 0.25, 0.65, 0.3]
    write_replicates(corr_dir, "f", r_values)

    for se_multiplier in (1.0, 2.0):
        result = run_main(
            tmp_path,
            make_bl_cfg(
                tmp_path, str(corr_dir / "*.parquet"), se_multiplier=se_multiplier
            ),
        )
        row = get_row(result, "f")
        expected = expected_adjusted_r(r_values, se_multiplier)
        assert row["adjusted_r"] == pytest.approx(expected)
        assert row["feature_ok"] == (expected >= 0.5)
