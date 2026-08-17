from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_data_pipeline.wtnullblocklist as m


def write_bootstrap_files(tmp_path, rows: dict[str, list[float]]) -> str:
    """Write one bootstrap-replicate parquet per entry of ``rows`` (feature ->
    list of per-bootstrap values, one file per bootstrap index) and return
    the glob pattern matching them all."""
    boot_dir = tmp_path / "boots"
    boot_dir.mkdir()
    n_bootstraps = len(next(iter(rows.values())))
    for b in range(n_bootstraps):
        pl.DataFrame(
            {
                "feature": list(rows.keys()),
                "value": [values[b] for values in rows.values()],
            }
        ).write_parquet(boot_dir / f"bootstrap_{b}.parquet")
    return str(boot_dir / "*.parquet")


def make_cfg(tmp_path, wt_null_files, *, tukey_multiplier=1.5) -> OmegaConf:
    return OmegaConf.structured(
        m.WtNullBlocklistConfig(
            output_dir=str(tmp_path / "out"),
            wt_null_files=wt_null_files,
            tukey_multiplier=tukey_multiplier,
        )
    )


def run_main(tmp_path, wt_null_files, **kwargs) -> pl.DataFrame:
    m.main.__wrapped__(make_cfg(tmp_path, wt_null_files, **kwargs))
    return pl.read_parquet(tmp_path / "out" / "blocklist.parquet")


def expected_fence(values: list[float], multiplier: float) -> float:
    arr = np.array(values)
    q1 = np.percentile(arr, 25, method="linear")
    q3 = np.percentile(arr, 75, method="linear")
    return q1 + multiplier * (q3 - q1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_output_schema(tmp_path) -> None:
    files = write_bootstrap_files(tmp_path, {"f1": [0.1, 0.2, 0.3]})
    result = run_main(tmp_path, files)
    assert set(result.columns) == {
        "feature",
        "feature_ok",
        "null_mean",
        "threshold",
        "n_bootstraps",
    }


def test_main_null_mean_averages_across_bootstraps(tmp_path) -> None:
    files = write_bootstrap_files(tmp_path, {"f1": [0.1, 0.3, 0.2]})
    result = run_main(tmp_path, files)
    assert result.filter(pl.col("feature") == "f1")["null_mean"][0] == pytest.approx(
        0.2
    )
    assert result.filter(pl.col("feature") == "f1")["n_bootstraps"][0] == 3


def test_main_tukey_fence_matches_hand_computed_ground_truth(tmp_path) -> None:
    # Five features with distinct null means spread out enough that the
    # fence sits strictly between the lowest and highest values.
    rows = {
        "f1": [0.05, 0.05, 0.05],
        "f2": [0.10, 0.10, 0.10],
        "f3": [0.20, 0.20, 0.20],
        "f4": [0.30, 0.30, 0.30],
        "f5": [0.90, 0.90, 0.90],
    }
    files = write_bootstrap_files(tmp_path, rows)
    result = run_main(tmp_path, files, tukey_multiplier=1.5)
    null_means = [0.05, 0.10, 0.20, 0.30, 0.90]
    fence = expected_fence(null_means, 1.5)
    assert result["threshold"][0] == pytest.approx(fence)
    for feature, mean in zip(rows, null_means):
        expected_ok = mean <= fence
        row = result.filter(pl.col("feature") == feature)
        assert row["feature_ok"][0] == expected_ok


def test_main_tukey_multiplier_changes_threshold(tmp_path) -> None:
    rows = {
        "f1": [0.05, 0.05],
        "f2": [0.10, 0.10],
        "f3": [0.20, 0.20],
        "f4": [0.90, 0.90],
    }
    files = write_bootstrap_files(tmp_path, rows)
    strict = run_main(tmp_path, files, tukey_multiplier=0.1)
    lenient = run_main(tmp_path, files, tukey_multiplier=5.0)
    assert strict["threshold"][0] != pytest.approx(lenient["threshold"][0])
    assert strict["threshold"][0] < lenient["threshold"][0]


def test_main_degenerate_feature_excluded_from_fence_and_blocked(tmp_path) -> None:
    rows = {
        "f1": [0.05, 0.05, 0.05],
        "f2": [0.10, 0.10, 0.10],
        "degenerate": [float("nan"), float("nan"), float("nan")],
    }
    files = write_bootstrap_files(tmp_path, rows)
    result = run_main(tmp_path, files)
    degenerate_row = result.filter(pl.col("feature") == "degenerate")
    assert degenerate_row["null_mean"][0] is None
    assert degenerate_row["n_bootstraps"][0] == 0
    assert degenerate_row["feature_ok"][0] is False
    # The fence should be computed only over f1/f2's finite null means --
    # not skewed by the degenerate (NaN) feature.
    fence = expected_fence([0.05, 0.10], 1.5)
    assert result["threshold"][0] == pytest.approx(fence)


def test_main_partial_nan_bootstraps_averaged_over_finite_values_only(
    tmp_path,
) -> None:
    rows = {
        "f1": [0.1, float("nan"), 0.3],
        "f2": [0.1, 0.1, 0.1],
    }
    files = write_bootstrap_files(tmp_path, rows)
    result = run_main(tmp_path, files)
    f1_row = result.filter(pl.col("feature") == "f1")
    assert f1_row["null_mean"][0] == pytest.approx(0.2)
    assert f1_row["n_bootstraps"][0] == 2


def test_main_all_features_degenerate_blocks_everything(tmp_path) -> None:
    rows = {
        "f1": [float("nan"), float("nan")],
        "f2": [float("nan"), float("nan")],
    }
    files = write_bootstrap_files(tmp_path, rows)
    result = run_main(tmp_path, files)
    assert result["feature_ok"].to_list() == [False, False]
    assert result["threshold"].null_count() == 2


def test_main_raises_on_empty_glob(tmp_path) -> None:
    with pytest.raises(ValueError):
        run_main(tmp_path, str(tmp_path / "nomatch" / "*.parquet"))
