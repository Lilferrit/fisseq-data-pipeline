import numpy as np
import polars as pl
import pytest

from fisseq_data_pipeline.utils.filtering import (
    _exclude_blocked_features,
    downsample_group_to_target,
    drop_small_groups,
)

_GROUP_COL = "meta_barcode"


def _make_group_df(group_counts: dict[str, int]) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    groups: list[str] = []
    for g, n in group_counts.items():
        groups.extend([g] * n)
    n_total = len(groups)
    return pl.DataFrame(
        {
            "Intensity_Mean": rng.random(n_total).tolist(),
            _GROUP_COL: groups,
        }
    )


# ---------------------------------------------------------------------------
# _exclude_blocked_features
# ---------------------------------------------------------------------------


def _write_feature_block_list(tmp_path, feature_ok: dict[str, bool]):
    path = tmp_path / "feature_block_list.parquet"
    pl.DataFrame(
        {
            "feature": list(feature_ok.keys()),
            "feature_ok": list(feature_ok.values()),
        }
    ).write_parquet(path)
    return path


def test_exclude_blocked_features_none_is_no_op():
    assert _exclude_blocked_features(["Intensity_Mean", "Texture_Var"], None) == [
        "Intensity_Mean",
        "Texture_Var",
    ]


def test_exclude_blocked_features_drops_blocked(tmp_path):
    path = _write_feature_block_list(
        tmp_path, {"Intensity_Mean": True, "Texture_Var": False}
    )
    result = _exclude_blocked_features(["Intensity_Mean", "Texture_Var"], str(path))
    assert result == ["Intensity_Mean"]


def test_exclude_blocked_features_all_ok_returns_unchanged(tmp_path):
    path = _write_feature_block_list(
        tmp_path, {"Intensity_Mean": True, "Texture_Var": True}
    )
    result = _exclude_blocked_features(["Intensity_Mean", "Texture_Var"], str(path))
    assert result == ["Intensity_Mean", "Texture_Var"]


# ---------------------------------------------------------------------------
# drop_small_groups
# ---------------------------------------------------------------------------


def test_drop_small_groups_removes_small_groups():
    df = _make_group_df({"bc1": 10, "bc2": 3})
    result = drop_small_groups(df, _GROUP_COL, min_rows=5)
    assert "bc2" not in result.get_column(_GROUP_COL).to_list()
    assert "bc1" in result.get_column(_GROUP_COL).to_list()


def test_drop_small_groups_retains_at_threshold():
    df = _make_group_df({"bc1": 5, "bc2": 4})
    result = drop_small_groups(df, _GROUP_COL, min_rows=5)
    assert set(result.get_column(_GROUP_COL).to_list()) == {"bc1"}


def test_drop_small_groups_no_op_when_all_pass():
    df = _make_group_df({"bc1": 10, "bc2": 8})
    result = drop_small_groups(df, _GROUP_COL, min_rows=5)
    assert len(result) == len(df)


def test_drop_small_groups_row_count():
    df = _make_group_df({"bc1": 10, "bc2": 3, "bc3": 7})
    result = drop_small_groups(df, _GROUP_COL, min_rows=5)
    assert len(result) == 10 + 7


def test_drop_small_groups_no_exemptions():
    # Unlike ovwt.filter_min_cells, every group (including a "special" one)
    # is subject to the same threshold.
    df = _make_group_df({"WT": 2, "V1": 10})
    result = drop_small_groups(df, _GROUP_COL, min_rows=5)
    assert set(result.get_column(_GROUP_COL).to_list()) == {"V1"}


# ---------------------------------------------------------------------------
# downsample_group_to_target
# ---------------------------------------------------------------------------


def test_downsample_group_to_target_reduces_to_max_other_group():
    df = _make_group_df({"target": 100, "other_a": 20, "other_b": 30})
    result = downsample_group_to_target(df, _GROUP_COL, "target", seed=0)
    target_count = result.filter(pl.col(_GROUP_COL) == "target").height
    assert target_count == 30


def test_downsample_group_to_target_preserves_other_rows():
    df = _make_group_df({"target": 100, "other_a": 20, "other_b": 30})
    result = downsample_group_to_target(df, _GROUP_COL, "target", seed=0)
    assert result.filter(pl.col(_GROUP_COL) == "other_a").height == 20
    assert result.filter(pl.col(_GROUP_COL) == "other_b").height == 30


def test_downsample_group_to_target_no_op_when_target_already_smaller():
    df = _make_group_df({"target": 10, "other": 50})
    result = downsample_group_to_target(df, _GROUP_COL, "target", seed=0)
    assert result.filter(pl.col(_GROUP_COL) == "target").height == 10


def test_downsample_group_to_target_no_op_when_target_equals_max():
    df = _make_group_df({"target": 50, "other": 50})
    result = downsample_group_to_target(df, _GROUP_COL, "target", seed=0)
    assert result.filter(pl.col(_GROUP_COL) == "target").height == 50


def test_downsample_group_to_target_reproducible_with_same_seed():
    df = _make_group_df({"target": 100, "other": 30})
    r1 = downsample_group_to_target(df, _GROUP_COL, "target", seed=7)
    r2 = downsample_group_to_target(df, _GROUP_COL, "target", seed=7)
    assert (
        r1.filter(pl.col(_GROUP_COL) == "target")
        .sort("Intensity_Mean")
        .get_column("Intensity_Mean")
        .to_list()
        == r2.filter(pl.col(_GROUP_COL) == "target")
        .sort("Intensity_Mean")
        .get_column("Intensity_Mean")
        .to_list()
    )


def test_downsample_group_to_target_total_row_count():
    df = _make_group_df({"target": 100, "other": 30})
    result = downsample_group_to_target(df, _GROUP_COL, "target", seed=0)
    assert len(result) == 30 + 30


def test_downsample_group_to_target_integer_reduces_to_target():
    df = _make_group_df({"target": 100, "other": 30})
    result = downsample_group_to_target(df, _GROUP_COL, "target", seed=0, n=50)
    assert result.filter(pl.col(_GROUP_COL) == "target").height == 50


def test_downsample_group_to_target_integer_no_op_when_smaller_than_target():
    df = _make_group_df({"target": 10, "other": 30})
    result = downsample_group_to_target(df, _GROUP_COL, "target", seed=0, n=5000)
    assert result.filter(pl.col(_GROUP_COL) == "target").height == 10


def test_downsample_group_to_target_integer_no_op_when_equals_target():
    df = _make_group_df({"target": 50, "other": 30})
    result = downsample_group_to_target(df, _GROUP_COL, "target", seed=0, n=50)
    assert result.filter(pl.col(_GROUP_COL) == "target").height == 50


def test_downsample_group_to_target_no_other_groups_leaves_target_untouched():
    # target = max() over an empty group_by is None -> "target is not None"
    # guard keeps this a no-op rather than erroring.
    df = _make_group_df({"target": 40})
    result = downsample_group_to_target(df, _GROUP_COL, "target", seed=0)
    assert len(result) == 40


@pytest.mark.parametrize("n", [0, None])
def test_downsample_group_to_target_various_n_semantics(n):
    df = _make_group_df({"target": 100, "other": 10})
    result = downsample_group_to_target(df, _GROUP_COL, "target", seed=0, n=n)
    expected = 10 if n is None else n
    assert result.filter(pl.col(_GROUP_COL) == "target").height == expected
