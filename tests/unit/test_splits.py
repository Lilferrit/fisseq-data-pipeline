from __future__ import annotations

import polars as pl

from fisseq_data_pipeline.utils.splits import (
    TMP_IDX_COL,
    add_row_index,
    filter_by_index_file,
    get_replicate_lf,
)

# ---------------------------------------------------------------------------
# add_row_index
# ---------------------------------------------------------------------------


def test_add_row_index_adds_column() -> None:
    lf = pl.DataFrame({"f1": [1.0, 2.0, 3.0]}).lazy()
    result = add_row_index(lf).collect()
    assert TMP_IDX_COL in result.columns


def test_add_row_index_starts_at_zero_and_is_contiguous() -> None:
    lf = pl.DataFrame({"f1": [1.0, 2.0, 3.0]}).lazy()
    result = add_row_index(lf).collect()
    assert result[TMP_IDX_COL].to_list() == [0, 1, 2]


# ---------------------------------------------------------------------------
# get_replicate_lf
# ---------------------------------------------------------------------------


def test_get_replicate_lf_includes_matching_rows() -> None:
    lf = pl.DataFrame({TMP_IDX_COL: [0, 1, 2, 3], "f1": [1.0, 2.0, 3.0, 4.0]}).lazy()
    result = get_replicate_lf(lf, [0, 2]).collect()
    assert sorted(result[TMP_IDX_COL].to_list()) == [0, 2]


def test_get_replicate_lf_excludes_non_matching_rows() -> None:
    lf = pl.DataFrame({TMP_IDX_COL: [0, 1, 2, 3], "f1": [1.0, 2.0, 3.0, 4.0]}).lazy()
    result = get_replicate_lf(lf, [0, 2]).collect()
    assert 1 not in result[TMP_IDX_COL].to_list()
    assert 3 not in result[TMP_IDX_COL].to_list()


def test_get_replicate_lf_empty_idx_returns_empty() -> None:
    lf = pl.DataFrame({TMP_IDX_COL: [0, 1, 2], "f1": [1.0, 2.0, 3.0]}).lazy()
    result = get_replicate_lf(lf, []).collect()
    assert len(result) == 0


def test_get_replicate_lf_preserves_all_columns() -> None:
    lf = pl.DataFrame(
        {TMP_IDX_COL: [0, 1], "f1": [1.0, 2.0], "meta_foo": ["a", "b"]}
    ).lazy()
    result = get_replicate_lf(lf, [0]).collect()
    assert set(result.columns) == {TMP_IDX_COL, "f1", "meta_foo"}


# ---------------------------------------------------------------------------
# filter_by_index_file
# ---------------------------------------------------------------------------


def test_filter_by_index_file_none_keeps_all_rows() -> None:
    lf = pl.DataFrame({"f1": [1.0, 2.0, 3.0]}).lazy()
    result = filter_by_index_file(lf, None).collect()
    assert len(result) == 3


def test_filter_by_index_file_none_drops_tmp_idx_col() -> None:
    lf = pl.DataFrame({"f1": [1.0, 2.0, 3.0]}).lazy()
    result = filter_by_index_file(lf, None).collect()
    assert TMP_IDX_COL not in result.columns


def test_filter_by_index_file_filters_to_given_indices(tmp_path) -> None:
    lf = pl.DataFrame({"f1": [10.0, 20.0, 30.0, 40.0]}).lazy()
    idx_path = tmp_path / "half.parquet"
    pl.DataFrame({TMP_IDX_COL: [1, 3]}).write_parquet(idx_path)

    result = filter_by_index_file(lf, idx_path).collect()
    assert sorted(result["f1"].to_list()) == [20.0, 40.0]


def test_filter_by_index_file_drops_tmp_idx_col_when_filtering(tmp_path) -> None:
    lf = pl.DataFrame({"f1": [10.0, 20.0]}).lazy()
    idx_path = tmp_path / "half.parquet"
    pl.DataFrame({TMP_IDX_COL: [0]}).write_parquet(idx_path)

    result = filter_by_index_file(lf, idx_path).collect()
    assert TMP_IDX_COL not in result.columns
