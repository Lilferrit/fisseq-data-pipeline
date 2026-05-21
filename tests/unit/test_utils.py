from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from fisseq_data_pipeline.constants import (
    CONTROL_COLUMN_NAME,
    IMPACT_SCORE_COL,
    META_BARCODE_COL,
    META_BATCH_COL,
)
from fisseq_data_pipeline.utils import (
    DOT_COL,
    NORM_COL,
    compute_impact_score,
    compute_norm,
    compute_query_dot,
    get_aggregate_meta_data,
    load_batches,
)

# ---------------------------------------------------------------------------
# get_aggregate_meta_data
# ---------------------------------------------------------------------------

# 2 variants × 2 batches × 2 barcodes/variant — straightforward ground truth
_AGG_META_LF = pl.LazyFrame(
    {
        "meta_aa_changes": ["WT", "WT", "WT", "WT", "M1K", "M1K"],
        META_BARCODE_COL: ["bc_a", "bc_a", "bc_b", "bc_b", "bc_c", "bc_c"],
        META_BATCH_COL: ["batch1", "batch2", "batch1", "batch2", "batch1", "batch2"],
    }
)


def _row(df: pl.DataFrame, label: str, label_col: str = "meta_aa_changes") -> dict:
    return df.filter(pl.col(label_col) == label).row(0, named=True)


def test_get_aggregate_meta_data_num_cells() -> None:
    df = get_aggregate_meta_data(_AGG_META_LF, "meta_aa_changes").collect()
    assert _row(df, "WT")["meta_num_cells"] == 4
    assert _row(df, "M1K")["meta_num_cells"] == 2


def test_get_aggregate_meta_data_barcode_num_unique() -> None:
    df = get_aggregate_meta_data(_AGG_META_LF, "meta_aa_changes").collect()
    assert _row(df, "WT")[f"{META_BARCODE_COL}_num_unique"] == 2  # bc_a, bc_b
    assert _row(df, "M1K")[f"{META_BARCODE_COL}_num_unique"] == 1  # bc_c only


def test_get_aggregate_meta_data_batch_num_unique() -> None:
    df = get_aggregate_meta_data(_AGG_META_LF, "meta_aa_changes").collect()
    assert _row(df, "WT")[f"{META_BATCH_COL}_num_unique"] == 2
    assert _row(df, "M1K")[f"{META_BATCH_COL}_num_unique"] == 2


def test_get_aggregate_meta_data_barcode_counts_column_present() -> None:
    df = get_aggregate_meta_data(_AGG_META_LF, "meta_aa_changes").collect()
    assert f"{META_BARCODE_COL}_counts" in df.columns


def test_get_aggregate_meta_data_batch_counts_column_present() -> None:
    df = get_aggregate_meta_data(_AGG_META_LF, "meta_aa_changes").collect()
    assert f"{META_BATCH_COL}_counts" in df.columns


def test_get_aggregate_meta_data_warns_on_missing_column(
    caplog: pytest.LogCaptureFixture,
) -> None:
    lf = pl.LazyFrame({"meta_aa_changes": ["WT"], META_BARCODE_COL: ["bc_a"]})
    import logging

    with caplog.at_level(logging.WARNING):
        get_aggregate_meta_data(lf, "meta_aa_changes")
    assert META_BATCH_COL in caplog.text


def test_get_aggregate_meta_data_missing_batch_col_skips_batch_columns() -> None:
    lf = pl.LazyFrame(
        {"meta_aa_changes": ["WT", "WT"], META_BARCODE_COL: ["bc_a", "bc_b"]}
    )
    df = get_aggregate_meta_data(lf, "meta_aa_changes").collect()
    assert f"{META_BATCH_COL}_num_unique" not in df.columns
    assert f"{META_BATCH_COL}_counts" not in df.columns
    assert f"{META_BARCODE_COL}_num_unique" in df.columns
    assert f"{META_BARCODE_COL}_counts" in df.columns


def test_get_aggregate_meta_data_missing_barcode_col_skips_barcode_columns() -> None:
    lf = pl.LazyFrame({"meta_aa_changes": ["WT"], META_BATCH_COL: ["batch1"]})
    df = get_aggregate_meta_data(lf, "meta_aa_changes").collect()
    assert f"{META_BARCODE_COL}_num_unique" not in df.columns
    assert f"{META_BARCODE_COL}_counts" not in df.columns
    assert f"{META_BATCH_COL}_num_unique" in df.columns
    assert f"{META_BATCH_COL}_counts" in df.columns


def test_get_aggregate_meta_data_missing_both_cols_returns_only_num_cells() -> None:
    lf = pl.LazyFrame({"meta_aa_changes": ["WT", "M1K"]})
    df = get_aggregate_meta_data(lf, "meta_aa_changes").collect()
    assert set(df.columns) == {"meta_aa_changes", "meta_num_cells"}


def test_get_aggregate_meta_data_num_cells_correct_when_col_missing() -> None:
    lf = pl.LazyFrame(
        {
            "meta_aa_changes": ["WT", "WT", "M1K"],
            META_BARCODE_COL: ["bc_a", "bc_b", "bc_c"],
        }
    )
    df = get_aggregate_meta_data(lf, "meta_aa_changes").collect()
    assert _row(df, "WT")["meta_num_cells"] == 2
    assert _row(df, "M1K")["meta_num_cells"] == 1


# ---------------------------------------------------------------------------
# load_batches
# ---------------------------------------------------------------------------


@pytest.fixture
def single_parquet(tmp_path: Path) -> Path:
    df = pl.DataFrame({"meta_aa_changes": ["WT", "A1B"], "f1": [1.0, 2.0]})
    p = tmp_path / "batch_a.parquet"
    df.write_parquet(p)
    return p


@pytest.fixture
def multi_parquet(tmp_path: Path) -> Path:
    for stem, val in [("batch_x", 1.0), ("batch_y", 2.0)]:
        pl.DataFrame({"meta_aa_changes": ["WT"], "f1": [val]}).write_parquet(
            tmp_path / f"{stem}.parquet"
        )
    return tmp_path


def test_single_file_loads(single_parquet: Path) -> None:
    lf, _ = load_batches(str(single_parquet))
    df = lf.collect()
    assert len(df) == 2
    assert META_BATCH_COL in df.columns


def test_single_file_batch_name(single_parquet: Path) -> None:
    df, _ = load_batches(str(single_parquet))
    assert df.collect()[META_BATCH_COL].unique().to_list() == ["batch_a"]


def test_single_file_output_stem(single_parquet: Path) -> None:
    _, stem = load_batches(str(single_parquet))
    assert stem == "batch_a"


def test_glob_matches_multiple_files(multi_parquet: Path) -> None:
    lf, _ = load_batches(str(multi_parquet / "*.parquet"))
    df = lf.collect()
    assert len(df) == 2
    batch_names = sorted(df[META_BATCH_COL].unique().to_list())
    assert batch_names == ["batch_x", "batch_y"]


def test_glob_output_stem_is_output(multi_parquet: Path) -> None:
    _, stem = load_batches(str(multi_parquet / "*.parquet"))
    assert stem == "output"


def test_glob_batch_name_is_stem(multi_parquet: Path) -> None:
    lf, _ = load_batches(str(multi_parquet / "*.parquet"))
    for row in lf.collect().iter_rows(named=True):
        assert row[META_BATCH_COL] in ("batch_x", "batch_y")


def test_no_match_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No files matched"):
        load_batches(str(tmp_path / "*.parquet"))


# ---------------------------------------------------------------------------
# compute_norm
# ---------------------------------------------------------------------------


def test_compute_norm_pythagorean() -> None:
    lf = pl.LazyFrame({"f1": [3.0], "f2": [4.0]})
    result = compute_norm(lf).collect()
    assert result[NORM_COL][0] == pytest.approx(5.0)


def test_compute_norm_unit_vector() -> None:
    lf = pl.LazyFrame({"f1": [1.0], "f2": [0.0]})
    result = compute_norm(lf).collect()
    assert result[NORM_COL][0] == pytest.approx(1.0)


def test_compute_norm_multiple_rows() -> None:
    lf = pl.LazyFrame({"f1": [3.0, 0.0], "f2": [4.0, 5.0]})
    result = compute_norm(lf).collect()
    assert result[NORM_COL].to_list() == pytest.approx([5.0, 5.0])


def test_compute_norm_preserves_meta_columns() -> None:
    lf = pl.LazyFrame({"meta_label": ["a"], "f1": [3.0], "f2": [4.0]})
    result = compute_norm(lf).collect()
    assert "meta_label" in result.columns
    assert result["meta_label"][0] == "a"


def test_compute_norm_excludes_meta_from_norm() -> None:
    # meta columns must not contribute to the norm value
    lf = pl.LazyFrame({"meta_label": [99.0], "f1": [3.0], "f2": [4.0]})
    result = compute_norm(lf).collect()
    assert result[NORM_COL][0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# compute_query_dot
# ---------------------------------------------------------------------------


def test_compute_query_dot_basic() -> None:
    value_lf = pl.LazyFrame({"f1": [1.0], "f2": [2.0]})
    query_lf = pl.LazyFrame({"f1": [3.0], "f2": [4.0]})
    result = compute_query_dot(value_lf, query_lf).collect()
    assert result[DOT_COL][0] == pytest.approx(11.0)  # 1·3 + 2·4


def test_compute_query_dot_orthogonal() -> None:
    value_lf = pl.LazyFrame({"f1": [1.0], "f2": [0.0]})
    query_lf = pl.LazyFrame({"f1": [0.0], "f2": [1.0]})
    result = compute_query_dot(value_lf, query_lf).collect()
    assert result[DOT_COL][0] == pytest.approx(0.0)


def test_compute_query_dot_drops_query_columns() -> None:
    value_lf = pl.LazyFrame({"f1": [1.0], "f2": [2.0]})
    query_lf = pl.LazyFrame({"f1": [3.0], "f2": [4.0]})
    result = compute_query_dot(value_lf, query_lf).collect()
    assert "f1_query" not in result.columns
    assert "f2_query" not in result.columns


def test_compute_query_dot_preserves_meta_columns() -> None:
    value_lf = pl.LazyFrame({"meta_label": ["x"], "f1": [1.0], "f2": [2.0]})
    query_lf = pl.LazyFrame({"f1": [3.0], "f2": [4.0]})
    result = compute_query_dot(value_lf, query_lf).collect()
    assert "meta_label" in result.columns
    assert result["meta_label"][0] == "x"


def test_compute_query_dot_raises_on_empty_query() -> None:
    value_lf = pl.LazyFrame({"f1": [1.0]})
    query_lf = pl.LazyFrame({"f1": pl.Series([], dtype=pl.Float64)})
    with pytest.raises(ValueError, match="query must have exactly 1 row"):
        compute_query_dot(value_lf, query_lf)


def test_compute_query_dot_raises_on_multi_row_query() -> None:
    value_lf = pl.LazyFrame({"f1": [1.0]})
    query_lf = pl.LazyFrame({"f1": [1.0, 2.0]})
    with pytest.raises(ValueError, match="query must have exactly 1 row"):
        compute_query_dot(value_lf, query_lf)


# ---------------------------------------------------------------------------
# compute_impact_score
# ---------------------------------------------------------------------------


def test_compute_impact_score_control_is_zero() -> None:
    lf = pl.LazyFrame({CONTROL_COLUMN_NAME: [True], "f1": [1.0], "f2": [0.0]})
    result = compute_impact_score(lf).collect()
    assert result[IMPACT_SCORE_COL][0] == pytest.approx(0.0, abs=1e-9)


def test_compute_impact_score_orthogonal_is_half() -> None:
    # Control along f1; test row along f2 — cosine angle = 90° → score = 0.5
    lf = pl.LazyFrame(
        {
            CONTROL_COLUMN_NAME: [True, False],
            "f1": [1.0, 0.0],
            "f2": [0.0, 1.0],
        }
    )
    result = compute_impact_score(lf).collect()
    non_ctrl = result.filter(pl.col(CONTROL_COLUMN_NAME).not_())
    assert non_ctrl[IMPACT_SCORE_COL][0] == pytest.approx(0.5, abs=1e-9)


def test_compute_impact_score_opposite_is_one() -> None:
    # Opposite direction to the control median → max impact score
    lf = pl.LazyFrame(
        {
            CONTROL_COLUMN_NAME: [True, False],
            "f1": [1.0, -1.0],
            "f2": [0.0, 0.0],
        }
    )
    result = compute_impact_score(lf).collect()
    non_ctrl = result.filter(pl.col(CONTROL_COLUMN_NAME).not_())
    assert non_ctrl[IMPACT_SCORE_COL][0] == pytest.approx(1.0, abs=1e-9)


def test_compute_impact_score_drops_temp_columns() -> None:
    lf = pl.LazyFrame({CONTROL_COLUMN_NAME: [True], "f1": [1.0], "f2": [0.0]})
    result = compute_impact_score(lf).collect()
    assert NORM_COL not in result.columns
    assert DOT_COL not in result.columns


def test_compute_impact_score_output_columns() -> None:
    lf = pl.LazyFrame({CONTROL_COLUMN_NAME: [True], "f1": [1.0], "f2": [0.0]})
    result = compute_impact_score(lf).collect()
    assert IMPACT_SCORE_COL in result.columns
    assert CONTROL_COLUMN_NAME in result.columns
    assert "f1" in result.columns


def test_compute_impact_score_null_columns_excluded_from_calc() -> None:
    # f2 has a null so it is excluded; score is computed using f1 only.
    # Control median (f1=1); non-control (f1=-1) → opposite direction → score = 1.
    lf = pl.LazyFrame(
        {
            CONTROL_COLUMN_NAME: [True, False],
            "f1": [1.0, -1.0],
            "f2": [None, 1.0],
        }
    )
    result = compute_impact_score(lf).collect()
    non_ctrl = result.filter(pl.col(CONTROL_COLUMN_NAME).not_())
    assert non_ctrl[IMPACT_SCORE_COL][0] == pytest.approx(1.0, abs=1e-9)


def test_compute_impact_score_null_columns_kept_in_output() -> None:
    lf = pl.LazyFrame(
        {
            CONTROL_COLUMN_NAME: [True],
            "f1": [1.0],
            "f2": [None],
        }
    )
    result = compute_impact_score(lf).collect()
    assert "f2" in result.columns


def test_compute_impact_score_uses_control_median() -> None:
    # Two control rows whose median is (1, 0); verify the non-control score
    lf = pl.LazyFrame(
        {
            CONTROL_COLUMN_NAME: [True, True, False],
            "f1": [0.0, 2.0, 0.0],  # median f1 = 1.0
            "f2": [0.0, 0.0, 1.0],
        }
    )
    result = compute_impact_score(lf).collect()
    non_ctrl = result.filter(pl.col(CONTROL_COLUMN_NAME).not_())
    # control median = (1, 0); test = (0, 1) → orthogonal → 0.5
    assert non_ctrl[IMPACT_SCORE_COL][0] == pytest.approx(0.5, abs=1e-9)
