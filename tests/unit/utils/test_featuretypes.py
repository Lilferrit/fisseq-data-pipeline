from __future__ import annotations

import polars as pl

from fisseq_data_pipeline.utils.featuretypes import join_feature_type_files

# ---------------------------------------------------------------------------
# join_feature_type_files
# ---------------------------------------------------------------------------


def _write(tmp_path, name: str, df: pl.DataFrame) -> str:
    path = tmp_path / name
    df.write_parquet(path)
    return str(path)


def test_join_feature_type_files_single_file(tmp_path) -> None:
    path = _write(
        tmp_path,
        "mean.parquet",
        pl.DataFrame({"meta_aa_changes": ["A", "B"], "f1_mean": [1.0, 2.0]}),
    )
    result = join_feature_type_files([path], "meta_aa_changes")
    assert result.columns == ["meta_aa_changes", "f1_mean"]
    assert result["f1_mean"].to_list() == [1.0, 2.0]


def test_join_feature_type_files_joins_on_label_column(tmp_path) -> None:
    mean_path = _write(
        tmp_path,
        "mean.parquet",
        pl.DataFrame({"meta_aa_changes": ["A", "B"], "f1_mean": [1.0, 2.0]}),
    )
    std_path = _write(
        tmp_path,
        "std.parquet",
        pl.DataFrame({"meta_aa_changes": ["A", "B"], "f1_std": [0.1, 0.2]}),
    )
    result = join_feature_type_files([mean_path, std_path], "meta_aa_changes")
    assert set(result.columns) == {"meta_aa_changes", "f1_mean", "f1_std"}
    row_a = result.filter(pl.col("meta_aa_changes") == "A")
    assert row_a["f1_mean"][0] == 1.0
    assert row_a["f1_std"][0] == 0.1


def test_join_feature_type_files_preserves_row_count(tmp_path) -> None:
    mean_path = _write(
        tmp_path,
        "mean.parquet",
        pl.DataFrame({"meta_aa_changes": ["A", "B", "C"], "f1_mean": [1.0, 2.0, 3.0]}),
    )
    std_path = _write(
        tmp_path,
        "std.parquet",
        pl.DataFrame({"meta_aa_changes": ["A", "B", "C"], "f1_std": [0.1, 0.2, 0.3]}),
    )
    result = join_feature_type_files([mean_path, std_path], "meta_aa_changes")
    assert len(result) == 3
