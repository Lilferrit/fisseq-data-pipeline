from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from fisseq_data_pipeline.utils.batches import load_batches
from fisseq_data_pipeline.utils.constants import META_BATCH_COL

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


@pytest.fixture
def subdirectory_parquet(tmp_path: Path) -> Path:
    """Two files with the same name in different subdirectories."""
    for subdir in ("batch1", "batch2"):
        d = tmp_path / subdir
        d.mkdir()
        pl.DataFrame({"meta_aa_changes": ["WT"], "f1": [1.0]}).write_parquet(
            d / "filtered_cells.parquet"
        )
    return tmp_path


def test_use_parent_name_assigns_directory_as_batch(subdirectory_parquet: Path) -> None:
    lf, _ = load_batches(
        str(subdirectory_parquet / "*/filtered_cells.parquet"), use_parent_name=True
    )
    batch_names = sorted(lf.collect()[META_BATCH_COL].unique().to_list())
    assert batch_names == ["batch1", "batch2"]


def test_use_parent_name_false_assigns_stem(subdirectory_parquet: Path) -> None:
    lf, _ = load_batches(
        str(subdirectory_parquet / "*/filtered_cells.parquet"), use_parent_name=False
    )
    batch_names = lf.collect()[META_BATCH_COL].unique().to_list()
    assert batch_names == ["filtered_cells"]
