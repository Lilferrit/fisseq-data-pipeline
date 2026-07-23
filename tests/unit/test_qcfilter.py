import pathlib
from unittest.mock import patch

import polars as pl
import pytest
from omegaconf import OmegaConf
from polars.testing import assert_frame_equal

import fisseq_data_pipeline.qcfilter as m
from fisseq_data_pipeline.qcfilter import (
    DOWNSAMPLE_TAG,
    QcFilterConfig,
    add_downsampled_pseudo_variants,
    add_qc_queries,
    combine_cell_files,
    filter_columns,
    get_barcode_counts,
    get_barcodes_per_variant,
    read_file,
)

# get_barcode_counts and get_barcodes_per_variant expect data that has
# already been through filter_columns, so their inputs use meta_* column names.


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg():
    return OmegaConf.create(
        {
            "barcode_col_name": "upBarcode",
            "aa_changes_col_name": "aaChanges",
            "edit_distance_col_name": "editDistance",
            "label_column": "meta_aa_changes",
            "bc_threshold": 3,
            "variant_bc_threshold": 2,
            "edit_distance_threshold": 1,
        }
    )


def _make_cell_df(
    barcodes: list[str],
    aa_changes: list[str],
    edit_distances: list[int] | None = None,
) -> pl.DataFrame:
    """Build a minimal raw cell-level DataFrame for testing."""
    n = len(barcodes)
    return pl.DataFrame(
        {
            "upBarcode": barcodes,
            "aaChanges": aa_changes,
            "variantType": ["missense"] * n,
            "editDistance": edit_distances if edit_distances is not None else [0] * n,
            "Cells_AreaShape_Area": list(range(n, 0, -1)),
            "nuclei_intensity": [1.0] * n,  # dropped: lowercase first char
            "someExtra": ["x"] * n,  # dropped: no underscore
        }
    )


def _make_filtered_df(
    barcodes: list[str],
    aa_changes: list[str],
    edit_distances: list[int] | None = None,
    cfg=None,
) -> pl.DataFrame:
    """Build a cell DataFrame already passed through filter_columns."""
    raw = _make_cell_df(barcodes, aa_changes, edit_distances)
    return filter_columns(raw.lazy(), cfg).collect()


# ---------------------------------------------------------------------------
# get_barcode_counts
# ---------------------------------------------------------------------------


class TestGetBarcodeCounts:
    def test_passing_barcode_has_count(self, cfg):
        """Barcode with count >= bc_threshold gets a non-null barcode_ok."""
        df = _make_filtered_df(["bc1"] * 3 + ["bc2"] * 2, ["V1"] * 5, cfg=cfg)
        result = get_barcode_counts(df.lazy(), cfg).collect()

        bc1_row = result.filter(pl.col("meta_barcode") == "bc1")
        assert bc1_row["barcode_ok"][0] == 3

    def test_failing_barcode_is_null(self, cfg):
        """Barcode with count < bc_threshold gets a null barcode_ok."""
        df = _make_filtered_df(["bc1"] * 3 + ["bc2"] * 2, ["V1"] * 5, cfg=cfg)
        result = get_barcode_counts(df.lazy(), cfg).collect()

        bc2_row = result.filter(pl.col("meta_barcode") == "bc2")
        assert bc2_row["barcode_ok"][0] is None

    def test_one_row_per_barcode(self, cfg):
        """Output contains exactly one row per unique barcode."""
        df = _make_filtered_df(["bc1"] * 3 + ["bc2"] * 3, ["V1"] * 6, cfg=cfg)
        result = get_barcode_counts(df.lazy(), cfg).collect()

        assert result.shape[0] == 2

    def test_counts_are_correct(self, cfg):
        """The count column reflects the true number of cells per barcode."""
        df = _make_filtered_df(["bc1"] * 5 + ["bc2"] * 3, ["V1"] * 8, cfg=cfg)
        result = get_barcode_counts(df.lazy(), cfg).collect().sort("meta_barcode")

        assert result["count"].to_list() == [5, 3]


# ---------------------------------------------------------------------------
# get_barcodes_per_variant
# ---------------------------------------------------------------------------


class TestGetBarcodesPerVariant:
    def test_passing_variant_has_count(self, cfg):
        """Variant with barcode count >= variant_bc_threshold is flagged."""
        # V1 has two distinct barcodes; V2 has one
        df = _make_filtered_df(["bc1", "bc2", "bc3"], ["V1", "V1", "V2"], cfg=cfg)
        result = get_barcodes_per_variant(df.lazy(), cfg).collect()

        v1_row = result.filter(pl.col("meta_aa_changes") == "V1")
        assert v1_row["variant_barcode_count_ok"][0] == 2

    def test_failing_variant_is_null(self, cfg):
        """Variant with barcode count < variant_bc_threshold gets null."""
        df = _make_filtered_df(["bc1", "bc2", "bc3"], ["V1", "V1", "V2"], cfg=cfg)
        result = get_barcodes_per_variant(df.lazy(), cfg).collect()

        v2_row = result.filter(pl.col("meta_aa_changes") == "V2")
        assert v2_row["variant_barcode_count_ok"][0] is None

    def test_one_row_per_variant(self, cfg):
        """Output contains exactly one row per unique variant."""
        df = _make_filtered_df(
            ["bc1", "bc2", "bc3", "bc4"], ["V1", "V1", "V2", "V2"], cfg=cfg
        )
        result = get_barcodes_per_variant(df.lazy(), cfg).collect()

        assert result.shape[0] == 2

    def test_tagged_and_untagged_rows_pool_together(self, cfg):
        """A tagged variant (e.g. 'V1:downsampled-half') pools with its
        untagged base 'V1' under a single meta_aa_changes group, since
        tag-stripping happens in filter_columns before this function runs."""
        df = _make_filtered_df(["bc1", "bc2"], ["V1", "V1:downsampled-half"], cfg=cfg)
        result = get_barcodes_per_variant(df.lazy(), cfg).collect()

        assert result.shape[0] == 1
        assert result["meta_aa_changes"][0] == "V1"
        assert result["barcode_count"][0] == 2


# ---------------------------------------------------------------------------
# add_qc_queries
# ---------------------------------------------------------------------------


class TestAddQcQueries:
    @pytest.fixture
    def cell_df(self, cfg):
        """
        bc1: 3 cells for V1, all edit_distance=0 → passes all filters
        bc2: 3 cells for V1, all edit_distance=0 → passes all filters
        bc3: 3 cells for V2, all edit_distance=0 → passes barcode filter,
             but V2 has only 1 passing barcode so fails variant filter
        bc4: 1 cell for V1, edit_distance=0 → fails bc_threshold
        bc5: 1 cell for V1, edit_distance=2 → fails edit distance filter

        After edit filter (<=1): bc5 removed
        After barcode filter (>=3): bc1, bc2, bc3 pass; bc4 removed
        After variant filter (>=2): V1 has bc1+bc2 → passes; V2 has bc3 → fails
        Final: 6 rows (bc1×3 + bc2×3), all V1
        """
        return _make_filtered_df(
            ["bc1"] * 3 + ["bc2"] * 3 + ["bc3"] * 3 + ["bc4"] + ["bc5"],
            ["V1"] * 3 + ["V1"] * 3 + ["V2"] * 3 + ["V1"] + ["V1"],
            [0] * 3 + [0] * 3 + [0] * 3 + [0] + [2],
            cfg=cfg,
        )

    def test_edit_distance_filter(self, cell_df, cfg):
        """Rows with editDistance > threshold are removed."""
        filtered, _, _ = add_qc_queries(cell_df.lazy(), cfg)
        result = filtered.collect()
        assert result["meta_edit_distance"].max() <= cfg.edit_distance_threshold

    def test_barcode_filter_removes_rare_barcodes(self, cell_df, cfg):
        """Barcodes below bc_threshold do not appear in output."""
        filtered, _, _ = add_qc_queries(cell_df.lazy(), cfg)
        result = filtered.collect()
        assert "bc4" not in result["meta_barcode"].to_list()
        assert "bc5" not in result["meta_barcode"].to_list()

    def test_variant_filter_removes_rare_variants(self, cell_df, cfg):
        """Variants below variant_bc_threshold do not appear in output."""
        filtered, _, _ = add_qc_queries(cell_df.lazy(), cfg)
        result = filtered.collect()
        assert "V2" not in result["meta_aa_changes"].to_list()

    def test_passing_cells_are_retained(self, cell_df, cfg):
        """Cells from passing barcodes and variants are kept."""
        filtered, _, _ = add_qc_queries(cell_df.lazy(), cfg)
        result = filtered.collect()
        assert set(result["meta_barcode"].to_list()) == {"bc1", "bc2"}
        assert result.shape[0] == 6  # 3 cells each from bc1 and bc2

    def test_barcode_counts_frame_shape(self, cell_df, cfg):
        """barcode_counts has one row per barcode surviving the edit filter."""
        _, barcode_counts, _ = add_qc_queries(cell_df.lazy(), cfg)
        result = barcode_counts.collect()
        # barcode_counts is built after the edit distance filter, so barcodes
        # that fail that filter (e.g. bc5 with edit_distance=2) are excluded.
        n_post_edit_filter = cell_df.filter(
            pl.col("meta_edit_distance") <= cfg.edit_distance_threshold
        )["meta_barcode"].n_unique()
        assert result.shape[0] == n_post_edit_filter

    def test_variants_per_barcode_frame_shape(self, cell_df, cfg):
        """variants_per_barcode frame has one row per unique variant."""
        _, _, variants_per_barcode = add_qc_queries(cell_df.lazy(), cfg)
        # Only barcodes passing the barcode filter feed into this frame,
        # so the variant count may be less than in the raw data.
        result = variants_per_barcode.collect()
        assert result.shape[0] >= 1


# ---------------------------------------------------------------------------
# filter_columns
# ---------------------------------------------------------------------------


class TestFilterColumns:
    def test_meta_columns_are_created(self, cfg):
        """aaChanges, variantType, editDistance, barcode are aliased to meta_."""
        df = _make_cell_df(["bc1"] * 2, ["V1"] * 2)
        result = filter_columns(df.lazy(), cfg).collect()

        for col in (
            "meta_aa_changes",
            "meta_edit_distance",
            "meta_barcode",
        ):
            assert col in result.columns

    def test_cell_profiler_columns_retained(self, cfg):
        """Columns starting with uppercase and containing '_' are kept."""
        df = _make_cell_df(["bc1"] * 2, ["V1"] * 2)
        result = filter_columns(df.lazy(), cfg).collect()

        assert "Cells_AreaShape_Area" in result.columns

    def test_non_cell_profiler_columns_dropped(self, cfg):
        """Columns that are not CellProfiler features and not meta_ are dropped."""
        df = _make_cell_df(["bc1"] * 2, ["V1"] * 2)
        result = filter_columns(df.lazy(), cfg).collect()

        assert "nuclei_intensity" not in result.columns
        assert "someExtra" not in result.columns

    def test_tagged_variant_is_split(self, cfg):
        """A ':'-delimited tag is split off into meta_variant_tag."""
        df = _make_cell_df(["bc1"], ["M1K:downsampled-half"])
        result = filter_columns(df.lazy(), cfg).collect()

        assert result["meta_aa_changes"][0] == "M1K"
        assert result["meta_variant_tag"][0] == "downsampled-half"

    def test_untagged_variant_has_null_tag(self, cfg):
        """A variant with no ':' gets a null meta_variant_tag."""
        df = _make_cell_df(["bc1"], ["M1K"])
        result = filter_columns(df.lazy(), cfg).collect()

        assert result["meta_aa_changes"][0] == "M1K"
        assert result["meta_variant_tag"][0] is None


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_csv_adds_metadata_columns(self, tmp_path):
        """Reading a CSV adds meta_source_file and meta_origin_file_idx."""
        csv_file = tmp_path / "cells.csv"
        pl.DataFrame({"a": [1, 2, 3]}).write_csv(csv_file)

        result = read_file(csv_file).collect()

        assert "meta_source_file" in result.columns
        assert "meta_source_file_idx" in result.columns

    def test_parquet_adds_metadata_columns(self, tmp_path):
        """Reading a Parquet file adds meta_source_file and meta_source_file_idx."""
        pq_file = tmp_path / "cells.parquet"
        pl.DataFrame({"a": [1, 2, 3]}).write_parquet(pq_file)

        result = read_file(pq_file).collect()

        assert "meta_source_file" in result.columns
        assert "meta_source_file_idx" in result.columns

    def test_meta_source_file_value(self, tmp_path):
        """meta_source_file contains the path of the file that was read."""
        pq_file = tmp_path / "cells.parquet"
        pl.DataFrame({"a": [1]}).write_parquet(pq_file)

        result = read_file(pq_file).collect()

        assert result["meta_source_file"][0] == str(pq_file)

    def test_meta_source_file_idx_is_sequential(self, tmp_path):
        """meta_source_file_idx is a zero-based row index."""
        pq_file = tmp_path / "cells.parquet"
        pl.DataFrame({"a": [10, 20, 30]}).write_parquet(pq_file)

        result = read_file(pq_file).collect()

        assert result["meta_source_file_idx"].to_list() == [0, 1, 2]


# ---------------------------------------------------------------------------
# combine_cell_files
# ---------------------------------------------------------------------------


class TestCombineCellFiles:
    def test_row_count_is_sum_of_inputs(self, tmp_path):
        """Combined frame has as many rows as all input files together."""
        f1 = tmp_path / "a.parquet"
        f2 = tmp_path / "b.parquet"
        pl.DataFrame({"x": [1, 2]}).write_parquet(f1)
        pl.DataFrame({"x": [3, 4, 5]}).write_parquet(f2)

        result = combine_cell_files([f1, f2]).collect()

        assert result.shape[0] == 5

    def test_source_files_are_tracked(self, tmp_path):
        """Each row records which source file it came from."""
        f1 = tmp_path / "a.parquet"
        f2 = tmp_path / "b.parquet"
        pl.DataFrame({"x": [1]}).write_parquet(f1)
        pl.DataFrame({"x": [2]}).write_parquet(f2)

        result = combine_cell_files([f1, f2]).collect()
        sources = set(result["meta_source_file"].to_list())

        assert sources == {str(f1), str(f2)}


# ---------------------------------------------------------------------------
# add_downsampled_pseudo_variants
# ---------------------------------------------------------------------------


def _make_downsample_df(aa_changes: list[str], cfg) -> pl.DataFrame:
    """Build a post-filter_columns-shaped DataFrame with meta_source_file/
    meta_source_file_idx identity columns, as add_qc_queries's output would
    have (via read_file), for testing add_downsampled_pseudo_variants."""
    n = len(aa_changes)
    df = _make_filtered_df([f"bc{i}" for i in range(n)], aa_changes, cfg=cfg)
    return df.with_columns(
        pl.lit("f1").alias("meta_source_file"),
        pl.Series("meta_source_file_idx", list(range(n))),
    )


class TestAddDownsampledPseudoVariants:
    def test_full_fraction_keeps_all_and_tags(self, cfg):
        df = _make_downsample_df(["A1A"] * 10, cfg)
        result = add_downsampled_pseudo_variants(
            df.lazy(),
            cfg,
            downsample_classes=("Synonymous",),
            downsample_fraction=1.0,
            seed=0,
        ).collect()

        assert result.shape[0] == 10
        assert (result["meta_variant_tag"] == DOWNSAMPLE_TAG).all()
        # meta_aa_changes itself is unchanged — the tag column carries the mark
        assert (result["meta_aa_changes"] == "A1A").all()

    def test_zero_fraction_keeps_none(self, cfg):
        df = _make_downsample_df(["A1A"] * 10, cfg)
        result = add_downsampled_pseudo_variants(
            df.lazy(),
            cfg,
            downsample_classes=("Synonymous",),
            downsample_fraction=0.0,
            seed=0,
        ).collect()

        assert result.shape[0] == 0

    def test_partial_fraction_is_deterministic_across_calls(self, cfg):
        df = _make_downsample_df(["A1A"] * 10, cfg)
        result1 = add_downsampled_pseudo_variants(
            df.lazy(),
            cfg,
            downsample_classes=("Synonymous",),
            downsample_fraction=0.5,
            seed=7,
        ).collect()
        result2 = add_downsampled_pseudo_variants(
            df.lazy(),
            cfg,
            downsample_classes=("Synonymous",),
            downsample_fraction=0.5,
            seed=7,
        ).collect()

        assert (
            result1["meta_source_file_idx"].to_list()
            == result2["meta_source_file_idx"].to_list()
        )
        assert result1.shape[0] == 5

    def test_different_seed_can_change_selection(self, cfg):
        df = _make_downsample_df(["A1A"] * 10, cfg)
        result_a = add_downsampled_pseudo_variants(
            df.lazy(),
            cfg,
            downsample_classes=("Synonymous",),
            downsample_fraction=0.5,
            seed=0,
        ).collect()
        result_b = add_downsampled_pseudo_variants(
            df.lazy(),
            cfg,
            downsample_classes=("Synonymous",),
            downsample_fraction=0.5,
            seed=1,
        ).collect()

        # Same count either way, but not guaranteed to be the identical subset.
        assert result_a.shape[0] == result_b.shape[0] == 5

    def test_ineligible_class_excluded(self, cfg):
        df = _make_downsample_df(["WT"] * 10, cfg)
        result = add_downsampled_pseudo_variants(
            df.lazy(),
            cfg,
            downsample_classes=("Synonymous",),
            downsample_fraction=1.0,
            seed=0,
        ).collect()

        assert result.shape[0] == 0


# ---------------------------------------------------------------------------
# main() — downsample_fraction
# ---------------------------------------------------------------------------


def _write_cells(path, barcodes, aa_changes, edit_distances=None):
    n = len(barcodes)
    pl.DataFrame(
        {
            "upBarcode": barcodes,
            "aaChanges": aa_changes,
            "editDistance": edit_distances if edit_distances is not None else [0] * n,
            "Cells_AreaShape_Area": list(range(n)),
        }
    ).write_parquet(path)


def _make_qc_cfg(tmp_path, cell_files, output_root=None, **overrides):
    files = cell_files if isinstance(cell_files, list) else [cell_files]
    return OmegaConf.structured(
        QcFilterConfig(
            output_dir=str(tmp_path / "out"),
            output_root=output_root,
            cell_files=[str(p) for p in files],
            **overrides,
        )
    )


def test_main_downsample_fraction_none_matches_no_downsampling(tmp_path):
    source = tmp_path / "cells.parquet"
    _write_cells(source, [f"bc{i}" for i in range(10)], ["A1A"] * 10)

    qc_cfg = _make_qc_cfg(
        tmp_path,
        source,
        bc_threshold=1,
        variant_bc_threshold=1,
        edit_distance_threshold=1,
    )

    with patch("fisseq_data_pipeline.qcfilter.setup_logging"):
        m.main.__wrapped__(qc_cfg)

    result = pl.read_parquet(tmp_path / "out" / "filtered_cells.parquet")
    assert result.shape[0] == 10
    assert result["meta_variant_tag"].is_null().all()


def test_main_downsample_pseudo_rows_only_from_qc_survivors(tmp_path):
    source = tmp_path / "cells.parquet"
    barcodes = [f"bc{i}" for i in range(10)] + ["bc_fail"]
    aa_changes = ["A1A"] * 11
    # bc_fail has editDistance=5, above threshold=1, so it must be dropped by
    # add_qc_queries *before* any downsampling runs on it.
    edit_distances = [0] * 10 + [5]
    _write_cells(source, barcodes, aa_changes, edit_distances)

    qc_cfg = _make_qc_cfg(
        tmp_path,
        source,
        bc_threshold=1,
        variant_bc_threshold=1,
        edit_distance_threshold=1,
        downsample_fraction=1.0,
        downsample_seed=0,
    )

    with patch("fisseq_data_pipeline.qcfilter.setup_logging"):
        m.main.__wrapped__(qc_cfg)

    result = pl.read_parquet(tmp_path / "out" / "filtered_cells.parquet")
    assert "bc_fail" not in result["meta_barcode"].to_list()
    # 10 QC survivors, downsampled at fraction=1.0 -> 10 originals + 10 pseudo
    assert result.shape[0] == 20
    assert (result["meta_variant_tag"] == DOWNSAMPLE_TAG).sum() == 10


def test_main_downsample_reproducible_with_fixed_seed(tmp_path):
    source = tmp_path / "cells.parquet"
    _write_cells(source, [f"bc{i}" for i in range(10)], ["A1A"] * 10)

    qc_cfg_a = _make_qc_cfg(
        tmp_path / "run_a",
        source,
        bc_threshold=1,
        variant_bc_threshold=1,
        edit_distance_threshold=1,
        downsample_fraction=0.5,
        downsample_seed=7,
    )
    qc_cfg_b = _make_qc_cfg(
        tmp_path / "run_b",
        source,
        bc_threshold=1,
        variant_bc_threshold=1,
        edit_distance_threshold=1,
        downsample_fraction=0.5,
        downsample_seed=7,
    )

    with patch("fisseq_data_pipeline.qcfilter.setup_logging"):
        m.main.__wrapped__(qc_cfg_a)
        m.main.__wrapped__(qc_cfg_b)

    result_a = pl.read_parquet(tmp_path / "run_a" / "out" / "filtered_cells.parquet")
    result_b = pl.read_parquet(tmp_path / "run_b" / "out" / "filtered_cells.parquet")

    sort_key = ["meta_source_file", "meta_source_file_idx", "meta_variant_tag"]
    assert_frame_equal(result_a.sort(sort_key), result_b.sort(sort_key))


def test_main_downsample_barcode_counts_and_variants_per_barcode_exclude_pseudo_rows(
    tmp_path,
):
    source = tmp_path / "cells.parquet"
    _write_cells(source, [f"bc{i}" for i in range(10)], ["A1A"] * 10)

    qc_cfg_with = _make_qc_cfg(
        tmp_path / "run_with",
        source,
        bc_threshold=1,
        variant_bc_threshold=1,
        edit_distance_threshold=1,
        downsample_fraction=1.0,
        downsample_seed=0,
    )
    qc_cfg_without = _make_qc_cfg(
        tmp_path / "run_without",
        source,
        bc_threshold=1,
        variant_bc_threshold=1,
        edit_distance_threshold=1,
    )

    with patch("fisseq_data_pipeline.qcfilter.setup_logging"):
        m.main.__wrapped__(qc_cfg_with)
        m.main.__wrapped__(qc_cfg_without)

    for name in ("barcode_counts", "variants_per_barcode"):
        result_with = pl.read_parquet(tmp_path / "run_with" / "out" / f"{name}.parquet")
        result_without = pl.read_parquet(
            tmp_path / "run_without" / "out" / f"{name}.parquet"
        )
        assert_frame_equal(
            result_with.sort(result_with.columns),
            result_without.sort(result_without.columns),
        )
