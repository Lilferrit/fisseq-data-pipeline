from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest
import yaml
from omegaconf import OmegaConf

import fisseq_data_pipeline.input as m
from fisseq_data_pipeline.input import (
    InputStageConfig,
    add_downsampled_pseudo_variants,
    classify_variants,
    filter_variants,
    load_and_concat,
    load_and_tag,
    load_feature_patterns,
    select_output_columns,
    select_top_missense,
)

# ---------------------------------------------------------------------------
# load_and_tag / load_and_concat
# ---------------------------------------------------------------------------


def test_load_and_tag_parquet_adds_origin_columns(tmp_path):
    p = tmp_path / "cells.parquet"
    pl.DataFrame({"aaChanges": ["WT", "M1K"]}).write_parquet(p)

    result = load_and_tag(str(p)).collect()

    assert result["origin_file"].to_list() == [str(p), str(p)]
    assert result["origin_row_idx"].to_list() == [0, 1]


def test_load_and_tag_rejects_unsupported_extension(tmp_path):
    p = tmp_path / "cells.txt"
    p.write_text("aaChanges\nWT\n")

    with pytest.raises(ValueError):
        load_and_tag(str(p)).collect()


def test_load_and_concat_sums_row_counts(tmp_path):
    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    pl.DataFrame({"aaChanges": ["WT", "M1K"]}).write_parquet(p1)
    pl.DataFrame({"aaChanges": ["A1A"]}).write_parquet(p2)

    result = load_and_concat([str(p1), str(p2)]).collect()

    assert result.shape[0] == 3


# ---------------------------------------------------------------------------
# classify_variants
# ---------------------------------------------------------------------------


class TestClassifyVariants:
    def test_untagged_variant_base_equals_aa_changes(self):
        lf = pl.DataFrame({"aaChanges": ["M1K"]}).lazy()
        result = classify_variants(lf).collect()

        assert result["variant_base"][0] == "M1K"
        assert result["variant_class"][0] == "Single Missense"

    def test_tagged_variant_classifies_as_its_base(self):
        lf = pl.DataFrame({"aaChanges": ["M1K:downsampled-half"]}).lazy()
        result = classify_variants(lf).collect()

        assert result["variant_base"][0] == "M1K"
        assert result["variant_class"][0] == "Single Missense"

    def test_synonymous_classification(self):
        lf = pl.DataFrame({"aaChanges": ["A1A"]}).lazy()
        result = classify_variants(lf).collect()

        assert result["variant_class"][0] == "Synonymous"

    def test_wt_classification(self):
        lf = pl.DataFrame({"aaChanges": ["WT"]}).lazy()
        result = classify_variants(lf).collect()

        assert result["variant_class"][0] == "WT"


# ---------------------------------------------------------------------------
# select_top_missense
# ---------------------------------------------------------------------------


class TestSelectTopMissense:
    def test_selects_by_descending_count(self):
        lf = pl.DataFrame(
            {
                "variant_base": ["V1"] * 3 + ["V2"] * 5 + ["V3"] * 1,
                "variant_class": ["Single Missense"] * 9,
            }
        ).lazy()

        top = select_top_missense(lf, top_n_missense=2)

        assert top == ["V2", "V1"]

    def test_ignores_non_missense_rows(self):
        lf = pl.DataFrame(
            {
                "variant_base": ["V1", "WT", "A1A"],
                "variant_class": ["Single Missense", "WT", "Synonymous"],
            }
        ).lazy()

        top = select_top_missense(lf, top_n_missense=5)

        assert top == ["V1"]

    def test_tie_break_is_alphabetical(self):
        lf = pl.DataFrame(
            {
                "variant_base": ["V2", "V1"],
                "variant_class": ["Single Missense", "Single Missense"],
            }
        ).lazy()

        top = select_top_missense(lf, top_n_missense=1)

        assert top == ["V1"]


# ---------------------------------------------------------------------------
# filter_variants
# ---------------------------------------------------------------------------


class TestFilterVariants:
    def test_keeps_fixed_classes(self):
        lf = pl.DataFrame(
            {
                "variant_base": ["WT", "A1A", "A1fs", "M1K"],
                "variant_class": ["WT", "Synonymous", "Frameshift", "Single Missense"],
            }
        ).lazy()

        result = filter_variants(lf, top_missense=[]).collect()

        assert set(result["variant_base"].to_list()) == {"WT", "A1A", "A1fs"}

    def test_keeps_selected_missense(self):
        lf = pl.DataFrame(
            {
                "variant_base": ["M1K", "M2L"],
                "variant_class": ["Single Missense", "Single Missense"],
            }
        ).lazy()

        result = filter_variants(lf, top_missense=["M1K"]).collect()

        assert result["variant_base"].to_list() == ["M1K"]

    def test_unselected_missense_dropped(self):
        lf = pl.DataFrame(
            {
                "variant_base": ["M1K", "M2L"],
                "variant_class": ["Single Missense", "Single Missense"],
            }
        ).lazy()

        result = filter_variants(lf, top_missense=[]).collect()

        assert result.shape[0] == 0

    def test_none_top_missense_keeps_all_missense(self):
        lf = pl.DataFrame(
            {
                "variant_base": ["WT", "A1A", "A1fs", "M1K", "M2L"],
                "variant_class": [
                    "WT",
                    "Synonymous",
                    "Frameshift",
                    "Single Missense",
                    "Single Missense",
                ],
            }
        ).lazy()

        result = filter_variants(lf, top_missense=None).collect()

        assert set(result["variant_base"].to_list()) == {
            "WT",
            "A1A",
            "A1fs",
            "M1K",
            "M2L",
        }


# ---------------------------------------------------------------------------
# add_downsampled_pseudo_variants
# ---------------------------------------------------------------------------


class TestAddDownsampledPseudoVariants:
    def _base_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "aaChanges": ["A1A"] * 10,
                "variant_base": ["A1A"] * 10,
                "variant_class": ["Synonymous"] * 10,
                "origin_file": ["f1"] * 10,
                "origin_row_idx": list(range(10)),
            }
        )

    def test_full_fraction_keeps_all_and_tags(self):
        lf = self._base_df().lazy()
        result = add_downsampled_pseudo_variants(
            lf, downsample_classes=("Synonymous",), downsample_fraction=1.0, seed=0
        ).collect()

        assert result.shape[0] == 10
        assert (result["aaChanges"] == "A1A:downsampled-half").all()

    def test_zero_fraction_keeps_none(self):
        lf = self._base_df().lazy()
        result = add_downsampled_pseudo_variants(
            lf, downsample_classes=("Synonymous",), downsample_fraction=0.0, seed=0
        ).collect()

        assert result.shape[0] == 0

    def test_partial_fraction_is_deterministic_across_calls(self):
        lf = self._base_df().lazy()
        result1 = add_downsampled_pseudo_variants(
            lf, downsample_classes=("Synonymous",), downsample_fraction=0.5, seed=7
        ).collect()
        result2 = add_downsampled_pseudo_variants(
            lf, downsample_classes=("Synonymous",), downsample_fraction=0.5, seed=7
        ).collect()

        assert (
            result1["origin_row_idx"].to_list() == result2["origin_row_idx"].to_list()
        )
        assert result1.shape[0] == 5

    def test_different_seed_can_change_selection(self):
        lf = self._base_df().lazy()
        result_a = add_downsampled_pseudo_variants(
            lf, downsample_classes=("Synonymous",), downsample_fraction=0.5, seed=0
        ).collect()
        result_b = add_downsampled_pseudo_variants(
            lf, downsample_classes=("Synonymous",), downsample_fraction=0.5, seed=1
        ).collect()

        # Same count either way, but not guaranteed to be the identical subset.
        assert result_a.shape[0] == result_b.shape[0] == 5

    def test_ineligible_class_excluded(self):
        df = self._base_df()
        df = df.with_columns(pl.lit("WT").alias("variant_class"))
        result = add_downsampled_pseudo_variants(
            df.lazy(),
            downsample_classes=("Synonymous",),
            downsample_fraction=1.0,
            seed=0,
        ).collect()

        assert result.shape[0] == 0


# ---------------------------------------------------------------------------
# select_output_columns
# ---------------------------------------------------------------------------


def test_select_output_columns_keeps_identity_columns_unprefixed():
    lf = pl.DataFrame(
        {
            "upBarcode": ["bc1"],
            "aaChanges": ["M1K"],
            "editDistance": [0],
            "variant_base": ["M1K"],
            "variant_class": ["Single Missense"],
            "origin_file": ["f1"],
            "origin_row_idx": [0],
            "Cells_AreaShape_Area": [1.0],
        }
    ).lazy()

    result = select_output_columns(lf).collect()

    for col in ("upBarcode", "aaChanges", "editDistance"):
        assert col in result.columns


# ---------------------------------------------------------------------------
# load_feature_patterns
# ---------------------------------------------------------------------------


def test_load_feature_patterns_strips_and_skips_blank_lines(tmp_path):
    p = tmp_path / "patterns.txt"
    p.write_text("Cells_AreaShape_*\n\n  Nuclei_Texture_*  \n\n")

    result = load_feature_patterns(str(p))

    assert result == ["Cells_AreaShape_*", "Nuclei_Texture_*"]


# ---------------------------------------------------------------------------
# select_output_columns — allowlist/blocklist
# ---------------------------------------------------------------------------


def _feature_lf():
    return pl.DataFrame(
        {
            "upBarcode": ["bc1"],
            "aaChanges": ["M1K"],
            "editDistance": [0],
            "Cells_AreaShape_Area": [1.0],
            "Cells_Intensity_Mean": [2.0],
            "Nuclei_Texture_Contrast": [3.0],
        }
    ).lazy()


def test_select_output_columns_allowlist_keeps_only_matches():
    result = select_output_columns(
        _feature_lf(), feature_allowlist=["Cells_AreaShape_*"]
    ).collect()

    assert "Cells_AreaShape_Area" in result.columns
    assert "Cells_Intensity_Mean" not in result.columns
    assert "Nuclei_Texture_Contrast" not in result.columns
    for col in ("upBarcode", "aaChanges", "editDistance"):
        assert col in result.columns


def test_select_output_columns_blocklist_drops_matches():
    result = select_output_columns(
        _feature_lf(), feature_blocklist=["Nuclei_Texture_*"]
    ).collect()

    assert "Cells_AreaShape_Area" in result.columns
    assert "Cells_Intensity_Mean" in result.columns
    assert "Nuclei_Texture_Contrast" not in result.columns


def test_select_output_columns_allowlist_then_blocklist():
    result = select_output_columns(
        _feature_lf(),
        feature_allowlist=["Cells_*"],
        feature_blocklist=["Cells_Intensity_*"],
    ).collect()

    assert "Cells_AreaShape_Area" in result.columns
    assert "Cells_Intensity_Mean" not in result.columns
    assert "Nuclei_Texture_Contrast" not in result.columns


# ---------------------------------------------------------------------------
# main() — Task 1: downsampling is opt-in
# ---------------------------------------------------------------------------


def _write_source(path, variants):
    rows = []
    for i, v in enumerate(variants):
        rows.append(
            {
                "upBarcode": f"bc{i}",
                "aaChanges": v,
                "editDistance": 0,
                "Cells_AreaShape_Area": float(i),
            }
        )
    pl.DataFrame(rows).write_parquet(path)


def _write_config(path, source_path, downsample_fraction=None):
    config = {"input_paths": [str(source_path)]}
    if downsample_fraction is not None:
        config["downsample_fraction"] = downsample_fraction
    with open(path, "w") as f:
        yaml.safe_dump(config, f)


def _make_cfg(tmp_path, config_path, output_root=None):
    return OmegaConf.structured(
        InputStageConfig(
            output_dir=str(tmp_path / "out"),
            output_root=output_root,
            log_level="info",
            config_path=str(config_path),
        )
    )


def test_main_downsampling_disabled_by_default(tmp_path):
    source = tmp_path / "source.parquet"
    _write_source(source, ["M1K"] * 10)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, source)  # no downsample_fraction key at all

    with patch("fisseq_data_pipeline.input.setup_logging"):
        m.main.__wrapped__(_make_cfg(tmp_path, config_path))

    result = pl.read_parquet(tmp_path / "out" / "output.parquet")
    assert "downsampled-half" not in ":".join(result["aaChanges"].to_list())
    assert result.shape[0] == 10


def test_main_downsampling_enabled_when_set(tmp_path):
    source = tmp_path / "source.parquet"
    _write_source(source, ["M1K"] * 10)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, source, downsample_fraction=0.5)

    with patch("fisseq_data_pipeline.input.setup_logging"):
        m.main.__wrapped__(_make_cfg(tmp_path, config_path))

    result = pl.read_parquet(tmp_path / "out" / "output.parquet")
    tagged = [
        v for v in result["aaChanges"].to_list() if v.endswith(":downsampled-half")
    ]
    assert len(tagged) == 5
    assert result.shape[0] == 15


def test_main_top_n_missense_null_by_default_keeps_all_missense(tmp_path):
    source = tmp_path / "source.parquet"
    # Three distinct Single Missense variants, one cell each.
    _write_source(source, ["M1K", "M2L", "M3Q"])
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, source)  # no top_n_missense key at all

    with patch("fisseq_data_pipeline.input.setup_logging"):
        m.main.__wrapped__(_make_cfg(tmp_path, config_path))

    result = pl.read_parquet(tmp_path / "out" / "output.parquet")
    assert set(result["aaChanges"].to_list()) == {"M1K", "M2L", "M3Q"}


def test_main_output_root_names_output_file(tmp_path):
    source = tmp_path / "source.parquet"
    _write_source(source, ["M1K"] * 10)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, source)

    with patch("fisseq_data_pipeline.input.setup_logging"):
        m.main.__wrapped__(_make_cfg(tmp_path, config_path, output_root="batch1"))

    assert (tmp_path / "out" / "batch1.output.parquet").exists()


# ---------------------------------------------------------------------------
# main() — feature_blocklist_file
# ---------------------------------------------------------------------------


def test_main_blocked_feature_absent_from_output(tmp_path):
    source = tmp_path / "source.parquet"
    rows = [
        {
            "upBarcode": f"bc{i}",
            "aaChanges": "M1K",
            "editDistance": 0,
            "Cells_AreaShape_Area": float(i),
            "Nuclei_Texture_Contrast": float(i),
        }
        for i in range(10)
    ]
    pl.DataFrame(rows).write_parquet(source)

    blocklist_file = tmp_path / "blocklist.txt"
    blocklist_file.write_text("Nuclei_Texture_*\n")

    config_path = tmp_path / "config.yaml"
    config = {
        "input_paths": [str(source)],
        "feature_blocklist_file": str(blocklist_file),
    }
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    with patch("fisseq_data_pipeline.input.setup_logging"):
        m.main.__wrapped__(_make_cfg(tmp_path, config_path))

    result = pl.read_parquet(tmp_path / "out" / "output.parquet")
    assert "Cells_AreaShape_Area" in result.columns
    assert "Nuclei_Texture_Contrast" not in result.columns
