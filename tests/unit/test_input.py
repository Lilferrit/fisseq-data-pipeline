from __future__ import annotations

import os
import pathlib
from unittest.mock import patch

import polars as pl
import pytest
import yaml
from omegaconf import OmegaConf
from polars.testing import assert_frame_equal

import fisseq_data_pipeline.input as m
from fisseq_data_pipeline.input import (
    InputStageConfig,
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


def _classified_lf():
    return pl.DataFrame(
        {
            "upBarcode": ["bc1"],
            "aaChanges": ["M1K"],
            "editDistance": [0],
            "variant_base": ["M1K"],
            "variant_class": ["Single Missense"],
            "origin_file": ["f1"],
            "origin_row_idx": [0],
            "Cells_AreaShape_Area": [1.0],
            "Nuclei_Texture_Contrast": [2.0],
        }
    ).lazy()


def test_select_output_columns_no_duplicate_or_lost_columns():
    result = select_output_columns(_classified_lf()).collect()

    assert len(result.columns) == len(set(result.columns))
    assert set(result.columns) == {
        "upBarcode",
        "aaChanges",
        "editDistance",
        "meta_variant_base",
        "meta_variant_class",
        "meta_origin_file",
        "meta_origin_row_idx",
        "Cells_AreaShape_Area",
        "Nuclei_Texture_Contrast",
    }
    for unprefixed in (
        "variant_base",
        "variant_class",
        "origin_file",
        "origin_row_idx",
    ):
        assert unprefixed not in result.columns


def test_select_output_columns_allowlist_blocklist_ignore_metadata_columns():
    result = select_output_columns(
        _classified_lf(), feature_allowlist=["Cells_*"]
    ).collect()

    assert "Cells_AreaShape_Area" in result.columns
    assert "Nuclei_Texture_Contrast" not in result.columns
    # metadata columns still present, correctly prefixed, unaffected by allowlist
    assert "meta_variant_base" in result.columns
    assert "meta_variant_class" in result.columns
    assert "meta_origin_file" in result.columns
    assert "meta_origin_row_idx" in result.columns


def test_select_output_columns_blocked_feature_not_recovered_via_meta_catchall():
    result = select_output_columns(
        _classified_lf(), feature_blocklist=["Nuclei_Texture_*"]
    ).collect()

    assert "Nuclei_Texture_Contrast" not in result.columns
    assert "meta_Nuclei_Texture_Contrast" not in result.columns


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
    df = pl.DataFrame(rows)
    if pathlib.Path(path).suffix.lower() == ".csv":
        df.write_csv(path)
    else:
        df.write_parquet(path)


def _write_config(
    path,
    source_path,
    top_n_missense=None,
    convert_first=None,
    temp_dir=None,
):
    sources = source_path if isinstance(source_path, list) else [source_path]
    config = {"input_paths": [str(p) for p in sources]}
    if top_n_missense is not None:
        config["top_n_missense"] = top_n_missense
    if convert_first is not None:
        config["convert_first"] = convert_first
    if temp_dir is not None:
        config["temp_dir"] = str(temp_dir)
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


# ---------------------------------------------------------------------------
# main() — convert_first / temp_dir
# ---------------------------------------------------------------------------

_STABLE_SORT_KEY = ["meta_origin_file", "meta_origin_row_idx", "aaChanges"]


def _write_mixed_sources(tmp_path):
    """One CSV and one Parquet source, with a mix of variant classes/counts."""
    csv_path = tmp_path / "sources" / "a.csv"
    parquet_path = tmp_path / "sources" / "b.parquet"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _write_source(csv_path, ["WT", "A1A", "A1A", "M1K", "M1K", "M1K", "M2L", "M2L"])
    _write_source(
        parquet_path, ["A1A", "A1A", "M1K", "M2L", "M2L", "M3Q", "M3Q", "M3Q"]
    )
    return [csv_path, parquet_path]


def test_main_convert_first_matches_non_converted_output(tmp_path):
    sources = _write_mixed_sources(tmp_path)

    temp_dir = tmp_path / "convtmp"
    temp_dir.mkdir()
    config_converted = tmp_path / "config_converted.yaml"
    _write_config(
        config_converted,
        sources,
        top_n_missense=2,
        convert_first=True,
        temp_dir=temp_dir,
    )
    config_plain = tmp_path / "config_plain.yaml"
    _write_config(config_plain, sources, top_n_missense=2)

    with patch("fisseq_data_pipeline.input.setup_logging"):
        m.main.__wrapped__(_make_cfg(tmp_path / "run_a", config_converted))
        m.main.__wrapped__(_make_cfg(tmp_path / "run_b", config_plain))

    result_a = pl.read_parquet(tmp_path / "run_a" / "out" / "output.parquet")
    result_b = pl.read_parquet(tmp_path / "run_b" / "out" / "output.parquet")

    assert_frame_equal(result_a.sort(_STABLE_SORT_KEY), result_b.sort(_STABLE_SORT_KEY))


def test_main_convert_first_skips_merge_when_no_extra_pass_needed(tmp_path):
    source = tmp_path / "source.parquet"
    _write_source(source, ["M1K", "M2L", "A1A", "WT"] * 3)

    config_true = tmp_path / "config_true.yaml"
    _write_config(config_true, source, convert_first=True)
    config_false = tmp_path / "config_false.yaml"
    _write_config(config_false, source)

    with patch("fisseq_data_pipeline.input.setup_logging"):
        with patch.object(m, "convert_and_merge_inputs") as mock_merge:
            m.main.__wrapped__(_make_cfg(tmp_path / "run_a", config_true))
        m.main.__wrapped__(_make_cfg(tmp_path / "run_b", config_false))

    mock_merge.assert_not_called()

    result_a = pl.read_parquet(tmp_path / "run_a" / "out" / "output.parquet")
    result_b = pl.read_parquet(tmp_path / "run_b" / "out" / "output.parquet")
    assert_frame_equal(result_a.sort(_STABLE_SORT_KEY), result_b.sort(_STABLE_SORT_KEY))


def test_main_convert_first_triggered_by_top_n_missense_alone(tmp_path):
    sources = _write_mixed_sources(tmp_path)

    temp_dir = tmp_path / "convtmp"
    temp_dir.mkdir()
    config_converted = tmp_path / "config_converted.yaml"
    _write_config(
        config_converted,
        sources,
        top_n_missense=2,
        convert_first=True,
        temp_dir=temp_dir,
    )
    config_plain = tmp_path / "config_plain.yaml"
    _write_config(config_plain, sources, top_n_missense=2)

    with patch("fisseq_data_pipeline.input.setup_logging"):
        with patch.object(
            m, "convert_and_merge_inputs", wraps=m.convert_and_merge_inputs
        ) as spy:
            m.main.__wrapped__(_make_cfg(tmp_path / "run_a", config_converted))
        m.main.__wrapped__(_make_cfg(tmp_path / "run_b", config_plain))

    spy.assert_called_once()

    result_a = pl.read_parquet(tmp_path / "run_a" / "out" / "output.parquet")
    result_b = pl.read_parquet(tmp_path / "run_b" / "out" / "output.parquet")
    assert_frame_equal(result_a.sort(_STABLE_SORT_KEY), result_b.sort(_STABLE_SORT_KEY))


def test_main_convert_first_false_ignores_temp_dir_and_tmpdir(tmp_path):
    source = tmp_path / "source.parquet"
    _write_source(source, ["M1K", "M2L", "A1A", "WT"] * 3)

    config_temp_dir = tmp_path / "config_temp_dir"
    config_temp_dir.mkdir()
    env_temp_dir = tmp_path / "env_temp_dir"
    env_temp_dir.mkdir()

    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        source,
        top_n_missense=2,
        convert_first=False,
        temp_dir=config_temp_dir,
    )

    with patch.dict(os.environ, {"TMPDIR": str(env_temp_dir)}):
        with patch("fisseq_data_pipeline.input.setup_logging"):
            with patch.object(m, "convert_and_merge_inputs") as mock_merge:
                m.main.__wrapped__(_make_cfg(tmp_path, config_path))

    mock_merge.assert_not_called()
    assert list(config_temp_dir.iterdir()) == []
    assert list(env_temp_dir.iterdir()) == []


def test_main_convert_first_cleans_up_temp_file_after_success(tmp_path):
    source = tmp_path / "source.parquet"
    _write_source(source, ["M1K", "M2L", "A1A", "WT"] * 3)

    temp_dir = tmp_path / "convtmp"
    temp_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        source,
        top_n_missense=2,
        convert_first=True,
        temp_dir=temp_dir,
    )

    with patch("fisseq_data_pipeline.input.setup_logging"):
        m.main.__wrapped__(_make_cfg(tmp_path, config_path))

    assert not (temp_dir / m.CONVERTED_INPUT_FILENAME).exists()


def test_main_convert_first_cleans_up_temp_file_on_failure(tmp_path):
    source = tmp_path / "source.parquet"
    _write_source(source, ["M1K", "M2L", "A1A", "WT"] * 3)

    temp_dir = tmp_path / "convtmp"
    temp_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        source,
        top_n_missense=2,
        convert_first=True,
        temp_dir=temp_dir,
    )

    real_sink_parquet = pl.LazyFrame.sink_parquet
    call_count = {"n": 0}

    def fake_sink_parquet(self, path, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # let the merge step's sink_parquet call succeed normally
            return real_sink_parquet(self, path, *args, **kwargs)
        raise RuntimeError("boom")

    with patch("fisseq_data_pipeline.input.setup_logging"):
        with patch.object(pl.LazyFrame, "sink_parquet", fake_sink_parquet):
            with pytest.raises(RuntimeError, match="boom"):
                m.main.__wrapped__(_make_cfg(tmp_path, config_path))

    assert not (temp_dir / m.CONVERTED_INPUT_FILENAME).exists()


def test_main_convert_first_temp_dir_falls_back_to_gettempdir(tmp_path):
    source = tmp_path / "source.parquet"
    _write_source(source, ["M1K", "M2L", "A1A", "WT"] * 3)

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, source, top_n_missense=2, convert_first=True)

    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()

    env_without_tmpdir = dict(os.environ)
    env_without_tmpdir.pop("TMPDIR", None)

    with patch.dict(os.environ, env_without_tmpdir, clear=True):
        with patch("fisseq_data_pipeline.input.setup_logging"):
            with patch.object(m.tempfile, "gettempdir", return_value=str(fallback_dir)):
                with patch.object(
                    m, "convert_and_merge_inputs", wraps=m.convert_and_merge_inputs
                ) as spy:
                    m.main.__wrapped__(_make_cfg(tmp_path, config_path))

    spy.assert_called_once_with([str(source)], str(fallback_dir))
