from __future__ import annotations

import pathlib
from unittest.mock import patch

import polars as pl
import pytest
import yaml
from omegaconf import OmegaConf

import fisseq_data_pipeline.input as m
from fisseq_data_pipeline.input import (
    InputStageConfig,
    load_and_concat,
    load_and_tag,
    load_feature_patterns,
    select_output_columns,
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
# select_output_columns
# ---------------------------------------------------------------------------


def test_select_output_columns_keeps_identity_columns_unprefixed():
    lf = pl.DataFrame(
        {
            "upBarcode": ["bc1"],
            "aaChanges": ["M1K"],
            "editDistance": [0],
            "origin_file": ["f1"],
            "origin_row_idx": [0],
            "Cells_AreaShape_Area": [1.0],
        }
    ).lazy()

    result = select_output_columns(lf).collect()

    for col in ("upBarcode", "aaChanges", "editDistance"):
        assert col in result.columns


def _tagged_lf():
    return pl.DataFrame(
        {
            "upBarcode": ["bc1"],
            "aaChanges": ["M1K"],
            "editDistance": [0],
            "origin_file": ["f1"],
            "origin_row_idx": [0],
            "Cells_AreaShape_Area": [1.0],
            "Nuclei_Texture_Contrast": [2.0],
        }
    ).lazy()


def test_select_output_columns_no_duplicate_or_lost_columns():
    result = select_output_columns(_tagged_lf()).collect()

    assert len(result.columns) == len(set(result.columns))
    assert set(result.columns) == {
        "upBarcode",
        "aaChanges",
        "editDistance",
        "meta_origin_file",
        "meta_origin_row_idx",
        "Cells_AreaShape_Area",
        "Nuclei_Texture_Contrast",
    }
    for unprefixed in ("origin_file", "origin_row_idx"):
        assert unprefixed not in result.columns


def test_select_output_columns_allowlist_blocklist_ignore_metadata_columns():
    result = select_output_columns(
        _tagged_lf(), feature_allowlist=["Cells_*"]
    ).collect()

    assert "Cells_AreaShape_Area" in result.columns
    assert "Nuclei_Texture_Contrast" not in result.columns
    # metadata columns still present, correctly prefixed, unaffected by allowlist
    assert "meta_origin_file" in result.columns
    assert "meta_origin_row_idx" in result.columns


def test_select_output_columns_blocked_feature_not_recovered_via_meta_catchall():
    result = select_output_columns(
        _tagged_lf(), feature_blocklist=["Nuclei_Texture_*"]
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
# main()
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


def _write_config(path, source_path):
    sources = source_path if isinstance(source_path, list) else [source_path]
    config = {"input_paths": [str(p) for p in sources]}
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


def test_main_keeps_all_variants(tmp_path):
    source = tmp_path / "source.parquet"
    _write_source(source, ["M1K", "M2L", "M3Q"])
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, source)

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


def test_main_merges_multiple_sources(tmp_path):
    csv_path = tmp_path / "sources" / "a.csv"
    parquet_path = tmp_path / "sources" / "b.parquet"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _write_source(csv_path, ["WT", "A1A"])
    _write_source(parquet_path, ["M1K", "M2L"])
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, [csv_path, parquet_path])

    with patch("fisseq_data_pipeline.input.setup_logging"):
        m.main.__wrapped__(_make_cfg(tmp_path, config_path))

    result = pl.read_parquet(tmp_path / "out" / "output.parquet")
    assert result.shape[0] == 4
    assert set(result["aaChanges"].to_list()) == {"WT", "A1A", "M1K", "M2L"}


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
