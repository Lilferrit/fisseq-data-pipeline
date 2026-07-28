from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_data_pipeline.aggregatefeaturetype as m


def _get_row(df: pl.DataFrame, label: str) -> dict:
    return df.filter(pl.col("meta_aa_changes") == label).to_dicts().pop()


def write_agg_input_parquet(tmp_path, *, with_barcode: bool = False) -> None:
    """Write cell-level test parquet with WT controls, synonymous and missense variants."""
    data = {
        "meta_aa_changes": [
            "WT",
            "WT",
            "WT",
            "A1A",
            "A1A",
            "A1A",
            "A2A",
            "A2A",
            "A2A",
            "A1B",
            "A1B",
            "A1B",
        ],
        "meta_is_control": [
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        "f1": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 10.0, 10.0, 10.0],
        "f2": [0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 30.0, 30.0, 30.0],
    }
    if with_barcode:
        data["meta_barcode"] = [
            "bc1",
            "bc2",
            "bc3",
            "bc1",
            "bc2",
            "bc1",
            "bc2",
            "bc3",
            "bc1",
            "bc2",
            "bc3",
            "bc1",
        ]
    pl.DataFrame(data).write_parquet(tmp_path / "input.parquet")


def make_ft_cfg(
    tmp_path,
    *,
    output_root=None,
    aggregator="mean",
    index_file=None,
    downsample_wt=None,
    seed=0,
) -> OmegaConf:
    """Return a DictConfig for FeatureTypeAggregateConfig with test defaults."""
    return OmegaConf.structured(
        m.FeatureTypeAggregateConfig(
            output_dir=str(tmp_path / "out"),
            output_root=output_root,
            input_file=str(tmp_path / "input.parquet"),
            aggregator=aggregator,
            index_file=index_file,
            downsample_wt=downsample_wt,
            seed=seed,
        )
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_output_has_only_label_and_stat_columns(tmp_path) -> None:
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregatefeaturetype.setup_logging"):
        m.main.__wrapped__(make_ft_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert set(result.columns) == {"meta_aa_changes", "f1_mean", "f2_mean"}


def test_main_index_file_none_aggregates_all_rows(tmp_path) -> None:
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregatefeaturetype.setup_logging"):
        m.main.__wrapped__(make_ft_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    # All four groups (WT is control and excluded; A1A, A2A, A1B remain).
    assert set(result["meta_aa_changes"].to_list()) == {"A1A", "A2A", "A1B"}


def test_main_index_file_filters_rows(tmp_path) -> None:
    # Custom dataset (unlike write_agg_input_parquet, whose per-group values
    # are constant, which would make a single-row filter indistinguishable
    # from the full-group aggregate): A1B has three distinct f1 values, so
    # filtering to a subset changes the aggregated mean.
    pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "A1B", "A1B", "A1B"],
            "meta_is_control": [True, True, False, False, False],
            "f1": [0.0, 0.0, 10.0, 20.0, 30.0],
        }
    ).write_parquet(tmp_path / "input.parquet")
    # Row index 2 is the first A1B row (f1=10.0); rows 0-1 are WT.
    idx_path = tmp_path / "half1.parquet"
    pl.DataFrame({"tmp_cell_idx": [2]}).write_parquet(idx_path)

    with patch("fisseq_data_pipeline.aggregatefeaturetype.setup_logging"):
        m.main.__wrapped__(
            make_ft_cfg(
                tmp_path,
                index_file=str(idx_path),
                output_root=str(tmp_path / "filtered"),
            )
        )
    filtered_result = pl.read_parquet(tmp_path / "filtered.input.parquet")

    with patch("fisseq_data_pipeline.aggregatefeaturetype.setup_logging"):
        m.main.__wrapped__(
            make_ft_cfg(tmp_path, output_root=str(tmp_path / "unfiltered"))
        )
    unfiltered_result = pl.read_parquet(tmp_path / "unfiltered.input.parquet")

    # Filtering to a single A1B row means only that A1B row contributes, and
    # its single-cell mean must equal the raw feature value at that row exactly.
    assert set(filtered_result["meta_aa_changes"].to_list()) == {"A1B"}
    filtered_row = filtered_result.filter(pl.col("meta_aa_changes") == "A1B").row(
        0, named=True
    )
    assert filtered_row["f1_mean"] == pytest.approx(10.0)

    unfiltered_row = unfiltered_result.filter(pl.col("meta_aa_changes") == "A1B").row(
        0, named=True
    )
    assert filtered_row["f1_mean"] != pytest.approx(unfiltered_row["f1_mean"])


def test_main_output_root_naming(tmp_path) -> None:
    write_agg_input_parquet(tmp_path)
    root = str(tmp_path / "run1")
    with patch("fisseq_data_pipeline.aggregatefeaturetype.setup_logging"):
        m.main.__wrapped__(make_ft_cfg(tmp_path, output_root=root))
    assert (tmp_path / "run1.input.parquet").exists()


# ---------------------------------------------------------------------------
# main: downsample_wt
# ---------------------------------------------------------------------------


def _write_downsample_input(tmp_path) -> None:
    """20 control rows with distinct f1 values, plus one variant group."""
    pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * 20 + ["A1B"] * 3,
            "meta_is_control": [True] * 20 + [False] * 3,
            "f1": [float(i) for i in range(20)] + [5.0, 5.0, 5.0],
        }
    ).write_parquet(tmp_path / "input.parquet")


def test_main_downsample_wt_changes_output(tmp_path) -> None:
    _write_downsample_input(tmp_path)
    with patch("fisseq_data_pipeline.aggregatefeaturetype.setup_logging"):
        m.main.__wrapped__(
            make_ft_cfg(tmp_path, aggregator="KS", output_root=str(tmp_path / "full"))
        )
        m.main.__wrapped__(
            make_ft_cfg(
                tmp_path,
                aggregator="KS",
                output_root=str(tmp_path / "down"),
                downsample_wt=0.25,
                seed=1,
            )
        )
    full_row = _get_row(pl.read_parquet(tmp_path / "full.input.parquet"), "A1B")
    down_row = _get_row(pl.read_parquet(tmp_path / "down.input.parquet"), "A1B")
    assert full_row["f1_KS"] != pytest.approx(down_row["f1_KS"])


def test_main_downsample_wt_none_leaves_output_unaffected(
    tmp_path,
) -> None:
    _write_downsample_input(tmp_path)
    with patch("fisseq_data_pipeline.aggregatefeaturetype.setup_logging"):
        m.main.__wrapped__(make_ft_cfg(tmp_path, aggregator="mean"))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert set(result["meta_aa_changes"].to_list()) == {"A1B"}


def test_main_downsample_wt_different_seeds_differ(tmp_path) -> None:
    _write_downsample_input(tmp_path)
    with patch("fisseq_data_pipeline.aggregatefeaturetype.setup_logging"):
        m.main.__wrapped__(
            make_ft_cfg(
                tmp_path,
                aggregator="KS",
                output_root=str(tmp_path / "seed1"),
                downsample_wt=0.25,
                seed=1,
            )
        )
        m.main.__wrapped__(
            make_ft_cfg(
                tmp_path,
                aggregator="KS",
                output_root=str(tmp_path / "seed2"),
                downsample_wt=0.25,
                seed=2,
            )
        )
    seed1_row = _get_row(pl.read_parquet(tmp_path / "seed1.input.parquet"), "A1B")
    seed2_row = _get_row(pl.read_parquet(tmp_path / "seed2.input.parquet"), "A1B")
    assert seed1_row["f1_KS"] != pytest.approx(seed2_row["f1_KS"])


def test_main_downsample_wt_float_out_of_range_raises(tmp_path) -> None:
    _write_downsample_input(tmp_path)
    with patch("fisseq_data_pipeline.aggregatefeaturetype.setup_logging"):
        with pytest.raises(ValueError):
            m.main.__wrapped__(make_ft_cfg(tmp_path, downsample_wt=1.5))


def test_main_downsample_wt_nonpositive_int_raises(tmp_path) -> None:
    _write_downsample_input(tmp_path)
    with patch("fisseq_data_pipeline.aggregatefeaturetype.setup_logging"):
        with pytest.raises(ValueError):
            m.main.__wrapped__(make_ft_cfg(tmp_path, downsample_wt=-1))
