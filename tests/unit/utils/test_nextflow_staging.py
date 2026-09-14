import pytest

from fisseq_data_pipeline.utils.nextflow_staging import reconstruct_staged_paths


def test_single_file_has_no_index():
    """Nextflow substitutes stageAs' `*` with an empty string for exactly one file."""
    assert reconstruct_staged_paths(1, "res_input") == ["res_input_.parquet"]


def test_multiple_files_are_one_indexed():
    assert reconstruct_staged_paths(3, "res_input") == [
        "res_input_1.parquet",
        "res_input_2.parquet",
        "res_input_3.parquet",
    ]


def test_two_files_start_at_one_not_zero():
    assert reconstruct_staged_paths(2, "agg_input") == [
        "agg_input_1.parquet",
        "agg_input_2.parquet",
    ]


def test_prefix_is_honored():
    assert reconstruct_staged_paths(1, "other") == ["other_.parquet"]


@pytest.mark.parametrize("n", [0, -1])
def test_non_positive_n_raises(n):
    with pytest.raises(ValueError, match="n must be >= 1"):
        reconstruct_staged_paths(n, "res_input")


@pytest.mark.parametrize("n", [1, 2, 5, 10])
def test_length_always_matches_n(n):
    assert len(reconstruct_staged_paths(n, "res_input")) == n
