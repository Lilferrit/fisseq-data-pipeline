import numpy as np
import pytest

from fisseq_data_pipeline.filter import get_clean_masks


def test_get_clean_masks_identifies_all_nan_zero_var_and_bad_rows():
    # Construct a 4x4 matrix with:
    # col0: all NaN  -> all_nan=True, zero_variance=True
    # col1: constant 5s -> all_nan=False, zero_variance=True
    # col2: varying numbers with one NaN -> not all_nan, nonzero variance
    # col3: proper varying numbers       -> valid
    X = np.array(
        [
            [np.nan, 5.0, 1.0, 10.0],
            [np.nan, 5.0, 2.0, 20.0],
            [np.nan, 5.0, np.nan, 30.0],
            [np.nan, 5.0, 4.0, 40.0],
        ]
    )

    all_nan, zero_var, contains_nan = get_clean_masks(X)

    # Column-level checks
    assert all_nan.tolist() == [True, False, False, False]
    assert zero_var.tolist() == [True, True, False, False]

    # Row-level checks
    # Row 2 has a NaN in col2 (a non-all-NaN column) -> True
    # Others have no NaNs in valid columns -> False
    assert contains_nan.tolist() == [False, False, True, False]


@pytest.mark.parametrize(
    "X, expected_all_nan, expected_zero_var, expected_contains_nan",
    [
        # All good: no NaNs, variance > 0
        (
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            [False, False],
            [False, False],
            [False, False],
        ),
        # Single row with NaN in valid column
        (
            np.array([[1.0, 2.0], [np.nan, 3.0]], dtype=np.float32),
            [False, False],
            [True, True],
            [False, True],
        ),
    ],
)
def test_get_clean_masks_various_cases(
    X, expected_all_nan, expected_zero_var, expected_contains_nan
):
    all_nan, zero_var, contains_nan = get_clean_masks(X)
    assert all_nan.tolist() == expected_all_nan
    assert zero_var.tolist() == expected_zero_var
    assert contains_nan.tolist() == expected_contains_nan
