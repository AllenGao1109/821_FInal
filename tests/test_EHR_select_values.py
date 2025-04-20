"""File for testing EHR_select_values function.

Work by Max Freitas
Last Updated April 18 2025.
"""

import pandas as pd
import pytest

from EHR_processing_functions import EHR_select_values


def test_EHR_select_values_missing_column() -> None:
    """Test ValueError when specified column doesn't exist."""
    input_df = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
    with pytest.raises(
        ValueError, match="Column missing not found in DataFrame"
    ):
        EHR_select_values(input_df, "missing", int_range="10,20")


def test_EHR_select_values_numeric_range_exclusive() -> None:
    """Test numeric range filtering (exclusive of endpoints)."""
    input_df = pd.DataFrame(
        {"patient_id": [1, 2, 3, 4], "age": [30, 40, 50, 60]}
    )
    expected = pd.DataFrame({"patient_id": [3], "age": [50]}, index=[2])
    result = EHR_select_values(input_df, "age", int_range="40,60")
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_select_values_numeric_range_inclusive() -> None:
    """Test numeric range filtering (inclusive of endpoints)."""
    input_df = pd.DataFrame(
        {"patient_id": [1, 2, 3, 4], "age": [30, 40, 50, 60]}
    )
    expected = pd.DataFrame(
        {"patient_id": [2, 3, 4], "age": [40, 50, 60]}, index=[1, 2, 3]
    )
    result = EHR_select_values(
        input_df, "age", int_range="40,60", include_endpoints=True
    )
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_select_values_value_list() -> None:
    """Test filtering by list of specific values."""
    input_df = pd.DataFrame(
        {"id": [1, 2, 3, 4], "color": ["red", "blue", "green", "blue"]}
    )
    expected = pd.DataFrame(
        {"id": [2, 4], "color": ["blue", "blue"]}, index=[1, 3]
    )
    result = EHR_select_values(input_df, "color", values=["blue"])
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_select_values_missing_params() -> None:
    """Test ValueError when neither range nor values are provided."""
    input_df = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(
        ValueError, match="Must provide either 'int_range' or 'values'"
    ):
        EHR_select_values(input_df, "value")


def test_EHR_select_values_too_many_range_values() -> None:
    """Test ValueError when range contains more than two numbers."""
    input_df = pd.DataFrame({"value": [1, 2, 3]})

    expected_error = (
        r"Invalid range format. Use 'lower,upper' with numeric values."
    )

    with pytest.raises(ValueError, match=expected_error):
        EHR_select_values(input_df, "value", int_range="10,20,30")


def test_EHR_select_values_single_value_in_range() -> None:
    """Test ValueError when range contains only one number."""
    input_df = pd.DataFrame({"age": [25, 30, 35]})

    expected_error = (
        r"Invalid range format. Use 'lower,upper' with numeric values."
    )

    with pytest.raises(ValueError, match=expected_error):
        EHR_select_values(input_df, "age", int_range="30")


def test_EHR_select_values_empty_range_string() -> None:
    """Test ValueError when range string is empty."""
    input_df = pd.DataFrame({"score": [85, 90, 95]})

    expected_error = (
        r"Invalid range format. Use 'lower,upper' with numeric values."
    )

    with pytest.raises(ValueError, match=expected_error):
        EHR_select_values(input_df, "score", int_range="")
