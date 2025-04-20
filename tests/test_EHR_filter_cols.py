"""File for testing EHR_filter_cols function.

Work by Max Freitas
Last Updated April 18 2025.
"""

import re

import pandas as pd
import pytest

from ehr_utils import EHR_filter_cols


def test_EHR_filter_cols_basic() -> None:
    """Test basic filtering of specified values from single column."""
    input_df = pd.DataFrame(
        {
            "patient_id": [1, 2, 3, 4],
            "status": ["active", "inactive", "active", "pending"],
        }
    )

    expected = pd.DataFrame(
        {"patient_id": [1, 3], "status": ["active", "active"]}, index=[0, 2]
    )

    result = EHR_filter_cols(input_df, ["status"], ["inactive", "pending"])
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_filter_cols_missing_columns() -> None:
    """Test ValueError is raised when columns don't exist."""
    input_df = pd.DataFrame(
        {"patient_id": [1, 2], "status": ["active", "inactive"]}
    )

    expected_error = re.escape(
        "Columns not found in DataFrame: ['missing_col']"
    )

    with pytest.raises(ValueError, match=expected_error):
        EHR_filter_cols(input_df, ["status", "missing_col"], ["inactive"])


def test_EHR_filter_cols_empty_variables() -> None:
    """Test ValueError is raised when variables list is empty."""
    input_df = pd.DataFrame(
        {"patient_id": [1, 2], "status": ["active", "inactive"]}
    )

    with pytest.raises(
        ValueError, match="At least one column must be specified"
    ):
        EHR_filter_cols(input_df, [], ["inactive"])


def test_EHR_filter_cols_empty_values() -> None:
    """Test ValueError is raised when values_to_remove is empty."""
    input_df = pd.DataFrame(
        {"patient_id": [1, 2], "status": ["active", "inactive"]}
    )

    with pytest.raises(
        ValueError, match="At least one value to remove must be specified"
    ):
        EHR_filter_cols(input_df, ["status"], [])


def test_EHR_filter_cols_multi_column() -> None:
    """Test filtering across multiple columns."""
    input_df = pd.DataFrame(
        {
            "patient_id": [1, 2, 3, 4, 5],
            "status": ["active", "inactive", "active", "pending", "active"],
            "department": ["cardio", "neuro", "neuro", "cardio", "ortho"],
        }
    )

    expected = pd.DataFrame(
        {
            "patient_id": [1, 5],
            "status": ["active", "active"],
            "department": ["cardio", "ortho"],
        },
        index=[0, 4],
    )

    result = EHR_filter_cols(
        input_df, ["status", "department"], ["inactive", "pending", "neuro"]
    )
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_filter_cols_numeric_values() -> None:
    """Test filtering numeric values from numeric columns."""
    input_df = pd.DataFrame(
        {
            "patient_id": [1, 2, 3, 4],
            "age": [25, 30, 35, 25],
            "score": [100, 200, 300, 100],
        }
    )

    expected = pd.DataFrame(
        {"patient_id": [2, 3], "age": [30, 35], "score": [200, 300]},
        index=[1, 2],
    )

    result = EHR_filter_cols(input_df, ["age", "score"], [25, 100])
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_filter_cols_mixed_data_types() -> None:
    """Test filtering mixed data types (str, int, bool)."""
    input_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "is_active": [True, False, True],
        }
    )

    expected = pd.DataFrame(
        {
            "id": [1, 3],
            "name": ["Alice", "Charlie"],
            "is_active": [True, True],
        },
        index=[0, 2],
    )

    result = EHR_filter_cols(input_df, ["is_active", "name"], [False, "Bob"])
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_filter_cols_case_sensitivity() -> None:
    """Test that filtering is case-sensitive."""
    input_df = pd.DataFrame(
        {"patient_id": [1, 2, 3], "status": ["Active", "active", "INACTIVE"]}
    )

    expected = pd.DataFrame(
        {"patient_id": [2, 3], "status": ["active", "INACTIVE"]}, index=[1, 2]
    )

    result = EHR_filter_cols(input_df, ["status"], ["Active"])
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_filter_cols_multi_value_single_column() -> None:
    """Test filtering multiple values from single column."""
    input_df = pd.DataFrame(
        {
            "patient_id": [1, 2, 3, 4, 5],
            "department": ["Cardio", "Neuro", "Ortho", "Cardio", "Psych"],
        }
    )

    expected = pd.DataFrame(
        {"patient_id": [2, 3], "department": ["Neuro", "Ortho"]}
    )

    result = EHR_filter_cols(input_df, ["department"], ["Cardio", "Psych"])
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_EHR_filter_cols_empty_dataframe() -> None:
    """Test function with empty DataFrame."""
    input_df = pd.DataFrame(columns=["patient_id", "status"])

    # Test with missing columns (should still raise error)
    with pytest.raises(ValueError, match="Inputed DataFrame is empty."):
        EHR_filter_cols(input_df, ["status"], ["active"])


def test_EHR_filter_cols_multi_column_mixed_conditions() -> None:
    """Test filtering across multiple columns with different value types."""
    input_df = pd.DataFrame(
        {
            "case_id": [101, 102, 103, 104],
            "priority": ["High", "Medium", "Low", "Medium"],
            "is_urgent": [True, False, False, True],
        }
    )

    expected = pd.DataFrame(
        {"case_id": [102], "priority": ["Medium"], "is_urgent": [False]},
        index=[1],
    )

    result = EHR_filter_cols(
        input_df, ["priority", "is_urgent"], ["High", "Low", True]
    )
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_filter_cols_all_rows_removed() -> None:
    """Test ValueError when filtering would remove all rows."""
    input_df = pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "status": ["Processed", "Processed", "Processed"],
        }
    )

    with pytest.raises(
        ValueError, match="Filtering would remove all rows from the DataFrame"
    ):
        EHR_filter_cols(input_df, ["status"], ["Processed"])
