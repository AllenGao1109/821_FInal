"""File for testing EHR_select_cols function.

Work by Max Freitas
Last Updated April 18 2025.
"""

import pandas as pd
import pytest

from EHR_processing_functions import EHR_select_cols


def test_EHR_select_cols_basic() -> None:
    """Basic test of EHR_select_cols.

    Expected: select two cols choosen:
        'patient_id' and 'age'
    """
    input_df = pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "age": [25, 30, 35],
            "gender": ["M", "F", "M"],
        }
    )

    expected = pd.DataFrame({"patient_id": [1, 2, 3], "age": [25, 30, 35]})

    result = EHR_select_cols(input_df, ["patient_id", "age"])
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_select_cols_single_col_select() -> None:
    """Basic test of EHR_select_cols.

    Expected: select one col choosen:
        'patient_id'
    """
    input_df = pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "age": [25, 30, 35],
            "gender": ["M", "F", "M"],
        }
    )

    expected = pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
        }
    )

    result = EHR_select_cols(input_df, ["patient_id"])
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_select_cols_with_index_preservation() -> None:
    """Test of EHR_select cols if index is preserved after column selection.

    Expected: Selected columns with original index maintained.
    """
    input_df = pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "age": [25, 30, 35],
            "gender": ["M", "F", "M"],
        },
        index=["a", "b", "c"],
    )

    expected = pd.DataFrame(
        {"patient_id": [1, 2, 3], "age": [25, 30, 35]}, index=["a", "b", "c"]
    )

    result = EHR_select_cols(input_df, ["patient_id", "age"])
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_select_cols_with_different_column_order() -> None:
    """Test of EHR_select_cols.

    Where: selection order is not the same as order of cols in original df.

    Expected: Columns in the order specified, not original order.
    """
    input_df = pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "age": [25, 30, 35],
            "gender": ["M", "F", "M"],
        }
    )

    expected = pd.DataFrame({"age": [25, 30, 35], "patient_id": [1, 2, 3]})

    result = EHR_select_cols(input_df, ["age", "patient_id"])
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_select_cols_with_mixed_dtypes() -> None:
    """Test of EHR_select_cols with mixed data types in selected columns.

    Expected: Correctly selects columns with different dtypes.
    """
    input_df = pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "age": [25.5, 30.2, 35.8],
            "gender": ["M", "F", "M"],
            "is_active": [True, False, True],
        }
    )

    expected = pd.DataFrame(
        {"patient_id": [1, 2, 3], "is_active": [True, False, True]}
    )

    result = EHR_select_cols(input_df, ["patient_id", "is_active"])
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_select_cols_with_missing_column() -> None:
    """Test that error is raised when any column is missing."""
    input_df = pd.DataFrame({"patient_id": [1, 2, 3], "age": [25, 30, 35]})

    expected_error_pattern = (
        r"The following columns were not found in the DataFrame: \['red'\]\\n"
        r"Available columns: \['patient_id', 'age'\]"
    )

    with pytest.raises(KeyError, match=expected_error_pattern):
        EHR_select_cols(input_df, ["red", "age"])


def test_EHR_select_cols_with_non_list_input() -> None:
    """Test that TypeError is raised when input is not a list."""
    input_df = pd.DataFrame({"patient_id": [1, 2, 3], "age": [25, 30, 35]})

    expected_error_pattern = r"variable_list must be a list, got tuple"

    with pytest.raises(TypeError, match=expected_error_pattern):
        # if we dont the ignore, mypy will give warning,
        #   but we want to try to pass invalid input.
        EHR_select_cols(input_df, ("patient_id", "age"))  # type: ignore[arg-type]


def test_EHR_select_cols_with_empty_list() -> None:
    """Test that ValueError is raised when empty list is provided."""
    input_df = pd.DataFrame({"patient_id": [1, 2, 3], "age": [25, 30, 35]})

    expected_error_pattern = (
        r"Empty column list provided - "
        "must specify at least one column"
    )

    with pytest.raises(ValueError, match=expected_error_pattern):
        EHR_select_cols(input_df, [])


def test_EHR_select_cols_with_duplicate_columns() -> None:
    """Test that ValueError is raised when duplicate columns are requested."""
    input_df = pd.DataFrame({"patient_id": [1, 2, 3], "age": [25, 30, 35]})

    expected_error_pattern = r"Duplicate columns in selection list: \['age'\]"

    with pytest.raises(ValueError, match=expected_error_pattern):
        EHR_select_cols(input_df, ["patient_id", "age", "age"])


def test_EHR_select_cols_with_case_mismatch() -> None:
    """Test exact error message for case mismatch."""
    input_df = pd.DataFrame({"PatientID": [1, 2, 3], "Age": [25, 30, 35]})

    # Use regex pattern that accounts for either column order
    expected_error_pattern = (
        r"The following columns were not found in the DataFrame: "
        r"\['patientid'\]\\n"
        r"Available columns: \[("
        r"'PatientID', 'Age'|"
        r"'Age', 'PatientID'"
        r")\]"
    )

    with pytest.raises(KeyError, match=expected_error_pattern):
        EHR_select_cols(input_df, ["patientid", "Age"])
