"""File for testing EHR_preprocessing function.

Work by Max Freitas
Last Updated April 18 2025.
"""

import re

import numpy as np
import pandas as pd
import pytest

from ehr_utils import EHR_preprocessing


def test_EHR_preprocessing_basic_remove() -> None:
    """Basic test of EHR_preprocessing where remove = True.

    Output:
        Remove rows with any np.nan.
    """
    input_df = pd.DataFrame(
        {"col1": ["hello", np.nan], "col2": [1, 2], "col3": [1.5, np.nan]}
    )

    expected = pd.DataFrame(
        {"col1": ["hello"], "col2": [1], "col3": [1.5]},
    )

    result = EHR_preprocessing(input_df, remove=True, impute=False)
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_preprocessing_basic_no_remove() -> None:
    """Basic test of EHR_preprocessing where remove = False.

    Output:
        identical to input.
    """
    input_df = pd.DataFrame(
        {"col1": ["hello", "world"], "col2": [1, np.nan], "col3": [1.5, 2.5]}
    )

    expected = pd.DataFrame(
        {"col1": ["hello", "world"], "col2": [1, np.nan], "col3": [1.5, 2.5]}
    )

    result = EHR_preprocessing(input_df, remove=False, impute=False)
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_preprocessing_mean_impute() -> None:
    """Test of EHR_preprocessing with mean imputation.

    Output:
        float and int col np.nan replaced with mean value for that column.
    """
    input_df = pd.DataFrame(
        {
            "col1": ["hello", "world", "flow"],
            "float_col2": [1, np.nan, 3],
            "col3": [1.5, 2.5, np.nan],
        }
    )

    expected = pd.DataFrame(
        {
            "col1": ["hello", "world", "flow"],
            "int_col2": [1, 2, 3],
            "col3": [1.5, 2.5, 2.0],
        }
    )

    result = EHR_preprocessing(input_df, remove=False, impute="mean")
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_preprocessing_median_impute() -> None:
    """Test of EHR_preprocessing with median imputation.

    Output:
        float and int col np.nan replaced with median value for that column.
    """
    input_df = pd.DataFrame(
        {
            "col1": ["hello", "world", "flow", np.nan],
            "float_col2": [1, np.nan, 1, 2],
            "col3": [1.5, 2.5, 2.5, np.nan],
        }
    )

    expected = pd.DataFrame(
        {
            "col1": ["hello", "world", "flow", np.nan],
            "int_col2": [1, 1, 1, 2],
            "col3": [1.5, 2.5, 2.5, 2.5],
        }
    )

    result = EHR_preprocessing(input_df, remove=False, impute="median")
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_preprocessing_max_value_impute() -> None:
    """Test of EHR_preprocessing with max imputation.

    Output:
        float and int col np.nan replaced with max value for that column.
    """
    input_df = pd.DataFrame(
        {
            "col1": ["hello", "world", "flow", np.nan],
            "float_col2": [1, np.nan, 1, 2],
            "col3": [1.5, 2.5, 3.5, np.nan],
        }
    )

    expected = pd.DataFrame(
        {
            "col1": ["hello", "world", "flow", np.nan],
            "int_col2": [1, 2, 1, 2],
            "col3": [1.5, 2.5, 3.5, 3.5],
        }
    )

    result = EHR_preprocessing(input_df, remove=False, impute="largest")
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_preprocessing_min_value_impute() -> None:
    """Test of EHR_preprocessing with min imputation.

    Output:
        float and int col np.nan replaced with min value for that column.
    """
    input_df = pd.DataFrame(
        {
            "col1": ["hello", "world", "flow", np.nan],
            "float_col2": [1, np.nan, 1, 2],
            "col3": [1.5, 2.5, 3.5, np.nan],
        }
    )

    expected = pd.DataFrame(
        {
            "col1": ["hello", "world", "flow", np.nan],
            "int_col2": [1, 1, 1, 2],
            "col3": [1.5, 2.5, 3.5, 1.5],
        }
    )

    result = EHR_preprocessing(input_df, remove=False, impute="smallest")
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_preprocessing_float_rename_remove_true() -> None:
    """Test col rename of floats.

    Expected:
        If remove = True, and Impute = False
            float_col1= [1, np.nan] -> int_col1[1]
            col2 = [3, np.nan]      -> col2[3]
    """
    input_df = pd.DataFrame({"float_col1": [1, np.nan], "col2": [3, np.nan]})

    expected = pd.DataFrame(
        {
            "int_col1": [1],
            "col2": [3],
        }
    )

    result = EHR_preprocessing(input_df, remove=True)
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_preprocessing_float_rename_remove_false() -> None:
    """Test col rename of floats.

    Expected:
        If remove = False, and Impute = False
            float_co1= [1, np.nan] -> float_col1[1, np.nan]
            col2 = [3, np.nan]      -> col2[3, np.nan]
    """
    input_df = pd.DataFrame({"float_col1": [1, np.nan], "col2": [3, np.nan]})

    expected = pd.DataFrame(
        {
            "float_col1": [1, np.nan],
            "col2": [3, np.nan],
        }
    )

    result = EHR_preprocessing(input_df, remove=False)
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_preprocessing_impute_and_remove() -> None:
    """Test with imputation and removal.

    For col2:
        The mean produces a integer, so column becomes int as well.
    """
    input_df = pd.DataFrame(
        {
            "str_col1": ["hello", "world", "foo", np.nan],
            "float_col2": [1, np.nan, 3, 5],
            "float_col3": [1.5, 2.5, np.nan, 3.5],
        }
    )

    expected = pd.DataFrame(
        {
            "str_col1": ["hello", "world", "foo"],
            "int_col2": [1, 3, 3],
            "float_col3": [1.5, 2.5, 2.5],
        },
        index=[0, 1, 2],
    )

    result = EHR_preprocessing(input_df, remove=True, impute="mean")
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_preprocessing_impute_and_remove_no_float_conversion() -> None:
    """Test with imputation and removal.

    For col2:
        The mean produces a float, so column stays float.
    """
    input_df = pd.DataFrame(
        {
            "str_col1": ["hello", "world", "foo"],
            "float_col2": [0, np.nan, 1],
            "float_col3": [1.5, 2.5, 5.0],
        }
    )

    expected = pd.DataFrame(
        {
            "str_col1": ["hello", "world", "foo"],
            "float_col2": [0, 0.5, 1],
            "float_col3": [1.5, 2.5, 5.0],
        },
        index=[0, 1, 2],
    )

    result = EHR_preprocessing(input_df, remove=True, impute="mean")
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_preprocessing_invalid_impute_strategy() -> None:
    """Test that invalid impute strategy raises ValueError."""
    input_df = pd.DataFrame(
        {"col1": ["hello", "world"], "col2": [1, np.nan], "col3": [1.5, 2.5]}
    )

    with pytest.raises(ValueError, match="Unknown imputation strategy"):
        EHR_preprocessing(input_df, remove=False, impute="invalid_strategy")


def test_EHR_preprocessing_empty_dataframe() -> None:
    """Test with empty dataframe."""
    input_df = pd.DataFrame(columns=["col1", "col2", "col3"])

    with pytest.raises(ValueError, match="DataFrame is empty."):
        EHR_preprocessing(input_df)


def test_EHR_preprocessing_all_nan_column() -> None:
    """Test with a column that's entirely np.nan.

    This should raise a ValueError, because now DataFrame is empty.
    """
    input_df = pd.DataFrame(
        {
            "col1": ["hello", "world"],
            "col2": [np.nan, np.nan],
            "col3": [1.5, 2.5],
        }
    )

    match = "DataFrame is empty: At least one column is all np.nan"
    with pytest.raises(ValueError, match=match):
        EHR_preprocessing(input_df)


def test_EHR_preprocessing_datetime_columns() -> None:
    """Test with datetime columns (should be left unchanged)."""
    input_df = pd.DataFrame(
        {
            "date_col": pd.to_datetime(["2020-01-01", np.nan, "2020-01-03"]),
            "int_col": [1, 2, 3],
        }
    )

    expected = pd.DataFrame(
        {
            "date_col": pd.to_datetime(["2020-01-01", "2020-01-03"]),
            "int_col": [1, 3],
        },
        index=[0, 2],
    )

    result = EHR_preprocessing(input_df, remove=True)
    pd.testing.assert_frame_equal(result, expected)


def test_EHR_removal_true_pd_na_not_removed() -> None:
    """Test where pd.NA values exist but aren't removed with remove=True."""
    input_df = pd.DataFrame(
        {"col1": ["hello", "NA"], "col2": [1, "NA"], "col3": [1.5, 2.5]}
    )

    warning_message = (
        "No rows were removed despite remove=True"
        "This suggests:"
        "   1. No np.nan  "
        "   2. Missing format is different"
        "   3. No missing values"
    )

    with pytest.warns(UserWarning, match=re.escape(warning_message)):
        EHR_preprocessing(input_df, remove=True)
