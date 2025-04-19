"""File for testing EHR_standarize function.

Work by Max Freitas
Last Updated April 18 2025.
"""


from _collections_abc import Sequence
from typing import Any, Union

import pandas as pd
import pytest
from EHR_processing_functions import EHR_standardize


def test_EHR_standarize_basic(tmp_path: Any) -> pd.DataFrame:
    """Basic test of EHR_standarize, where all values are present."""
    # Create a simple CSV file
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2,col3\nhello,1,1.5\nworld,2,2.5\n")
    
    
    result = EHR_standardize(dataset=str(csv_file), 
    potential_missing= [""], 
    format="csv")
    expected= pd.DataFrame({
        "str_col1": ["hello", "world"],
        "int_col2": [1, 2],
        "float_col3": [1.5, 2.5]
    })

    pd.testing.assert_frame_equal(result, expected)
    

def test_EHR_standarize_wrong_file_type(tmp_path:Any) -> None:
    """Test of EHR standarize, where file type is not supported.
    
    Only supported types are 'csv', 'tsv', and 'xlsx'.
    Should raise ValueError for other file types.
    """
    txt_file= tmp_path / "test.txt"
    txt_file.write_text("col1,col2,col3\nhello,1,1.5\nworld,,2.5\n")

    with pytest.raises(ValueError, match="Unsupported format: txt"):
        EHR_standardize(str(txt_file), potential_missing=[""], format="txt")


def test_EHR_standarize_wrong_file_format(tmp_path: Any) -> None:
    """Test of EHR standarize, where file type does not match file extension.
    
    Should raise ValueError for mismatched file type.
    """
    file = tmp_path / "data.csv"
    file.write_text("col1,col2\nhello,1\nworld,2")

    with pytest.raises(
        ValueError, match=r"File extension '.csv' does not match " \
        "expected format: 'tsv'"
        ):
        EHR_standardize(str(file), potential_missing=[""], format="tsv")


def test_EHR_standarize_empty_file(tmp_path: Any) -> None:
    """Test of EHR_standarize where file is empty.
    
    Should raise ValueError for empty file.
    """
    file= tmp_path/"data.csv"
    file.write_text("")

    with pytest.raises(
        ValueError, match=r".*data\.csv is empty"
    ):
        EHR_standardize(str(file), potential_missing=[""], format="csv")

def test_EHR_standarize_invalid_potential_missing(tmp_path: Any) -> None:
    """Test of EHR_standarize where potential_missing is not list[str].
    
    Should raise ValueError for invalid potential_missing format
    """
    # list of invalid inputs to test:
    invalid_inputs = [
        None,
        "missing",                  # string, not a list
        123,                        # int
        {"key": "value"},           # dict
        (1, 2),                     # tuple
    ]
    # data
    file = tmp_path / "data.csv"
    file.write_text("col1,col2\nhello,1\nworld,2")

    for invalid in invalid_inputs:
        with pytest.raises(ValueError, match="is not a list of strings"):
            # if we dont the ignore, mypy will give warning, 
            #   but we want to try to pass invalid input.
            EHR_standardize(str(file), potential_missing=invalid, format="csv") # type: ignore[arg-type]


def test_EHR_standardize_multiple_missing_values(tmp_path: Any) -> None:
    """Test with multiple missing value representations.

    Expected Behavior:
        "NA"   -> nan
        "?"    -> nan
        "NULL" -> nan
    """
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2,col3\nhello,NA,1.5\nworld,?,NULL\n")
    
    missing_values = ["NA", "?", "NULL"]
    result = EHR_standardize(
        dataset=str(csv_file),
        potential_missing=missing_values,
        format="csv"
    )
    for missing_val in missing_values:
        assert not (result == missing_val).any().any(), \
            f"Found '{missing_val}' in DataFrame after standardization"

def test_EHR_standardize_integer_stored_as_float(tmp_path: Any) -> None:
    """Test: integer values stored as floats are correctly labeled as int."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2\n1.0,1.5\n2.0,2.5\n")
    
    result = EHR_standardize(
        dataset=str(csv_file),
        potential_missing=[""],
        format="csv"
    )
    assert list(result.columns) == ["int_col1", "float_col2"]

def test_EHR_standardize_all_missing_column(tmp_path: Any) -> None:
    """Test a column where all values are missing."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2,col3\nhello,,\nworld,,\n")
    
    result = EHR_standardize(
        dataset=str(csv_file),
        potential_missing=[""],
        format="csv"
    )
    assert pd.isna(result["int_col2"]).all()


def test_EHR_standardize_boolean_columns(tmp_path: Any) -> None:
    """Test boolean columns (should be treated as strings)."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2\nTrue,1\nFalse,2\n")
    
    result = EHR_standardize(
        dataset=str(csv_file),
        potential_missing=[""],
        format="csv"
    )
    assert list(result.columns) == ["str_col1", "int_col2"]

def test_EHR_standardize_datetime_columns(tmp_path: Any) -> None:
    """Test datetime columns (should be treated as strings)."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2\n2023-01-01,1\n2023-01-02,2\n")
    
    result = EHR_standardize(
        dataset=str(csv_file),
        potential_missing=[""],
        format="csv"
    )
    assert list(result.columns) == ["str_col1", "int_col2"]

def test_EHR_standardize_duplicate_columns(tmp_path: Any) -> None:
    """Test duplicate column names."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col1,col2\nhello,1,1.5\nworld,2,2.5\n")
    
    result = EHR_standardize(
        dataset=str(csv_file),
        potential_missing=[""],
        format="csv"
    )
    assert len(result.columns) == 3  # Ensure all columns are processed


def test_EHR_standardize_scientific_notation(tmp_path: Any) -> None:
    """Test columns with scientific notation numbers.
    
    Expected: 
        1e6 -> 1000000 (int)
    """
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2\n1e6,1e7\n2e8,2e5\n")
    
    result = EHR_standardize(
        dataset=str(csv_file),
        potential_missing=[""],
        format="csv"
    )
    assert list(result.columns) == ["int_col1", "int_col2"]


def test_EHR_standardize_column_renaming_logic(tmp_path: Any) -> None:
    """Verify column type prefixes are applied correctly.
    
    Expected:
        int_col -> int_int_col
        float_col -> float_float_col
        str_col -> str_str_col
    """
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("int_col,float_col,str_col\n1,1.5,hello\n2,2.5,world\n")
    
    result = EHR_standardize(
        dataset=str(csv_file),
        potential_missing=[""],
        format="csv"
    )
    
    assert all(
        col.startswith(("int_", "float_", "str_")) for col in result.columns
        )
    

def test_EHR_standardize_numeric_missing_values(tmp_path: Any) -> None:
    """Test numeric values treated as missing.
    
    Expected:
        -1 (str)  -> nan
        999 (str) -> nan
        -1 (int)  -> nan
        999 (int) -> nan
    """
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2,col3\nhello,-1,1.5\nworld,999,2.5\n")
    
    missing_values : Sequence[Union[str, int, float]]=["-1", "999", -1, 999]
    result = EHR_standardize(
        dataset=str(csv_file),
        potential_missing=missing_values,
        format="csv"
    )
    for missing_val in missing_values:
        assert not (result == missing_val).any().any(), \
            f"Found '{missing_val}' in DataFrame after standardization"

def test_EHR_standardize_float_missing_values(tmp_path: Any) -> None:
    """Test float values treated as missing.
    
    Expected:
        1.0 (float)     -> nan
        999.0 (float)   -> nan

    """
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2,col3\nhello,1.0,1.5\nworld,999.0,2.5\n")
    
    missing_values=[1.0, 999.0]
    result = EHR_standardize(
        dataset=str(csv_file),
        potential_missing=missing_values,
        format="csv"
    )
    for missing_val in missing_values:
        assert not (result == missing_val).any().any(), \
            f"Found '{missing_val}' in DataFrame after standardization"
        



    