"""File for creating EHR_standarize function.

Work by Max Freitas
Last Updated April 17 2025.

"""

import os
import warnings
from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


def EHR_standardize(
    dataset: str,
    potential_missing: Sequence[Union[str, int, float]],
    format: str = "csv",
) -> pd.DataFrame:
    """Reads a dataset; standarize missing values, standarize col names.

    Column Names changed to represent their type
        1. if str: col1 -> str_col1
        2. if int: col1 -> int_col1
        3. if float: col1 -> float_col1

    Args:
        dataset: Path to the dataset file.
        potential_missing: List of values to be treated as missing.
            Can only be str, int, or float
        format: File format: 'csv', 'tsv', or 'xlsx'.

    Returns:
        A pandas DataFrame with
        standardized missing values, and standarized col names.
    """
    # make sure dataset is not empty
    if os.path.getsize(dataset) == 0:
        raise ValueError(f"{dataset} is empty")

    # check extension matches inputed format
    ext = os.path.splitext(dataset)[1].lower().lstrip(".")
    if ext != format:
        raise ValueError(
            f"File extension '.{ext}' does not match "
            f"expected format: '{format}'"
        )

    # check potential_missing format (must be str, int, float)
    if not isinstance(potential_missing, list) or not all(
        isinstance(x, (str, int, float)) for x in potential_missing
    ):
        raise ValueError(
            f"{potential_missing}is not a list of strings, integers or floats"
        )

    # read in
    if format == "csv":
        df = pd.read_csv(dataset)
    elif format == "tsv":
        df = pd.read_csv(dataset, sep="\t")
    elif format == "xlsx":
        df = pd.read_excel(dataset)
    else:
        raise ValueError(f"Unsupported format: {format}")

    # raise warning if duplicate col names
    if len(df.columns) != len(set(df.columns)):
        warnings.warn(
            "Duplicate column names detected "
            "and automatically renamed by pandas",
            UserWarning,
            stacklevel=2,
        )

    # Replace missing values with the standardized format
    with pd.option_context("future.no_silent_downcasting", True):
        df = df.replace(potential_missing, np.nan)
    df = df.infer_objects(copy=False)

    # rename cols based on their dtype

    new_cols: dict[str, Any] = {}

    for col in df.columns:
        dtype = df[col].dtype

        # make sure all integer cols are renamed as int
        if pd.api.types.is_integer_dtype(dtype):
            new_cols[col] = f"int_{col}"
        elif pd.api.types.is_float_dtype(df[col]):
            if (df[col].dropna() % 1 == 0).all():
                df[col] = pd.to_numeric(df[col], downcast="integer")
                new_cols[col] = f"int_{col}"
            else:
                new_cols[col] = f"float_{col}"
        else:
            new_cols[col] = f"str_{col}"

    df.rename(columns=new_cols, inplace=True)

    return df


def EHR_preprocessing(
    df: pd.DataFrame, remove: bool = True, impute: str | bool = False
) -> pd.DataFrame:
    """Takes in a pandas dataframe, and outputs a processed dataframe.

    Converts float columns with float_ prefix:
        to int if they have no np.nan after processing
        change prefix from float_ to int_ if col has prefix float_

    Expects na format to be np.nan.

    Args:
        df: pd.DataFrame object

        remove: should missing values be removed
            "True": remove rows with missing values
                (after impute, if impute=True)
            "False": don't remove rows with missing values

        impute: only applies to cols of type int or float
            "False": don't impute (just remove if remove=True)
            "mean": change missing values to column mean
            "largest": change missing value to column max
            "smallest": change missing value to column min
            "median": change missing value to column median

    Potential Problem:
        If column is expected to be int, but contains "NaN' or similar value,
        It will automatically become a float.

    Returns:
        A pandas DataFrame with rows with missing values removed or replaced.

    """
    # check if df is empty
    if df.empty:
        raise ValueError("DataFrame is empty.")
    df = df.copy()

    # deal with imputation
    if impute is not False:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if impute == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif impute == "largest":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].max())
        elif impute == "smallest":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].min())
        elif impute == "median":
            df[numeric_cols] = df[numeric_cols].fillna(
                df[numeric_cols].median()
            )
        else:
            raise ValueError(f"Unknown imputation strategy: {impute}")

    if remove:
        pre_drop_count = len(df)
        df = df.dropna()

        if df.empty:
            raise ValueError(
                "DataFrame is empty: At least one column is all np.nan"
            )

        if pre_drop_count == len(df) and impute is False:
            warnings.warn(
                "No rows were removed despite remove=True"
                "This suggests:"
                "   1. No np.nan  "
                "   2. Missing format is different"
                "   3. No missing values",
                UserWarning,
                stacklevel=2,
            )

    # convert float columns with integer-like values to int
    rename_map = {}

    for col in df.select_dtypes(include="float").columns:
        no_nans = df[col].isna().sum() == 0
        is_integer_like = (df[col].dropna() % 1 == 0).all()

        if no_nans and is_integer_like:
            df[col] = df[col].astype("int64")
            if col.startswith("float_"):
                base_name = col.removeprefix("float_")
                rename_map[col] = f"int_{base_name}"

    df.rename(columns=rename_map, inplace=True)

    return df


def EHR_select_cols(
    df: pd.DataFrame, variable_list: list[str]
) -> pd.DataFrame:
    """Select specificied columns from the dataframe.

    Args:
        df: Input DataFrame

        variable_list: list of column names to select

    Returns:
        A pandas DataFrame with only selected columns
    """
    # Validate input type
    if not isinstance(variable_list, list):
        raise TypeError(
            f"variable_list must be a list, got {type(variable_list).__name__}"
        )

    # Validate non-empty list
    if not variable_list:
        raise ValueError(
            "Empty column list provided - must specify at least one column"
        )

    # Check for duplicate columns in request
    if len(variable_list) != len(set(variable_list)):
        dupes = {col for col in variable_list if variable_list.count(col) > 1}
        raise ValueError(
            f"Duplicate columns in selection list: {sorted(dupes)}"
        )

    # check variable_list exists in df
    missing_cols = [col for col in variable_list if col not in df.columns]

    if missing_cols:
        raise KeyError(
            f"The following columns were not found in the DataFrame:"
            f" {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )
    return df[variable_list]


def EHR_filter_cols(
    df: pd.DataFrame,
    variables: list[str],
    values_to_remove: Sequence[Union[str, int, float, bool]],
) -> pd.DataFrame:
    """Filter out specified values from specified column(s) in a DataFrame.

    Removes row if any of the specified columns contain the specified value.

    Args:
        df: Input Dataframe
        variables: columns you wish to filter based
        values_to_remove: value(s) to remove
            Any rows that have value(s) that are removed will be deleted

    Returns:
        pd.DataFrame:
        Filtered DataFrame with rows containing the specified values removed

    """
    # Check for empty DataFrame
    if df.empty:
        raise ValueError("Inputed DataFrame is empty.")

    # check cols exist
    if not all(col in df.columns for col in variables):
        missing_cols = [col for col in variables if col not in df.columns]
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    # make sure variables is not empty
    if not variables:
        raise ValueError("At least one column must be specified")

    # make sure values_to_remove is non-empty
    if not values_to_remove:
        raise ValueError("At least one value to remove must be specified.")

    # filter the DataFrame
    mask = ~df[variables].isin(values_to_remove).any(axis=1)

    # Check if filtering would remove all rows
    mask = ~df[variables].isin(values_to_remove).any(axis=1)
    if not mask.any():
        raise ValueError("Filtering would remove all rows from the DataFrame")
    return df[mask].copy()


def EHR_select_values(
    df: pd.DataFrame,
    variable: str,
    int_range: Optional[str] = None,
    values: Optional[list[str]] = None,
    include_endpoints: bool = False,
) -> pd.DataFrame:
    """Select rows based on value for a specific column.

    Args:
        df: Input DataFrame

        variable: variable we wish to filter based on
        int_range: interval of values we wish to select values from.
            if _intrange="40,60" all values between 40 and 60 are selected.
        include_endpoints: if true include endpoints of range.
        values: list of values to select
            if values= ("white", "black", "asian"),
            rows with those values for the variable of interest are selected.

    Returns:
        A pandas DataFrame with only selected values for specified columns.
    """
    # check for valid input
    if variable not in df.columns:
        raise ValueError(f"Column {variable} not found in DataFrame")

    # handle double input
    if int_range and values is not None:
        raise ValueError("Can only choose range or values.")

    # handle numeric range
    if int_range is not None:
        try:
            parts = [float(x.strip()) for x in int_range.split(",")]
            if len(parts) != 2:
                raise ValueError(
                    "Range string must contain exactly"
                    "two numbers separated by a comma."
                )

            lower, upper = sorted(parts)

            if include_endpoints:
                return df[(df[variable] >= lower) & (df[variable] <= upper)]
            else:
                return df[(df[variable] > lower) & (df[variable] < upper)]
        except ValueError as err:
            raise ValueError(
                "Invalid range format. Use 'lower,upper' with numeric values."
            ) from err

    if values is not None:
        return df[df[variable].isin(values)]

    raise ValueError("Must provide either 'int_range' or 'values'.")
