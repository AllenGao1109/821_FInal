""""File for creating EHR_standarize function.

Work by Max Freitas
Last Updated April 16 2025.

"""

from typing import Any, Optional

import numpy as np
import pandas as pd


def EHR_standardize(
    dataset: str,
    potential_missing: list[Any],
    format: str = "csv"
) -> pd.DataFrame:
    """Reads a dataset; standarize missing values, standarize col names.

    Column Names changed to represent their type
        1. if str: col1 -> str_col1
        2. if int: col1 -> int_col1
        3. if float: col1 -> float_col1

    Args:
        dataset: Path to the dataset file.
        potential_missing: List of values to be treated as missing.
        format: File format: 'csv', 'tsv', or 'xlsx'.

    Returns:
        A pandas DataFrame with 
        standardized missing values, and standarized col names.
    """
    if format == "csv":
        df = pd.read_csv(dataset)
    elif format == "tsv":
        df = pd.read_csv(dataset, sep="\t")
    elif format == "xlsx":
        df = pd.read_excel(dataset)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Replace missing values with the standardized format
    df = df.replace(potential_missing, np.nan)


    # rename cols based on their dtype

    new_cols:dict[str, Any]={}

    for col in df.columns:
        dtype_str=str(df[col].dtype)
        dtype_prefix=dtype_str.split("[")[0].split()[0]
        new_cols[col]=f"{dtype_prefix}_{col}"

    df.rename(columns=new_cols, inplace=True)

    return df

# test function
df_standarized=EHR_standardize(dataset="diabetic_data.csv", 
                potential_missing=["NaN", "NA", "?"], format="csv")

# verify "?" is gone
print("Number of ?:",(df_standarized == "?").sum().sum())


def EHR_preprocessing(df:pd.DataFrame, 
    remove: bool=True, 
    impute:str|bool=False
) -> pd.DataFrame:
    """Takes in a pandas dataframe, and outputs a processed dataframe.

    Expects na format to be np.nan or pd.NA

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

    Returns:
        A pandas DataFrame with rows with missing values removed or replaced.
        
    """   
    df= df.copy()
       
    # deal with imputation
    if impute is not False:
        numeric_cols = df.select_dtypes(include=['number']).columns
        if impute == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif impute == "largest":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].max())
        elif impute == "smallest":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].min())
        elif impute == "median":
            df[numeric_cols] =(
                df[numeric_cols].fillna(df[numeric_cols].median())  
            )
        else:
            raise ValueError(f"Unknown imputation strategy: {impute}")
    
    # if we use this all rows will be deleted, each row has at least one NA
    # Handle removal of remaining NAs if requested
    # if remove:
    #     df = df.dropna()
    
    return df

# to test that na's are properly removed
df_processed=EHR_preprocessing(df_standarized, 
                               remove=True, impute="mean")
print("\nNumber of remaining NA:", df_processed.isna().sum().sum())

# check na structure
print("Original shape:", df_standarized.shape)



# Check if all rows have at least one NaN 
# (causing dropna() to remove everything)
print("\nRows with at least one NaN:", df_standarized.isna().any(axis=1).sum())

def EHR_select_cols(df: pd.DataFrame, variable_list:list[str]) -> pd.DataFrame:
    """Select specificied columns from the dataframe.

    Args:
        df: Input DataFrame
        
        variable_list: list of column names to select
    """
    return df[variable_list]

# check select is working correctly
df_subset = EHR_select_cols(df_processed,
                             variable_list=["object_weight", "object_race"])
print(df_subset.head())


def EHR_select_values(df: pd.DataFrame, 
    variable:str, 
    int_range: Optional[str]= None,
    values: Optional[list[str]]= None,
    include_endpoints: bool= False) -> pd.DataFrame:
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

    """
    # check for valid input
    if variable not in df.columns:
        raise ValueError(f"Column '{variable} not found in DataFrame")
    
    # handle double input
    if int_range and values is not None:
        raise ValueError("Can only choose range or values.")
    
    # handle numeric range
    if int_range is not None:
        try:
            parts = [float(x.strip()) for x in int_range.split(",")]
            if len(parts) != 2:
                raise ValueError("Range string must contain exactly " 
                "two numbers separated by a comma.")
            
            lower, upper = sorted(parts)

            if include_endpoints:
                return df[(df[variable] >= lower) & (df[variable] <= upper)]
            else:
                return df[(df[variable] > lower) & (df[variable] < upper)]
        except ValueError as err :
            raise ValueError("Invalid range format. " \
            "Use 'lower,upper' with numeric values.")  from err   
    
    if values is not None:
        return df[df[variable].isin(values)]

    raise ValueError("Must provide either 'int_range' or 'values'.")

df_subset_range=EHR_select_values(df_processed, 
     variable="object_race", 
     values=["African American", "Caucasian"])
print(df_subset_range.head())