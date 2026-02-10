import pandas as pd


def detect_missing_values(df: pd.DataFrame):
    """
    This function returns missing values for each column.
    """
    missing_df = pd.DataFrame(
        data={
            "Missing_value_count": df.isna().sum(),
            "Missing_value_Percentage": df.isna().mean() * 100
        }
    )


def describe_data(df: pd.DataFrame, include: str = "all"):
    """This function returns a general description of the data"""
    return df.describe(include=include).T
