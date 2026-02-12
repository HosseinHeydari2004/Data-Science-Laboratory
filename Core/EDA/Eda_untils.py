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


def unique_values_data(df: pd.DataFrame, col: str):
    return df[col].unique()


def detect_outliers(df: pd.DataFrame, col: str)->pd.DataFrame:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(col < lower) | (col > upper)]
