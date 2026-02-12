from pathlib import Path

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
    return missing_df


def describe_data(df: pd.DataFrame, include: str = "all"):
    """This function returns a general description of the data"""
    return df.describe(include=include).T


def unique_values_data(df: pd.DataFrame, col: str):
    """Returns unique values"""
    return df[col].unique()


def detect_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Returns outliers using the IQR method"""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(col < lower) | (col > upper)]


def remove_missing_values(
        df: pd.DataFrame,
        threshold: float = 30.0,
        save_path: str = "C:\Users\hosse\OneDrive\Desktop\AI Source\پروژه ها و نمونه کار ها\پروژه ها+ نمونه کار ها\ml_lab\Data\processed"
):
    def get_high_missing_columns():
        """
        Returns columns with missing percentage above threshold.
        """
        missing_percentage = df.isna().mean() * 100
        high_missing = missing_percentage[missing_percentage > threshold]

        return high_missing

    df_clean = df.dropna()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(save_path, index=False)
