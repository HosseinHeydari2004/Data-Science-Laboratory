import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
from scipy import stats
from sklearn.impute import SimpleImputer


class EDA:
    @classmethod
    def check_unique(cls, data: pd.DataFrame, select_column: str) -> np.ndarray:
        return data[select_column].unique()

    @classmethod
    def list_columns(cls, data: pd.DataFrame) -> list:
        return data.columns.to_list()

    @classmethod
    def information_data(cls, data: pd.DataFrame) -> Styler:
        d = pd.DataFrame(
            data={
                "columns": data.columns,
                "data type": data.dtypes.values,
                "missing values": data.isna().sum(),
                "percent missing values(%)": (data.isna().sum() / len(data)) * 100,
                "memory usage": data.memory_usage(deep=True, index=False)
            }
        ).reset_index(drop=True)

        def highlight_bad_values(s):
            if s.max() == 0:
                return ['' for _ in s]
            return ['background-color: red' if v == s.max() else '' for v in s]

        return d.style.format({
            "percent missing values(%)": "{:.2f}%",
            "memory usage (KB)": "{:.2f} KB"
        }).apply(highlight_bad_values, subset=["missing values", "percent missing values(%)"])

    @classmethod
    def describe_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        return data.describe(include="all").rename({"top": "mode"}).T

    @classmethod
    def check_date_in_data(cls, data: pd.DataFrame) -> bool | tuple[bool, list]:
        date_cols = [c for c in data.columns if c in ["date", "datetime", "Date", "Datetime"]]
        if date_cols:
            target_col = date_cols[0]
            data[target_col] = pd.to_datetime(data[target_col])
            return True, date_cols
        return False

    @classmethod
    def change_dtype_datetime64(cls, data: pd.DataFrame) -> pd.DataFrame:
        if EDA.check_date_in_data(data=data)[0]:
            data[EDA.check_date_in_data(data=data)[1][0]] = pd.to_datetime(
                data[EDA.check_date_in_data(data=data)[1][0]])
            return data

    @classmethod
    def get_duplicate(cls, data: pd.DataFrame) -> int | bool:
        if data.duplicated().sum() > 0:
            return data.duplicated().sum()
        else:
            return False

    @classmethod
    def show_duplicate_values(cls, data: pd.DataFrame) -> pd.Series | pd.DataFrame:
        return data[data.duplicated()]

    @classmethod
    def delete_duplicate_values(cls, data: pd.DataFrame) -> pd.DataFrame:
        return data.drop_duplicates()

    @classmethod
    def detect_numeric_type(cls, data: pd.DataFrame):
        numeric = data.select_dtypes(include="number").columns.to_list()
        return numeric

    @classmethod
    def detect_object_type(cls, data: pd.DataFrame):
        return data.select_dtypes(include="object").columns.to_list()

    @classmethod
    def detect_time_type(cls, data: pd.DataFrame):
        return data.select_dtypes(include=["datetime"]).columns.to_list()

    @classmethod
    def all_correlation(cls, data: pd.DataFrame):
        return data.corr(numeric_only=True)

    @classmethod
    def col_correlation(cls, data: pd.Series, col1: pd.Series, col2: pd.Series):
        return data[col1].corr(other=data[col2])


class handle_MissingValue:
    """
    A utility class to analyze and handle missing values in pandas DataFrames.

    This class provides streamlined methods to detect null values and apply
    various imputation or deletion strategies to maintain data integrity
    for machine learning pipelines.
    """

    @classmethod
    def check_missing_values(cls, data: pd.DataFrame):
        if data.isna().any(axis=1).sum() > 0:
            return True
        return False

    @classmethod
    def report_high_missing_value(
            cls, data: pd.DataFrame, threshold: int = 30
    ) -> tuple[bool, float | int, int] | bool:

        percent_missing = (data.isna().any(axis=1).sum() / len(data)) * 100
        total_missing_value = data.isna().any(axis=1).sum()
        if percent_missing > threshold:
            return True, percent_missing, total_missing_value
        return False

    @classmethod
    def remove_missing_values(cls, data: pd.DataFrame, axis: str = "row"):
        if axis == "row":
            return data.dropna(axis=0)
        else:
            return data.dropna(axis=1)


    @classmethod
    def find_high_col_missing_values(cls, data: pd.DataFrame, threshold: int = 30) -> dict:
        percent_missing = (data.isna().sum() / len(data)) * 100
        return percent_missing[percent_missing > threshold].to_dict()

    @classmethod
    def show_missing_values(cls, data: pd.DataFrame) -> pd.Series | pd.DataFrame:
        return data[data.isna().any(axis=1)]

    @classmethod
    def fill_SimpleImputer(
            cls, x: pd.DataFrame | pd.Series = None, strategy="mean", fill=0
    ) -> pd.Series | pd.DataFrame:
        if strategy == "constant":
            imputer = SimpleImputer(strategy="constant", fill_value=fill)
            x_filled = imputer.fit_transform(X=x)
            return x_filled
        else:
            imputer = SimpleImputer(strategy=strategy)
            x_filled = imputer.fit_transform(X=x)
            return x_filled


class handle_outliers:
    """"""

    @classmethod
    def detect_outliers(
            cls, data: pd.DataFrame, col: pd.DataFrame, method: str = "IQR",
            threshold: int = 3, reset_index: bool = False
    ) -> pd.DataFrame:
        if method == "IQR":
            Q1 = data[col].quantile(q=0.25)
            Q3 = data[col].quantile(q=0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            if reset_index:
                return outliers[col].reset_index(drop=True)
            else:
                return outliers[col]
        elif method == "Z_score":
            z = np.abs(stats.zscore(data[col]))
            outliers_index = np.where(z > threshold)
            if reset_index:
                return data.iloc[outliers_index][col].reset_index(drop=True)
            else:
                return data.iloc[outliers_index][col]
