from typing import Optional, Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.api.types import is_datetime64_any_dtype
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
        return data.describe(include="all").rename({"top": "mode"})

    @classmethod
    def check_date_object(cls, data: pd.DataFrame) -> bool | tuple[bool, str]:
        date_cols = [c for c in data.columns if c in ["date", "datetime", "Date", "Datetime"]]
        if not date_cols or is_datetime64_any_dtype(data[date_cols[0]]):
            return False
        else:
            return True, date_cols[0]

    @classmethod
    def change_dtype_datetime64(cls, data: pd.DataFrame) -> pd.DataFrame:
        ch, data_col = EDA.check_date_object(data=data)
        if ch:
            data[data_col] = pd.to_datetime(data[data_col], errors="coerce")
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

    @classmethod
    def delete_columns(cls, data: pd.DataFrame, col: str | list) -> pd.DataFrame | pd.Series:
        return data.drop(columns=col)

    @classmethod
    def show_first_5_row(cls, data: pd.DataFrame) -> pd.DataFrame:
        return data.head(5)

    @classmethod
    def show_last_5_row(cls, data: pd.DataFrame) -> pd.DataFrame:
        return data.tail(5)

    @classmethod
    def show_specific_row(cls, data: pd.DataFrame, index: int = 0) -> pd.DataFrame | pd.Series:
        return data.iloc[index]

    @classmethod
    def show_random_sample_rows(cls, data: pd.DataFrame, n: int = 5) -> pd.DataFrame | pd.Series:
        return data.sample(n=n)

    @classmethod
    def show_specific_column(cls, data: pd.DataFrame, col_name: str) -> pd.Series:
        return data[col_name]

    @classmethod
    def check_dtype_column(cls, data: pd.DataFrame | pd.Series, col: object) -> int:
        category = data.select_dtypes(include=["object"])
        numeric = data.select_dtypes(include=["number"])
        if col in category:
            return 0
        elif col in numeric:
            return 1

    @classmethod
    def select_manual_data(
            cls,
            data: pd.DataFrame,
            rows: tuple[int, int] = (0, 5),
            columns: Optional[tuple[int, int]] = None,
            mode: str = "Multiple rows and columns",
            column_name: Optional[str] | object = None,
            row_index: Optional[int] = None,
            value: Any = None,
            query: str = ""
    ) -> Any:

        if mode == "Multiple rows and columns":
            if columns is None:
                return data.iloc[rows[0]:rows[1], :]
            return data.iloc[rows[0]:rows[1], columns[0]:columns[1]]

        elif mode == "Multiple rows and one column":
            if isinstance(column_name, str):
                return data[column_name].iloc[rows[0]:rows[1]]
            return data.iloc[rows[0]:rows[1], 0]

        elif mode == "one row and Multiple columns":
            if columns is None:
                return data.iloc[row_index, :]
            return data.iloc[row_index, columns[0]:columns[1]]
        elif mode == "filter by value":
            return data[data[column_name] == value]
        elif mode == "search text":
            return data[data[column_name].astype(str).str.contains(str(value), case=False, na=False)]
        elif mode == "query":
            qy = data.query(expr=query)
            return qy


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
    def show_missing_values(cls, data: pd.DataFrame, reset_index: bool = False) -> pd.Series | pd.DataFrame:
        if reset_index:
            return data[data.isna().any(axis=1)].reset_index(drop=True)
        else:
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
            cls, data: pd.DataFrame, col: pd.DataFrame | str | object,
            method: str | object = "IQR",
            threshold: int = 3
    ) -> pd.DataFrame:
        if method == "IQR":
            Q1 = data[col].quantile(q=0.25)
            Q3 = data[col].quantile(q=0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            return outliers[col]
        elif method == "Z_score":
            z = np.abs(stats.zscore(data[col]))
            outliers_index = np.where(z > threshold)
            return data.iloc[outliers_index][col]

    @classmethod
    def delete_outliers(
            cls, data: pd.DataFrame, index_outliers: Any
    ) -> pd.DataFrame:
        return data.drop(index=index_outliers, errors='ignore').reset_index(drop=True)


class data_manipulation:
    @classmethod
    def delete_row(
            cls, data: pd.DataFrame, row_index: int | object | pd.Series = None
    ) -> DataFrame | None | Exception:
        if row_index not in data.index:
            raise ValueError(f"row {row_index} not founded!")
        return data.drop(index=row_index).reset_index(drop=True)

    @classmethod
    def delete_rows(
            cls, data: pd.DataFrame, rows_index: tuple[int, int] = None
    ) -> pd.DataFrame | None | Exception:
        start, end = rows_index
        indices_to_drop = list(range(start, end + 1))
        if not all(i in data.index for i in indices_to_drop):
            raise ValueError(f"rows {rows_index} not founded!")
        return data.drop(index=indices_to_drop).reset_index(drop=True)

    @classmethod
    def delete_column(cls, data: pd.DataFrame, col: str | pd.Series | object) -> pd.DataFrame | Exception:
        if col in data.columns:
            return data.drop(columns=col)
        else:
            raise ValueError(f"column {col} not founded!")

    @classmethod
    def delete_columns(cls, data: pd.DataFrame, list_col: list[str] | object) -> pd.DataFrame:
        missing_cols = [col for col in list_col if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {', '.join(missing_cols)}")
        return data.drop(columns=list_col)

    @classmethod
    def change_col_name(
            cls, data: pd.DataFrame, col_name_last: str | object,
            col_name_new: str | object
    ) -> pd.DataFrame:
        return data.rename(columns={col_name_last: col_name_new}, errors="raise")

    @classmethod
    def change_dtype(
            cls, data: pd.DataFrame, col: str | object, dtype: str | object | np.dtype
    ) -> pd.DataFrame:
        if col in data.columns:
            try:
                data_copy = data.copy()
                data_copy[col] = data[col].astype(dtype=dtype)
                return data_copy
            except Exception as e:
                raise ValueError(f"Error while converting column ‘{col}’ to ‘{dtype}’: {e}")
        else:
            raise ValueError(f"column {col} not in data")


