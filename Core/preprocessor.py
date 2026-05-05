import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler


class EDA:
    @classmethod
    def check_unique(cls, data: pd.DataFrame, select_columns: str) -> np.ndarray:
        return data[select_columns].unique()

    @classmethod
    def list_columns(cls, data: pd.DataFrame) -> list:
        return list(data.columns)

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

        def highlight_good_values(s):
            return ['background-color: blue' if v == s.min() else '' for v in s]

        return d.style.format({
            "percent missing values(%)": "{:.2f}%",
            "memory usage (KB)": "{:.2f} KB"
        }).apply(highlight_bad_values, subset=["missing values", "percent missing values(%)"]) \
            .apply(highlight_good_values, subset=["missing values", "percent missing values(%)"])

    @classmethod
    def remove_missing_values(cls):
        pass

    @classmethod
    def find_high_col_missing_values(cls, data: pd.DataFrame, threshold: int = 30) -> dict:
        percent_missing = (data.isna().sum() / len(data)) * 100
        return percent_missing[percent_missing > threshold].to_dict()

    @classmethod
    def report_high_missing_value(cls, data: pd.DataFrame, threshold: int = 30):
        if ((data.isna().sum() / len(data)) * 100) > threshold:
            return True
        return False
