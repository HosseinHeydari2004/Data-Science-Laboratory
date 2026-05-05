import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler


class EDA:
    @classmethod
    def missing_value_report(cls, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(
            data={
                "column": data.columns,
                "missing value": data.isna().sum(),
                "percent missing values": round((data.isna().sum() / len(data)) * 100, 3),
            }
        ).reset_index(drop=True)
        return df

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
        return d.style.highlight_max(
            color="red", subset=["missing values", "percent missing values(%)"]
        ).format({
            "percent missing values(%)":"{:.2f}%"
        })
