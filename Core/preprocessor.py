import numpy as np
import pandas as pd


class MissingValue:
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
    def columns(cls, data:pd.DataFrame)->np.ndarray:
        pass
