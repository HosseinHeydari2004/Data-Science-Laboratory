from pathlib import Path

import pandas as pd


def read_data(path: str):
    try:
        file_path = Path(path)

        if not file_path.exists():
            raise FileExistsError("File not found")

        if file_path.suffix == ".csv":
            return pd.read_csv(path)
        else:
            raise ValueError("Please upload the csv file")
    except Exception as E:
        raise "Error reading file"



