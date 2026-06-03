import os

from pandas import DataFrame


def save_data(data: DataFrame, file_name: str = "processed_data.csv"):
    try:
        folder_path = "Data\Proccessed_data"
        os.makedirs(os.path.dirname(folder_path), exist_ok=True)
        save_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.csv'):
            data.to_csv(save_path, index=False)
        elif file_name.endswith(('.xlsx', '.xls')):
            data.to_excel(save_path, index=False)
        else:
            data.to_csv(save_path, index=False)
        return True, save_path
    except Exception as e:
        return False, str(e)
