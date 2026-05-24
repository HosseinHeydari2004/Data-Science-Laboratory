import os

import streamlit as st
from pandas import DataFrame
from ydata_profiling import ProfileReport


@st.cache_data
def Auto_EDA(data: DataFrame):
    profile = ProfileReport(data, title="Data Profiling Report", explorative=True, minimal=True)
    profile.to_file("report_Auto_EDA/report.html")
    st.success(f"file save to 'report_Auto_EDA/report.html'", icon="✅")
    return profile.to_html()


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
