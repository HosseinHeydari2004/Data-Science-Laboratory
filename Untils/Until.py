import streamlit as st
from pandas import DataFrame
from ydata_profiling import ProfileReport


@st.cache_data
def Auto_EDA(data: DataFrame):
    profile = ProfileReport(data, title="Data Profiling Report", explorative=True, minimal=True)
    profile.to_file("report_Auto_EDA/report.html")
    st.success(f"file save to 'report_Auto_EDA/report.html'", icon="✅")
    return profile.to_html()
