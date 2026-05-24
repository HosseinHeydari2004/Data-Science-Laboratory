import os

import pandas as pd
import streamlit as st

st.write("# 🤖 Welcome to the Data Science and Machine Learning Laboratory!")
st.markdown(
    """
     ### 📊 From Raw Data to Strategic Decisions in Seconds
    This tool helps you analyze even the most complex data variables and predict the future. 
     
     **all without writing a single line of code.**

    ---

    #### 🚀 What Can You Do?
    - Smart Data Cleaning: Automatically identifies outliers and missing values.
    - Interactive Visualization: Create advanced charts with just a few clicks.
    - Machine Learning: Quickly train regression and classification models on your data.

    ---

    #### 🛠️ Getting Started
    1. From the menu on the below, upload your data file (CSV or Excel).
    2. In the visualization page, you can modify your data, display it using a chart, and view your data.
    3. Select your desired parameters in the model page.
    4. View analysis and prediction outputs in the model page.

    ---
    ## Ready to get started? Upload your first file from the below!
    """
)

SAVE_PATH = "Data/Main_Data"
os.makedirs(SAVE_PATH, exist_ok=True)

uploaded_file = st.file_uploader(
    "Select a CSV or Excel file:",
    type=['csv', 'xlsx'],
    key="home_page_uploader",
    max_upload_size=5,
    help="upload your data up 5MB",
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.session_state['df'] = df
    st.success(f"File '{uploaded_file.name}' loaded successfully!", icon="✅")
    if st.button("save data in local", icon="💾", help=f"save your data in '{SAVE_PATH}'"):
        try:
            full_path = os.path.join(SAVE_PATH, uploaded_file.name)

            uploaded_file.seek(0)
            with open(full_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File saved to: '{full_path}'", icon="✅")
        except Exception as e:
            st.error(f"Error saving file: {e}")
