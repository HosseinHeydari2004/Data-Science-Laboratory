import pandas as pd
import streamlit as st

st.write("# 🤖 Welcome to the Data Science and Machine Learning Laboratory!")
st.markdown(
    """
     ### 📊 From Raw Data to Strategic Decisions in Seconds
    This tool helps you analyze even the most complex data variables and predict the future – all without writing a single line of code.

    ---

    #### 🚀 What Can You Do?
    - Smart Data Cleaning: Automatically identifies outliers and missing values.
    - Interactive Visualization: Create advanced charts with just a few clicks.
    - Machine Learning: Quickly train regression and classification models on your data.

    ---

    #### 🛠️ Getting Started
    1. From the menu on the right, upload your data file (CSV or Excel).
    2. Select your desired parameters in the settings section.
    3. View analysis and prediction outputs in the tabs above.

    ---
    ## Ready to get started? Upload your first file from the below!
    """
)

uploaded_file = st.file_uploader("Select a CSV or Excel file:", type=['csv', 'xlsx'], key="home_page_uploader")

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state['df'] = df
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        st.balloons()
    except Exception as e:
        st.error(f"Error reading file: {e}")
