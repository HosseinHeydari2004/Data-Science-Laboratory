import os

import pandas as pd
import streamlit as st


def save_Auto_EDA(input_data, output_folder="report_Auto_EDA"):
    try:
        df = pd.read_csv(input_data)
        st.success(f"✅ File uploaded successfully: {input_data}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        base_name = os.path.basename(input_data).split('.')[0]
        output_file_path = os.path.join(output_folder, f"{base_name}_report.html")
        pass
    except:
        pass
