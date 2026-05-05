import streamlit as st

from Core.preprocessor import EDA

st.title("Exploratory Data Analysis (EDA)")

if 'df' in st.session_state:
    df = st.session_state['df']
    with st.expander("View dataset"):
        st.dataframe(df)
    with st.expander("information dataset"):
        st.dataframe(EDA.information_data(data=df))
        threshold = st.slider("Select Missing Value Threshold (%)", 0, 100, 30)
        critical_missing_cols = EDA.find_high_col_missing_values(data=df, threshold=threshold)
        if critical_missing_cols.keys():
            st.warning(
                f"The columns '{','.join(list(EDA.find_high_col_missing_values(data=df).keys()))}' "
                f"have a large number of missing values", icon="⚠️"
            )
            critical_missing = EDA.report_high_missing_value(data=df, threshold=threshold)
            if critical_missing:
                st.warning(f"High number of missing values: {critical_missing[2]}", icon="⚠️")


