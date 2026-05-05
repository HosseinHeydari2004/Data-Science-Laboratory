import streamlit as st

from Core.preprocessor import EDA

st.title(" Exploratory Data Analysis (EDA)")

if 'df' in st.session_state:
    df = st.session_state['df']
    with st.expander("View dataset"):
        st.dataframe(df)
    with st.expander("information dataset"):
        st.dataframe(EDA.information_data(data=df))
        if EDA.find_high_col_missing_values(data=df).keys():
            st.warning(
                f"The columns '{','.join(list(EDA.find_high_col_missing_values(data=df).keys()))}' "
                f"have a large number of missing values", width=500, icon="⚠️"
            )
