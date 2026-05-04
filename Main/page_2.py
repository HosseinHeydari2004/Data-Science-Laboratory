import streamlit as st

from Core.preprocessor import EDA

st.title(" Exploratory Data Analysis (EDA)")

if 'df' in st.session_state:
    df = st.session_state['df']
    with st.expander("View dataset"):
        st.dataframe(df)
    with st.expander("information dataset"):
        st.write(EDA.information_data(data=df))
