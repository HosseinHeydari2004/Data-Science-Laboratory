import streamlit as st

st.set_page_config(page_title="Data Science Lab", layout="wide")

home_page = st.Page("Main/home_page.py", title="Welcome", icon="🏠", default=True)
viz_page = st.Page("Main/page_2.py", title="Exploratory Data Analysis", icon="📊")

pg = st.navigation([home_page, viz_page])
pg.run()
