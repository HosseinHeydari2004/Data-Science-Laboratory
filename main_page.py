import streamlit as st

st.set_page_config(page_title="Data Science Lab", layout="wide")

home_page = st.Page("app/pages/home_page.py", title="Welcome", icon="🏠", default=True)
viz_page = st.Page("app/pages/page_2.py", title="Exploratory Data Analysis", icon="📊")
model_page = st.Page("app/pages/page_3.py", title="Model Training", icon="🔧")

pg = st.navigation([home_page, viz_page, model_page])
pg.run()
