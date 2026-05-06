import streamlit as st

from Core.preprocessor import EDA

st.title("Exploratory Data Analysis (EDA)")

if 'df' in st.session_state:
    df = st.session_state['df']
    with st.expander("View dataset"):
        st.dataframe(df)
    with st.expander("information dataset"):
        st.dataframe(EDA.information_data(data=df))
        if EDA.count_missing_values(data=df):
            threshold = st.slider("Select Missing Value Threshold (%)", 0, 100, 30)
            critical_missing_cols = EDA.find_high_col_missing_values(data=df, threshold=threshold)
            if critical_missing_cols.keys():
                st.warning(
                    f"The columns '{','.join(list(critical_missing_cols.keys()))}' "
                    f"have a large number of missing values", icon="⚠️"
                )
            critical_missing = EDA.report_high_missing_value(data=df, threshold=threshold)
            if critical_missing:
                st.warning(f"High number of missing values: {critical_missing[2]}", icon="⚠️")
            option = st.selectbox(
                "Do you want to delete rows or columns?",
                ("delete row", "delete col")
            )
            if st.button("delete missing value"):
                if option == "delete row":
                    df = EDA.remove_missing_values(data=df, axis=0)
                    st.session_state['df'] = df
                    st.success("✅ All missing values were deleted!")
                    st.rerun()
                elif option == "delete col":
                    st.warning("Deleting in column mode will delete the columns")
                    df = EDA.remove_missing_values(data=df, axis=1)
                    st.session_state['df'] = df
                    st.success("✅ All missing values were deleted!")
                    st.rerun()

        if EDA.check_date_in_data(data=df):
            st.warning(f"⚠️The date column is object")
            if st.button("change to Datetime"):
                df = EDA.change_dtype_datetime64(data=df)
                st.session_state['df'] = df
                st.success("✅ The date column was successfully updated!")
                st.rerun()
        duplicate = EDA.get_duplicate(data=df)
        if duplicate:
            threshold_duplicate = st.slider("Select duplicate Threshold", 0, len(df), 5)
            if duplicate > threshold_duplicate:
                st.warning(f"There are a lot of duplicate values")
            else:
                st.warning(f"Your data contains duplicate values")

    with st.expander("describe data"):
        describe = EDA.describe_data(data=df)
        st.dataframe(describe)
    with st.expander("unique values"):
        columns = EDA.list_columns(data=df)
        select_columns = st.selectbox(
            label="Please select the desired column",
            options=columns
        )
        st.dataframe(EDA.check_unique(data=df, select_column=select_columns))
