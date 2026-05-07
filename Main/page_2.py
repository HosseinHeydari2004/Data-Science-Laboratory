import streamlit as st

from Core.preprocessor import MissingValue, EDA, handle_outliers

st.title("Exploratory Data Analysis (EDA)")

if "success_msg" in st.session_state:
    st.success(st.session_state["success_msg"])
    del st.session_state["success_msg"]

if 'df' in st.session_state:
    df = st.session_state['df']
    with st.expander("view dataset"):
        st.dataframe(df)
    with st.expander("information dataset"):
        st.dataframe(EDA.information_data(data=df))
        missing_value = MissingValue.check_missing_values(data=df)
        if missing_value:
            threshold = st.slider("Select Missing Value Threshold (%)", 0, 100, 30)
            st.warning(
                """Important Note:
                If the count of missing values is below the threshold, rows should be dropped.
                Otherwise, columns should be dropped.
                """, icon="⚠️"
            )
            critical_missing_cols = MissingValue.find_high_col_missing_values(data=df, threshold=threshold)
            if critical_missing_cols.keys():
                st.warning(
                    f"The columns '{','.join(list(critical_missing_cols.keys()))}' "
                    f"have a large number of missing values", icon="⚠️"
                )
            critical_missing = MissingValue.report_high_missing_value(data=df, threshold=threshold)
            if critical_missing:
                st.warning(f"High number of missing values: {critical_missing[2]}", icon="⚠️")

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
            if st.button("delete duplicate values"):
                pass

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

    with st.expander("view outlier"):
        outliers_col_selectbox = st.selectbox(
            "Please select the desired column",
            options=EDA.detect_numeric_type(data=df)
        )
        method_selectbox = st.selectbox(
            "Please select your preferred method",
            options=["IQR", "Z_score"]
        )
        outliers = handle_outliers.detect_outliers(
            data=df, col=outliers_col_selectbox, method=method_selectbox
        )
        if len(outliers) > 0:
            if method_selectbox == "IQR":
                st.dataframe(handle_outliers.detect_outliers(data=df, col=outliers_col_selectbox))
            elif method_selectbox == "Z_score":
                threshold_outliers = st.slider(
                    "Select outliers Threshold",
                    1.0, 4.0, 3.0
                )
                st.dataframe(
                    handle_outliers.detect_outliers(data=df, col=outliers_col_selectbox, method=method_selectbox,
                                                    threshold=threshold_outliers)
                )
        else:
            st.warning(
                f"In the '{method_selectbox}' method, there are no outliers in '{outliers_col_selectbox}'"
            )
