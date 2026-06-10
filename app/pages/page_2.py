import streamlit as st

from Core.eda import handle_MissingValue, EDA, handle_outliers, data_manipulation
from Untils.Until import save_data
from components.charts import seaborn_chart, plotly_charts

st.title("📊 Exploratory Data Analysis (EDA)")
st.markdown(
    """
    
This page is the **Exploratory Data Analysis (EDA)** section of the Streamlit app, used to inspect, understand, clean, and visualize datasets before building machine learning models.
All functionalities are organized into **expanders** so users can focus on specific analysis tasks when needed.

### Sections

- **View dataset**
  - Displays the dataset in multiple ways.
  - Supports viewing the entire dataset, first/last rows, specific rows,
    random samples, individual columns, filtered records, and custom queries.
  - Helps users quickly inspect the structure and content of the data.

- **Information dataset**
  - Provides general information about the dataset.
  - Displays column names, data types, missing values, duplicate records,
    and overall dataset statistics.
  - Includes tools for handling missing values, removing duplicates,
    and converting date columns to datetime format.

- **Describe data**
  - Generates descriptive statistics for numerical features.
  - Includes count, mean, standard deviation, minimum, maximum,
    and quartile values.
  - Helps identify the distribution and scale of variables.

- **Unique values**
  - Displays unique values and their frequencies for selected columns.
  - Useful for understanding categorical variables and detecting
    unexpected or inconsistent values.

- **View outlier**
  - Detects outliers using both **IQR** and **Z-Score** methods.
  - Allows users to inspect detected outliers before removal.
  - Supports deleting outlier records directly from the dataset.

- **Data manipulation**
  - Provides tools for modifying and cleaning the dataset interactively.
  - Supports deleting single or multiple rows, removing columns,
    renaming columns, and changing data types.
  - Allows users to update the dataset directly from the interface
    without writing code.

- **Visualization**
  - Provides both static and interactive visualization tools.
  - Supports multiple chart types including:
    - Histogram
    - KDE Plot
    - Box Plot
    - Count Plot
    - Scatter Plot
    - Violin Plot
    - Line Plot
    - Correlation Heatmap
  - Interactive visualizations are powered by Plotly and support
    customization of colors, labels, scales, templates, and figure size.
  - Helps users explore distributions, trends, relationships,
    correlations, and patterns within the dataset.

### Dataset Persistence

- **Save Data**
  - Saves the modified dataset after cleaning, transformation,
    and exploratory analysis.
  - Allows users to preserve their changes for future modeling
    and machine learning workflows.
"""

)

if 'df' in st.session_state:
    df = st.session_state['df']
    with st.expander("view dataset"):
        select_show_data_mode = st.selectbox(
            "select mode for show data",
            options=[
                "entire dataFrame",
                "show the first 5 rows",
                "show the last 5 rows",
                "show a specific row",
                "show a random sample of rows",
                "show a specific column",
                "show rows from x to y and columns from a to b",
                "filter by value (column == x)",
                "search text in column",
                "search with query"
            ], index=0, key="select_show_data_mode"
        )
        if select_show_data_mode == "entire dataFrame":
            st.dataframe(data=df)
        elif select_show_data_mode == "show the first 5 rows":
            st.dataframe(EDA.show_first_5_row(data=df))
        elif select_show_data_mode == "show the last 5 rows":
            st.dataframe(EDA.show_last_5_row(data=df))
        elif select_show_data_mode == "show a specific row":
            st.info(f"Please select a range from index 0 to {len(df)}.", icon="ℹ️")
            select_specific_row = st.number_input(
                "enter you specific row",
                value=0, key="select_specific_row"
            )
            try:
                st.dataframe(EDA.show_specific_row(data=df, index=select_specific_row))
            except:
                st.error(f"The index you entered is out of bounds. Valid range is 0 to {len(df)}.", icon="⚠️")
        elif select_show_data_mode == "show a random sample of rows":
            st.info(f"Please select a range from index 0 to {len(df)}.", icon="ℹ️")
            select_total_n = st.number_input(
                "enter total number for show a random sample of rows",
                value=1, key="select_total_n"
            )
            try:
                st.dataframe(EDA.show_random_sample_rows(data=df, n=select_total_n))
            except:
                st.error(f"The index you entered is out of bounds. Valid range is 0 to {len(df)}.", icon="⚠️")
        elif select_show_data_mode == "show a specific column":
            select_columns_show = st.selectbox(
                "select columns to show",
                options=EDA.list_columns(data=df), key="select_columns_show"
            )
            st.dataframe(EDA.show_specific_column(data=df, col_name=select_columns_show))
        elif select_show_data_mode == "show rows from x to y and columns from a to b":
            select_mode_show_data_manual = st.selectbox(
                "select mode",
                options=[
                    None,
                    "Multiple rows and columns",
                    "Multiple rows and one column",
                    "one row and Multiple columns"
                ], key="select_mode_show_data_manual"
            )
            if select_mode_show_data_manual == "Multiple rows and columns":
                select_row_start = st.number_input(
                    "start from row",
                    min_value=0, max_value=len(df), value=0, key="select_row_start"
                )
                select_row_up = st.number_input(
                    "up to row",
                    min_value=0, max_value=len(df), value=1, key="select_row_up"
                )
                select_col_start = st.number_input(
                    "start from column",
                    min_value=0, max_value=len(EDA.list_columns(data=df)), value=None,
                    key="select_col_start"
                )
                select_col_up = st.number_input(
                    "up to column",
                    min_value=0, max_value=len(EDA.list_columns(data=df)), value=None,
                    key="select_col_up"
                )
                if (select_col_start is None) and (select_col_up is None):
                    st.dataframe(EDA.select_manual_data(
                        data=df, rows=(select_row_start, select_row_up),
                    ))
                else:
                    st.dataframe(EDA.select_manual_data(
                        data=df, rows=(select_row_start, select_row_up),
                        columns=(select_col_start, select_col_up)
                    ))
            elif select_mode_show_data_manual == "Multiple rows and one column":
                select_row_start = st.number_input(
                    "start from row",
                    min_value=0, max_value=len(df), value=0, key="select_row_start"
                )
                select_row_up = st.number_input(
                    "up to row",
                    min_value=0, max_value=len(df), value=1, key="select_row_up"
                )
                select_col_name = st.selectbox(
                    "select column name to show",
                    options=EDA.list_columns(data=df), key="select_col_name"
                )
                st.dataframe(
                    EDA.select_manual_data(
                        data=df, rows=(select_row_start, select_row_up),
                        column_name=select_col_name, mode="Multiple rows and one column"
                    )
                )
            elif select_mode_show_data_manual == "one row and Multiple columns":
                select_row_index = st.number_input(
                    "select row",
                    min_value=0, max_value=len(df), value=0
                )
                select_col_start = st.number_input(
                    "start from column",
                    min_value=0, max_value=len(EDA.list_columns(data=df)), value=None,
                    key="select_col_start2"
                )
                select_col_up = st.number_input(
                    "up to column",
                    min_value=0, max_value=len(EDA.list_columns(data=df)), value=None,
                    key="select_col_up2"
                )
                if (select_col_start is None) and (select_col_up is None):
                    st.dataframe(EDA.select_manual_data(
                        data=df, row_index=select_row_index,
                        mode="one row and Multiple columns",

                    ))
                else:
                    st.dataframe(EDA.select_manual_data(
                        data=df, row_index=select_row_index,
                        columns=(select_col_start, select_col_up),
                        mode="one row and Multiple columns",

                    ))
        elif select_show_data_mode == "filter by value (column == x)":
            select_col_name = st.selectbox(
                "select column name",
                options=EDA.list_columns(data=df),
                key="select_col_name3"
            )
            if EDA.check_dtype_column(data=df, col=select_col_name) == 0:
                select_object_value = st.text_input(
                    "select value",
                    key="select_object_value"
                )
                st.dataframe(
                    EDA.select_manual_data(
                        data=df, mode="filter by value",
                        column_name=select_col_name, value=select_object_value
                    )
                )
            else:
                select_number_value = st.number_input(
                    "select value",
                    key="select_number_value"
                )
                st.dataframe(
                    EDA.select_manual_data(
                        data=df, mode="filter by value",
                        column_name=select_col_name, value=select_number_value
                    )
                )
        elif select_show_data_mode == "search text in column":
            select_col_object = st.selectbox(
                "select column name",
                options=EDA.list_columns(data=df),
                key="select_col_object", index=0
            )
            select_object_value = st.text_input(
                "select value",
                key="select_object_value2"
            )
            st.write(EDA.select_manual_data(
                data=df, column_name=select_col_object,
                value=select_object_value, mode="search text"
            ))
        elif select_show_data_mode == "search with query":
            enter_query = st.text_input(
                "enter your query",
            )
            try:
                st.dataframe(
                    EDA.select_manual_data(
                        data=df, query=enter_query, mode="query"
                    )
                )
            except Exception as E:
                st.error(f"{E}")

    with st.expander("information dataset"):
        st.dataframe(EDA.information_data(data=df))
        missing_value = handle_MissingValue.check_missing_values(data=df)
        if missing_value:
            threshold = st.slider("Select Missing Value Threshold (%)", 0, 100, 30)
            st.warning(
                """Important Note:
                If the count of missing values is below the threshold, rows should be dropped.
                Otherwise, columns should be dropped.
                """,
                icon="⚠️"
            )
            critical_missing_cols = handle_MissingValue.find_high_col_missing_values(data=df, threshold=threshold)
            if critical_missing_cols.keys():
                st.warning(
                    f"The columns '{','.join(list(critical_missing_cols.keys()))}' "
                    f"have a large number of missing values", icon="⚠️"
                )
            critical_missing = handle_MissingValue.report_high_missing_value(data=df, threshold=threshold)
            if critical_missing:
                st.warning(f"High number of missing values: {critical_missing[2]}", icon="⚠️")

            select_show_missing_value = st.selectbox(
                "do you show missing values",
                options=[False, True], index=0, key="select_show_missing_value"
            )
            if select_show_missing_value:
                select_reset_index = st.selectbox(
                    "do you reset index",
                    options=[False, True], index=0, key="select_reset_index2"
                )
                st.dataframe(
                    handle_MissingValue.show_missing_values(data=df, reset_index=select_reset_index)
                )
            select_axis = st.selectbox(
                "select delete by row or col",
                options=["row", "column"], index=0, key="select_axis"
            )
            if select_axis == "row":
                if st.button("delete missing values"):
                    df = handle_MissingValue.remove_missing_values(data=df, axis="row")
                    st.session_state['df'] = df
                    st.success("✅ Missing values removed successfully!")
                    st.rerun()
            elif select_axis == "column":
                select_col_to_delete = st.multiselect(
                    "select columns to delete",
                    options=critical_missing_cols.keys(), key="select_col_to_delete"
                )
                if st.button("delete missing values"):
                    df = EDA.delete_columns(
                        data=df, col=select_col_to_delete
                    )
                    st.session_state['df'] = df
                    st.success("✅ Missing values removed successfully!")
                    st.rerun()

        if EDA.check_date_object(data=df):
            st.warning(f"⚠️The date column is object")
            if st.button("change to Datetime", icon="↘️", help="change data type"):
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
                st.warning(f"Your data contains duplicate values: {duplicate}")
            select_show_duplicate_values = st.selectbox(
                "do you show duplicate values",
                options=[False, True], index=0, key="select_show_duplicate_values"
            )
            if select_show_duplicate_values:
                st.dataframe(EDA.show_duplicate_values(data=df))
            if st.button("delete duplicate values"):
                df = EDA.delete_duplicate_values(data=df)
                st.session_state['df'] = df
                st.success("✅ Duplicate values deleted successfully!")
                st.rerun()

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
            options=EDA.detect_numeric_type(data=df), key="outliers_col_selectbox"
        )
        method_selectbox = st.selectbox(
            "Please select your preferred method",
            options=["IQR", "Z_score"], key="method_selectbox"
        )
        if method_selectbox == "IQR":
            outlier = handle_outliers.detect_outliers(
                data=df, col=outliers_col_selectbox,
                method="IQR"
            )
            if len(outlier) > 0:
                st.dataframe(outlier)
                select_delete_outliers = st.selectbox(
                    "are you delete outliers",
                    options=[False, True], index=0, key="select_delete_outliers"
                )
                if select_delete_outliers:
                    df = handle_outliers.delete_outliers(
                        data=df, index_outliers=outlier.index
                    )
                    st.session_state['df'] = df
                    st.success(f"outliers deleted", icon="✅")
                    st.rerun()
            else:
                st.info(
                    f"In the IQR method, there are no outliers in '{outliers_col_selectbox}'"
                )

        elif method_selectbox == "Z_score":
            outlier = handle_outliers.detect_outliers(
                data=df, col=outliers_col_selectbox,
                method="Z_score",
            )
            if len(outlier) > 0:
                threshold_outliers = st.slider(
                    "Select outliers Threshold",
                    1.0, 4.0, 3.0
                )
                st.dataframe(
                    handle_outliers.detect_outliers(
                        data=df, col=outliers_col_selectbox, method=method_selectbox,
                        threshold=threshold_outliers)
                )
                select_delete_outliers = st.selectbox(
                    "are you delete outliers",
                    options=[False, True], index=0, key="select_delete_outliers2"
                )
                if select_delete_outliers:
                    df = handle_outliers.delete_outliers(data=df, index_outliers=outlier.index)
                    st.session_state['df'] = df
                    st.success(f"outliers deleted", icon="✅")
                    st.rerun()
            else:
                st.info(f"In the Z_score method, there are no outliers in '{outliers_col_selectbox}'")

    with st.expander("Data manipulation"):
        select_manipulation_mode = st.selectbox(
            "select manipulation mode",
            options=[
                "delete row or rows",
                "delete column or columns",
                "change name columns",
                "change data type"
            ], index=0, key="select_manipulation_mode"
        )
        if select_manipulation_mode == "delete row or rows":
            select_delete_row_mode = st.selectbox(
                "select delete row or rows",
                options=["row", "rows"], index=0, key="select_delete_row_mode"
            )
            if select_delete_row_mode == "row":
                select_row = st.number_input(
                    "select row to delete",
                    min_value=0, max_value=len(df) - 1, value=0,
                    key="select_row"
                )

                if st.button("delete row"):
                    try:
                        df = data_manipulation.delete_row(data=df, row_index=select_row)
                        st.success(f"row {select_row} deleted", icon="✅")
                        st.session_state['df'] = df
                        st.rerun()
                    except Exception as E:
                        st.error(f"row {str(E)} not founded!", icon="⚠️")
            elif select_delete_row_mode == "rows":
                start_rows = st.number_input(
                    "start row",
                    min_value=0, max_value=len(df) - 1, value=0,
                    key="start_rows"
                )
                end_rows = st.number_input(
                    "end row",
                    min_value=0, max_value=len(df) - 1, value=1,
                    key="end_rows"
                )
                if st.button("delete rows"):
                    try:
                        df = data_manipulation.delete_rows(data=df, rows_index=(start_rows, end_rows))
                        st.session_state['df'] = df
                        st.success(f"rows {(start_rows, end_rows)} deleted", icon="✅")
                        st.rerun()
                    except Exception as E:
                        st.error(f"rows {(start_rows, end_rows)} not founded!")
        elif select_manipulation_mode == "delete column or columns":
            select_delete_cols = st.selectbox(
                "select delete column or columns",
                options=["column", "columns"], index=0, key="select_delete_cols"
            )
            if select_delete_cols == "column":
                select_col = st.selectbox(
                    "select column to delete",
                    options=[None] + EDA.list_columns(data=df), key="select_col", index=0
                )
                if st.button("delete column"):
                    try:
                        df = data_manipulation.delete_column(data=df, col=select_col)
                        st.success(f"column {select_col} deleted!", icon="✅")
                        st.session_state['df'] = df
                        st.rerun()
                    except Exception as E:
                        st.error(f"{E}")
            elif select_delete_cols == "columns":
                select_cols = st.multiselect(
                    "select columns",
                    options=EDA.list_columns(data=df), key="select_cols"
                )
                if st.button("delete columns"):
                    try:
                        df = data_manipulation.delete_columns(
                            data=df, list_col=select_cols
                        )
                        st.success(f"delete {select_cols} deleted", icon="✅")
                        st.session_state['df'] = df
                        st.rerun()
                    except Exception as E:
                        st.error(f"{E}")
        elif select_manipulation_mode == "change name columns":
            select_col = st.selectbox(
                "select col to change name",
                options=EDA.list_columns(data=df), key="col0x21"
            )
            new_name = st.text_input(
                "enter column new name", key="col0x22"
            )
            if st.button("change name column"):
                df = data_manipulation.change_col_name(
                    data=df, col_name_last=select_col, col_name_new=new_name
                )
                st.success(f"column '{select_col}' change to '{new_name}'", icon="✅")
                st.session_state['df'] = df
                st.rerun()
        elif select_manipulation_mode == "change data type":
            select_col = st.selectbox(
                "select column",
                options=EDA.list_columns(data=df), key="select_col_0x34"
            )
            select_dtype = st.selectbox(
                "select dtype",
                options=['int', 'float', 'bool', 'str',
                         'object', 'category', 'datetime64[ns]', 'timedelta64[ns]'],
                key="select_dtype"
            )
            if st.button("change dtype"):
                try:
                    df = data_manipulation.change_dtype(
                        data=df, col=select_col, dtype=select_dtype
                    )
                    st.success(
                        f"column '{select_col}' changed to dtype '{select_dtype}'"
                    )
                    st.session_state['df'] = df
                    st.rerun()
                except Exception as E:
                    st.error(f"{E}")

    with st.expander("Visualization"):
        numeric = EDA.detect_numeric_type(data=df)
        category = EDA.detect_object_type(data=df)
        Time_dtype = EDA.detect_time_type(data=df)
        select_plot_mode = st.selectbox(
            "please select plot mode:",
            options=["Simple", "interactive"]
        )
        if select_plot_mode == "Simple":
            with st.expander("histogram"):
                select_columns1 = st.selectbox(
                    "please select column",
                    options=numeric + category, key="st1"
                )
                select_columns2 = st.selectbox(
                    "please select column",
                    options=[None] + numeric + category, key="st2", index=0
                )
                select_hue = st.selectbox(
                    "please select hue",
                    options=[None] + category + numeric,
                    key="st3", index=0
                )
                select_kde_mode = st.selectbox(
                    "please select mode kde",
                    options=[False, True],
                    key="st4", index=0
                )
                select_fill_mode = st.selectbox(
                    "please select mode fill",
                    options=[False, True],
                    key="st5", index=1
                )
                select_width = st.number_input("please enter width figure:", value=5, key="select_width")
                select_height = st.number_input("please enter height figure:", value=5, key="select_height")
                select_dpi = st.number_input("please enter dpi:", value=80, max_value=200, key="select_dpi")
                select_main_title = st.text_input("please enter main title:", key="select_main_title")
                select_xlabel = st.text_input("please enter xlabel:", key="select_xlabel")
                select_ylabel = st.text_input("please enter ylabel:", key="select_ylabel")
                select_main_title_fontsize = st.number_input(
                    "please enter main title fontsize:", value=15, key="select_main_title_fontsize")
                select_xlabel_fontsize = st.number_input(
                    "please enter xlabel fontsize:", value=13, key="select_xlabel_fontsize")
                select_ylabel_fontsize = st.number_input(
                    "please enter ylabel fontsize:", value=13, key="select_ylabel_fontsize")
                select_ax_mode = st.selectbox(
                    "please select mode axis",
                    options=["both", "x", "y"], index=0, key="st6"
                )
                select_ax_fontsize = st.number_input(
                    "please enter axis fontsize", value=12, key="select_ax_fontsize")
                st.pyplot(seaborn_chart.histogram(
                    data=df, x=select_columns1, y=select_columns2,
                    hue=select_hue, kde=select_kde_mode, fill=select_fill_mode,
                    figsize=(select_width, select_height), dpi=select_dpi,
                    main_title=select_main_title, xlabel=select_xlabel,
                    ylabel=select_ylabel, main_title_fontsize=select_main_title_fontsize,
                    xlabel_fontsize=select_xlabel_fontsize, ylabel_fontsize=select_ylabel_fontsize,
                    ax_fontsize=select_ax_fontsize, ax_mode=select_ax_mode
                ))
            with st.expander("kde plot"):
                select_columns1 = st.selectbox(
                    "please select column",
                    options=[None] + numeric + category, key="st7"
                )
                select_columns2 = st.selectbox(
                    "please select column",
                    options=[None] + numeric + category, key="st8", index=0
                )
                select_hue = st.selectbox(
                    "please select hue",
                    options=[None] + category + numeric,
                    key="st9", index=0
                )
                select_fill_mode = st.selectbox(
                    "please select mode fill",
                    options=[False, True],
                    key="st11", index=1
                )
                select_multiple_mode = st.selectbox(
                    "please select multiple mode",
                    options=["layer", "stack", "fill"], key="st12", index=0
                )
                if (select_hue is None) and (select_multiple_mode == "fill"):
                    st.error(
                        "To use the 'multiple' parameter in 'fill' mode, you must also have the 'hue' parameter.",
                        icon="⚠️"
                    )
                select_width = st.number_input("please enter width figure:", value=5, key="select_width2")
                select_height = st.number_input("please enter height figure:", value=5, key="select_height2")
                select_dpi = st.number_input("please enter dpi:", value=80, max_value=200, key="select_dpi2")
                select_main_title = st.text_input("please enter main title:", key="select_main_title2")
                select_xlabel = st.text_input("please enter xlabel:", key="select_xlabel2")
                select_ylabel = st.text_input("please enter ylabel:", key="select_ylabel2")
                select_main_title_fontsize = st.number_input(
                    "please enter main title fontsize:", value=15, key="select_main_title_fontsize2")
                select_xlabel_fontsize = st.number_input(
                    "please enter xlabel fontsize:", value=13, key="select_xlabel_fontsize2")
                select_ylabel_fontsize = st.number_input(
                    "please enter ylabel fontsize:", value=13, key="select_ylabel_fontsize2")
                select_ax_mode = st.selectbox(
                    "please select mode axis",
                    options=["both", "x", "y"], index=0, key="st13"
                )
                select_ax_fontsize = st.number_input(
                    "please enter axis fontsize", value=12, key="select_ax_fontsize2")
                st.pyplot(seaborn_chart.kde_plot(
                    data=df, x=select_columns1, y=select_columns2,
                    hue=select_hue, fill=select_fill_mode,
                    multiple=select_multiple_mode, main_title=select_main_title,
                    xlabel=select_xlabel, ylabel=select_ylabel,
                    main_title_fontsize=select_main_title_fontsize,
                    xlabel_fontsize=select_xlabel_fontsize,
                    ylabel_fontsize=select_ylabel_fontsize,
                    ax_fontsize=select_ax_fontsize, ax_mode=select_ax_mode,
                    figsize=(select_width, select_height), dpi=select_dpi
                ))
            with st.expander("boxplot"):
                select_x = st.selectbox(
                    "please select column x",
                    options=numeric, key="select_columns_k2",
                )
                select_y = st.selectbox(
                    "please select column y",
                    options=[None] + category + numeric, key="select_columns_k3"
                )
                select_hue = st.selectbox(
                    "please select hue",
                    options=[None] + category + numeric,
                    key="hue2", index=0
                )
                select_fill_mode = st.selectbox(
                    "please select mode fill",
                    options=[False, True],
                    key="fill2", index=1
                )
                select_saturation = st.number_input(
                    "please enter saturation",
                    value=0.75, key="saturation"
                )
                select_gap = st.slider(
                    "please enter gap",
                    min_value=0.0, max_value=5.0, value=0.15, step=0.15
                )
                select_width = st.number_input("please enter width figure:", value=5, key="select_width3")
                select_height = st.number_input("please enter height figure:", value=5, key="select_height3")
                select_dpi = st.number_input("please enter dpi:", value=80, max_value=200, key="select_dpi3")
                select_main_title = st.text_input("please enter main title:", key="select_main_title3")
                select_xlabel = st.text_input("please enter xlabel:", key="select_xlabel3")
                select_ylabel = st.text_input("please enter ylabel:", key="select_ylabel3")
                select_main_title_fontsize = st.number_input(
                    "please enter main title fontsize:", value=15, key="select_main_title_fontsize3")
                select_xlabel_fontsize = st.number_input(
                    "please enter xlabel fontsize:", value=13, key="select_xlabel_fontsize3")
                select_ylabel_fontsize = st.number_input(
                    "please enter ylabel fontsize:", value=13, key="select_ylabel_fontsize3")
                select_ax_mode = st.selectbox(
                    "please select mode axis",
                    options=["both", "x", "y"], index=0, key="st14"
                )
                select_ax_fontsize = st.number_input(
                    "please enter axis fontsize", value=12, key="select_ax_fontsize3")
                st.pyplot(
                    seaborn_chart.boxplot(
                        data=df, x=select_x, y=select_y,
                        hue=select_hue, saturation=select_saturation,
                        fill=select_fill_mode, gap=select_gap, figsize=(select_width, select_height),
                        dpi=select_dpi, main_title=select_main_title, xlabel=select_xlabel,
                        ylabel=select_ylabel, main_title_fontsize=select_main_title_fontsize,
                        xlabel_fontsize=select_xlabel_fontsize, ylabel_fontsize=select_ylabel_fontsize,
                        ax_fontsize=select_ax_fontsize, ax_mode=select_ax_mode
                    )
                )

            with st.expander("countplot"):
                select_x = st.selectbox(
                    "please select x",
                    options=[None] + category + numeric, index=0, key="select_columns_k4"
                )
                select_y = st.selectbox(
                    "please select y",
                    options=[None] + category + numeric, index=0, key="select_columns_k5"
                )
                st.info(
                    "You must choose either x or y; otherwise, an error will occur.",
                    icon="⚠️"
                )
                select_hue = st.selectbox(
                    "please select hue",
                    options=[None] + category + numeric,
                    key="hue3", index=0
                )
                select_saturation = st.number_input(
                    "please enter saturation",
                    value=0.75, key="saturation2"
                )
                select_fill_mode = st.selectbox(
                    "please select fill mode",
                    options=[False, True], index=1, key="fill_mode2"
                )
                select_gap = st.slider(
                    "please enter gap",
                    min_value=0.0, max_value=5.0, value=0.15, step=0.15, key="gap"
                )
                select_stat = st.selectbox(
                    "please select stat",
                    options=["count", "percent", "proportion", "probability"], index=0, key="stat"
                )
                select_orient = st.selectbox(
                    "please select orient",
                    options=["v", "h", "x", "y"], index=0, key="orient"
                )
                select_width_plot = st.number_input(
                    "please enter width plot",
                    value=0.8, key="width_plot"
                )
                select_log_scale = st.selectbox(
                    "please select log scale",
                    options=[False, True], index=0, key="log_scale"
                )
                select_width = st.number_input("please enter width figure:", value=5, key="select_width4")
                select_height = st.number_input("please enter height figure:", value=5, key="select_height4")
                select_dpi = st.number_input("please enter dpi:", value=80, max_value=200, key="select_dpi4")
                select_main_title = st.text_input("please enter main title:", key="select_main_title4")
                select_xlabel = st.text_input("please enter xlabel:", key="select_xlabel4")
                select_ylabel = st.text_input("please enter ylabel:", key="select_ylabel4")
                select_main_title_fontsize = st.number_input(
                    "please enter main title fontsize:", value=15, key="select_main_title_fontsize4")
                select_xlabel_fontsize = st.number_input(
                    "please enter xlabel fontsize:", value=13, key="select_xlabel_fontsize4")
                select_ylabel_fontsize = st.number_input(
                    "please enter ylabel fontsize:", value=13, key="select_ylabel_fontsize4")
                select_ax_mode = st.selectbox(
                    "please select mode axis",
                    options=["both", "x", "y"], index=0, key="st15"
                )
                select_ax_fontsize = st.number_input(
                    "please enter axis fontsize", value=12, key="select_ax_fontsize4")
                st.pyplot(
                    seaborn_chart.countplot(
                        data=df, x=select_x, y=select_y,
                        hue=select_hue, stat=select_stat,
                        saturation=select_saturation, orient=select_orient,
                        width=select_width_plot, log_scale=select_log_scale,
                        dpi=select_dpi, figsize=(select_width, select_height),
                        main_title=select_main_title, xlabel=select_xlabel,
                        ylabel=select_ylabel, main_title_fontsize=select_main_title_fontsize,
                        xlabel_fontsize=select_xlabel_fontsize, ylabel_fontsize=select_ylabel_fontsize,
                        ax_fontsize=select_ax_fontsize, ax_mode=select_ax_mode,
                        fill=select_fill_mode, gap=select_gap
                    )
                )

            with st.expander("scatterplot"):
                select_x = st.selectbox(
                    "please select x",
                    options=[None] + numeric + category,
                    index=0, key="select_columns_k6"
                )
                select_y = st.selectbox(
                    "please select y",
                    options=[None] + numeric + category,
                    index=0, key="select_columns_k7"
                )
                select_hue = st.selectbox(
                    "please select hue",
                    options=[None] + category + numeric,
                    key="hue4", index=0
                )
                select_size = st.selectbox(
                    "please select size",
                    options=[None] + category + numeric,
                    index=0, key="select_size"
                )

                select_size_point = st.slider(
                    "please select size point",
                    min_value=50, max_value=500, step=5, key="select_size2"
                )
                select_style = st.selectbox(
                    "please select style",
                    options=[None] + category + numeric, index=0, key="select_style"
                )
                select_width = st.number_input("please enter width figure:", value=5, key="select_width5")
                select_height = st.number_input("please enter height figure:", value=5, key="select_height5")
                select_dpi = st.number_input("please enter dpi:", value=80, max_value=200, key="select_dpi5")
                select_main_title = st.text_input("please enter main title:", key="select_main_title5")
                select_xlabel = st.text_input("please enter xlabel:", key="select_xlabel5")
                select_ylabel = st.text_input("please enter ylabel:", key="select_ylabel5")
                select_main_title_fontsize = st.number_input(
                    "please enter main title fontsize:", value=15, key="select_main_title_fontsize5")
                select_xlabel_fontsize = st.number_input(
                    "please enter xlabel fontsize:", value=13, key="select_xlabel_fontsize5")
                select_ylabel_fontsize = st.number_input(
                    "please enter ylabel fontsize:", value=13, key="select_ylabel_fontsize5")
                select_ax_mode = st.selectbox(
                    "please select mode axis",
                    options=["both", "x", "y"], index=0, key="st16"
                )
                select_ax_fontsize = st.number_input(
                    "please enter axis fontsize", value=12, key="select_ax_fontsize5")

                st.pyplot(
                    seaborn_chart.scatterplot(
                        data=df, x=select_x, y=select_y, hue=select_hue, s=select_size_point, size=select_size,
                        style=select_style, figsize=(select_width, select_height),
                        dpi=select_dpi, main_title=select_main_title, xlabel=select_xlabel,
                        ylabel=select_ylabel, main_title_fontsize=select_main_title_fontsize,
                        xlabel_fontsize=select_xlabel_fontsize, ylabel_fontsize=select_ylabel_fontsize,
                        ax_mode=select_ax_mode, ax_fontsize=select_ax_fontsize
                    )
                )

            with st.expander("violin plot"):
                select_x = st.selectbox(
                    "please select x",
                    options=[None] + numeric + category, index=0, key="select_columns_k8"
                )
                select_y = st.selectbox(
                    "please select y",
                    options=[None] + numeric + category, index=0, key="select_columns_k9"
                )
                select_hue = st.selectbox(
                    "please select hue",
                    options=[None] + category + numeric, index=0, key="hue5",
                )
                select_orient = st.selectbox(
                    "please select orient",
                    options=["v", "h", "x", "y"], index=0, key="orient2"
                )
                select_fill_mode = st.selectbox(
                    "please select fill mode",
                    options=[False, True], index=0, key="fill_mode3"
                )
                select_gap = st.slider(
                    "please enter gap",
                    min_value=0.0, max_value=5.0, value=0.15, step=0.15, key="gap2"
                )
                select_width = st.number_input("please enter width figure:", value=5, key="select_width6")
                select_height = st.number_input("please enter height figure:", value=5, key="select_height6")
                select_dpi = st.number_input("please enter dpi:", value=80, max_value=200, key="select_dpi6")
                select_main_title = st.text_input("please enter main title:", key="select_main_title6")
                select_xlabel = st.text_input("please enter xlabel:", key="select_xlabel6")
                select_ylabel = st.text_input("please enter ylabel:", key="select_ylabel6")
                select_main_title_fontsize = st.number_input(
                    "please enter main title fontsize:", value=15, key="select_main_title_fontsize6")
                select_xlabel_fontsize = st.number_input(
                    "please enter xlabel fontsize:", value=13, key="select_xlabel_fontsize6")
                select_ylabel_fontsize = st.number_input(
                    "please enter ylabel fontsize:", value=13, key="select_ylabel_fontsize6")
                select_ax_mode = st.selectbox(
                    "please select mode axis",
                    options=["both", "x", "y"], index=0, key="st17"
                )
                select_ax_fontsize = st.number_input(
                    "please enter axis fontsize", value=12, key="select_ax_fontsize6")
                st.pyplot(
                    seaborn_chart.violin_plot(
                        data=df, x=select_x, y=select_y, hue=select_hue, gap=select_gap,
                        fill=select_fill_mode, orient=select_orient, figsize=(select_width, select_height),
                        dpi=select_dpi, main_title=select_main_title, xlabel=select_xlabel,
                        ylabel=select_ylabel, main_title_fontsize=select_main_title_fontsize,
                        xlabel_fontsize=select_xlabel_fontsize, ylabel_fontsize=select_ylabel_fontsize,
                        ax_fontsize=select_ax_fontsize, ax_mode=select_ax_mode
                    )
                )
            with st.expander("lineplot"):
                select_x = st.selectbox(
                    "please select x",
                    options=[None] + numeric + category + Time_dtype, index=0, key="select_columns_k10"
                )
                select_y = st.selectbox(
                    "please select y",
                    options=[None] + numeric + category + Time_dtype, index=0, key="select_columns_k11"
                )
                select_hue = st.selectbox(
                    "please select hue",
                    options=[None] + category + Time_dtype + numeric , index=0, key="hue6",
                )
                select_style = st.selectbox(
                    "please select style",
                    options=[None] + category + Time_dtype + numeric, index=0, key="select_style2"
                )
                select_size = st.selectbox(
                    "please select size",
                    options=[None] + category + numeric,
                    index=0, key="select_size3"
                )
                select_sort = st.selectbox(
                    "please select sort",
                    options=[False, True], index=1, key="select_sort"
                )
                select_width = st.number_input("please enter width figure:", value=5, key="select_width7")
                select_height = st.number_input("please enter height figure:", value=5, key="select_height7")
                select_dpi = st.number_input("please enter dpi:", value=80, max_value=200, key="select_dpi7")
                select_main_title = st.text_input("please enter main title:", key="select_main_title7")
                select_xlabel = st.text_input("please enter xlabel:", key="select_xlabel7")
                select_ylabel = st.text_input("please enter ylabel:", key="select_ylabel7")
                select_main_title_fontsize = st.number_input(
                    "please enter main title fontsize:", value=15, key="select_main_title_fontsize7")
                select_xlabel_fontsize = st.number_input(
                    "please enter xlabel fontsize:", value=13, key="select_xlabel_fontsize7")
                select_ylabel_fontsize = st.number_input(
                    "please enter ylabel fontsize:", value=13, key="select_ylabel_fontsize7")
                select_ax_mode = st.selectbox(
                    "please select mode axis",
                    options=["both", "x", "y"], index=0, key="st18"
                )
                select_ax_fontsize = st.number_input(
                    "please enter axis fontsize", value=12, key="select_ax_fontsize7")
                st.pyplot(
                    seaborn_chart.lineplot(
                        data=df, x=select_x, y=select_y, hue=select_hue, style=select_style,
                        size=select_size, sort=select_sort, figsize=(select_width, select_height),
                        dpi=select_dpi, main_title=select_main_title, xlabel=select_xlabel,
                        ylabel=select_ylabel, main_title_fontsize=select_main_title_fontsize,
                        xlabel_fontsize=select_xlabel_fontsize, ylabel_fontsize=select_ylabel_fontsize,
                        ax_fontsize=select_ax_fontsize, ax_mode=select_ax_mode
                    )
                )

            with st.expander("correlation heatmap"):
                select_annot = st.selectbox(
                    "please select annot",
                    options=[False, True], index=0, key="annot1"
                )
                select_fmt_mode = st.selectbox(
                    "please select mode fmt",
                    options=["Auto", "Manual"], index=0, key="select_fmt_mode1"
                )
                select_size_annot = st.slider(
                    "please select size annot",
                    min_value=5, max_value=25, value=15, step=1, key="select_size_annot"
                )
                select_width = st.number_input("please enter width figure:", value=5, key="select_width8")
                select_height = st.number_input("please enter height figure:", value=5, key="select_height8")
                select_dpi = st.number_input("please enter dpi:", value=80, max_value=200, key="select_dpi8")
                select_main_title = st.text_input("please enter main title:", key="select_main_title8")
                select_xlabel = st.text_input("please enter xlabel:", key="select_xlabel8")
                select_ylabel = st.text_input("please enter ylabel:", key="select_ylabel8")
                select_main_title_fontsize = st.number_input(
                    "please enter main title fontsize:", value=15, key="select_main_title_fontsize8")
                select_xlabel_fontsize = st.number_input(
                    "please enter xlabel fontsize:", value=13, key="select_xlabel_fontsize8")
                select_ylabel_fontsize = st.number_input(
                    "please enter ylabel fontsize:", value=13, key="select_ylabel_fontsize8")
                select_ax_mode = st.selectbox(
                    "please select mode axis",
                    options=["both", "x", "y"], index=0, key="st19"
                )
                select_ax_fontsize = st.number_input(
                    "please enter axis fontsize", value=12, key="select_ax_fontsize8")
                if select_fmt_mode == "Auto":
                    st.pyplot(
                        seaborn_chart.heatmap(
                            data=EDA.all_correlation(data=df), annot=select_annot,
                            figsize=(select_width, select_height),
                            dpi=select_dpi, main_title=select_main_title, xlabel=select_xlabel,
                            ylabel=select_ylabel, main_title_fontsize=select_main_title_fontsize,
                            xlabel_fontsize=select_xlabel_fontsize, ylabel_fontsize=select_ylabel_fontsize,
                            ax_fontsize=select_ax_fontsize, ax_mode=select_ax_mode,
                            annot_kws={"size": select_size_annot}

                        )
                    )
                else:
                    select_fmt = st.selectbox(
                        "please select fmt format",
                        options=[".1f", ".2f", ".3f", ".4f"], index=1, key="select_fmt"
                    )
                    st.pyplot(
                        seaborn_chart.heatmap(
                            data=EDA.all_correlation(data=df), annot=select_annot, fmt=select_fmt,
                            figsize=(select_width, select_height),
                            dpi=select_dpi, main_title=select_main_title, xlabel=select_xlabel,
                            ylabel=select_ylabel, main_title_fontsize=select_main_title_fontsize,
                            xlabel_fontsize=select_xlabel_fontsize, ylabel_fontsize=select_ylabel_fontsize,
                            ax_fontsize=select_ax_fontsize, ax_mode=select_ax_mode,
                            annot_kws={"size": select_size_annot}
                        )
                    )
        elif select_plot_mode == "interactive":
            with st.expander("scatter plot"):
                select_x = st.selectbox(
                    "please select x",
                    options=numeric + category + Time_dtype,
                    index=0, key="select_columns_plotly_1"
                )
                select_y = st.selectbox(
                    "please select y",
                    options=numeric + category + Time_dtype,
                    index=0, key="select_columns_plotly_2"
                )
                select_color = st.selectbox(
                    "please select color",
                    options=[None] + numeric + category + Time_dtype,
                    index=0, key="select_color_plotly"
                )
                select_size = st.selectbox(
                    "please select size",
                    options=[None] + numeric + category + Time_dtype, index=0, key="select_size_plotly"
                )
                select_symbol = st.selectbox(
                    "please select symbol",
                    options=[None] + numeric + category + Time_dtype,
                    index=0, key="select_symbol_plotly"
                )
                select_size_symbol = st.slider(
                    "please select size symbol",
                    min_value=5, max_value=25, value=15, key="select_symbol_size_plotly"
                )
                select_hover_data = st.multiselect(
                    "please select hover data",
                    options=[None] + numeric + category + Time_dtype,
                    key="select_hover_data_plotly"
                )
                select_log_x = st.selectbox(
                    "please select log x",
                    options=[False, True], index=0, key="select_log_x_plotly"
                )
                select_log_y = st.selectbox(
                    "please select log y",
                    options=[False, True], index=0, key="select_log_x_plotly2"
                )
                select_template = st.selectbox(
                    "please select template",
                    options=[
                        "plotly", "plotly_white", "plotly_dark",
                        "ggplot2", "seaborn", "simple_white",
                        "presentation", "none"
                    ],
                    index=0, key="select_template_plotly"
                )
                select_width = st.number_input("please enter width figure:", value=900, key="select_width9")
                select_height = st.number_input("please enter height figure:", value=500, key="select_height9")
                select_size_marker = st.number_input("please enter size marker:", value=10, max_value=100,
                                                     key="select_dpi9")
                select_main_title = st.text_input("please enter main title:", key="select_main_title9")
                select_xlabel = st.text_input("please enter xlabel:", key="select_xlabel9")
                select_ylabel = st.text_input("please enter ylabel:", key="select_ylabel9")
                select_main_title_fontsize = st.number_input(
                    "please enter main title fontsize:", value=15, key="select_main_title_fontsize9")
                select_xlabel_fontsize = st.number_input(
                    "please enter xlabel fontsize:", value=13, key="select_xlabel_fontsize9")
                select_ylabel_fontsize = st.number_input(
                    "please enter ylabel fontsize:", value=13, key="select_ylabel_fontsize9")
                select_tickfont_x = st.number_input(
                    "please enter x axis size:", value=13, key="select_tickfont_x")
                select_tickfont_y = st.number_input(
                    "please enter y axis size:", value=13, key="select_tickfont_y")
                st.plotly_chart(
                    plotly_charts.scatterplot(
                        data=df, x=select_x, y=select_y,
                        color=select_color, symbol=select_symbol,
                        size_max_symbol=select_size_symbol, hover_data=select_hover_data,
                        log_x=select_log_x, log_y=select_log_y, size_marker=select_size_marker,
                        figsize=(select_width, select_height), main_title=select_main_title,
                        xlabel=select_xlabel, ylabel=select_ylabel,
                        xlabel_fontsize=select_xlabel_fontsize, ylabel_fontsize=select_ylabel_fontsize,
                        main_title_fontsize=select_main_title_fontsize, tickfont_x=select_tickfont_x,
                        tickfont_y=select_tickfont_y, template=select_template

                    )
                )
            with st.expander("histogram"):
                select_x = st.selectbox(
                    "please select x",
                    options=numeric + category + Time_dtype,
                    index=0, key="select_x_plotly_hi"
                )
                select_y = st.selectbox(
                    "please select y",
                    options=[None] + numeric + category + Time_dtype,
                    index=0, key="select_y_plotly_hi"
                )
                select_color = st.selectbox(
                    "please select color",
                    options=[None] + numeric + category + Time_dtype,
                    index=0, key="select_color_plotly_hi"
                )
                select_log_x = st.selectbox(
                    "please select log x",
                    options=[False, True], index=0, key="select_log_x_plotly_hi"
                )
                select_log_y = st.selectbox(
                    "please select log y",
                    options=[False, True], index=0, key="select_log_y_plotly_hi"
                )
                select_template = st.selectbox(
                    "please select template",
                    options=[
                        "plotly", "plotly_white", "plotly_dark",
                        "ggplot2", "seaborn", "simple_white",
                        "presentation", "none"
                    ],
                    index=0, key="select_template_plotly_hi"
                )
                select_width = st.number_input("please enter width figure:", value=900, key="select_width10")
                select_height = st.number_input("please enter height figure:", value=500, key="select_height10")
                select_nbins = st.number_input("please enter number of bins:", value=60, max_value=300,
                                               key="select_nbins", min_value=30)
                select_bargap = st.slider(
                    "please enter bar gap",
                    min_value=0.1, max_value=10.0, value=0.1, step=0.1,
                    key="bar gap histogram"
                )
                select_main_title = st.text_input("please enter main title:", key="select_main_title10")
                select_xlabel = st.text_input("please enter xlabel:", key="select_xlabel10")
                select_ylabel = st.text_input("please enter ylabel:", key="select_ylabel10")
                select_main_title_fontsize = st.number_input(
                    "please enter main title fontsize:", value=15, key="select_main_title_fontsize10")
                select_xlabel_fontsize = st.number_input(
                    "please enter xlabel fontsize:", value=13, key="select_xlabel_fontsize10")
                select_ylabel_fontsize = st.number_input(
                    "please enter ylabel fontsize:", value=13, key="select_ylabel_fontsize10")
                select_tickfont_x = st.number_input(
                    "please enter x axis size:", value=13, key="select_tickfont_x_2")
                select_tickfont_y = st.number_input(
                    "please enter y axis size:", value=13, key="select_tickfont_y_2")
                st.plotly_chart(
                    plotly_charts.histogram(
                        data=df, x=select_x, y=select_y,
                        color=select_color,
                        log_x=select_log_x, log_y=select_log_y,
                        nbins=select_nbins, figsize=(select_width, select_height),
                        main_title=select_main_title, xlabel=select_xlabel, ylabel=select_ylabel,
                        main_title_fontsize=select_main_title_fontsize,
                        xlabel_fontsize=select_xlabel_fontsize, ylabel_fontsize=select_ylabel_fontsize,
                        tickfont_x=select_tickfont_x, tickfont_y=select_tickfont_y,
                        template=select_template, bargap=select_bargap
                    )
                )
            with st.expander("box plot"):
                select_x = st.selectbox(
                    "please select x",
                    options=[None] + numeric + category + Time_dtype,
                    index=0, key="select_x_plotly_box"
                )
                select_y = st.selectbox(
                    "please select y",
                    options=[None] + numeric + category + Time_dtype,
                    index=0, key="select_y_plotly_box"
                )
                select_color = st.selectbox(
                    "please select color",
                    options=[None] + numeric + category + Time_dtype,
                    index=0, key="select_color_plotly_box"
                )
                select_log_x = st.selectbox(
                    "please select log x",
                    options=[False, True], index=0, key="select_log_x_plotly_box"
                )
                select_log_y = st.selectbox(
                    "please select log y",
                    options=[False, True], index=0, key="select_log_y_plotly_box"
                )
                select_template = st.selectbox(
                    "please select template",
                    options=[
                        "plotly", "plotly_white", "plotly_dark",
                        "ggplot2", "seaborn", "simple_white",
                        "presentation", "none"
                    ],
                    index=0, key="select_template_plotly_box"
                )
                select_width = st.number_input("please enter width figure:", value=900, key="select_width12")
                select_height = st.number_input("please enter height figure:", value=500, key="select_height12")
                select_main_title = st.text_input("please enter main title:", key="select_main_title12")
                select_xlabel = st.text_input("please enter xlabel:", key="select_xlabel12")
                select_ylabel = st.text_input("please enter ylabel:", key="select_ylabel12")
                select_main_title_fontsize = st.number_input(
                    "please enter main title fontsize:", value=15, key="select_main_title_fontsize12")
                select_xlabel_fontsize = st.number_input(
                    "please enter xlabel fontsize:", value=13, key="select_xlabel_fontsize12")
                select_ylabel_fontsize = st.number_input(
                    "please enter ylabel fontsize:", value=13, key="select_ylabel_fontsize12")
                select_tickfont_x = st.number_input(
                    "please enter x axis size:", value=13, key="select_tickfont_x_3")
                select_tickfont_y = st.number_input(
                    "please enter y axis size:", value=13, key="select_tickfont_y_3")
                if st.button("plotting", use_container_width=True, icon="📈"):
                    st.plotly_chart(
                        plotly_charts.boxplot(
                            data=df, x=select_x, y=select_y,
                            color=select_color, log_x=select_log_x,
                            log_y=select_log_y, template=select_template,
                            figsize=(select_width, select_height),
                            main_title=select_main_title, xlabel=select_xlabel, ylabel=select_ylabel,
                            main_title_fontsize=select_main_title_fontsize,
                            xlabel_fontsize=select_xlabel_fontsize, ylabel_fontsize=select_ylabel_fontsize,
                            tickfont_x=select_tickfont_x, tickfont_y=select_tickfont_y,
                        )
                    )
            with st.expander("violin plot"):
                pass
            with st.expander("line plot"):
                select_x = st.selectbox(
                    "please select x",
                    options=[None] + numeric + category + Time_dtype,
                    index=0, key="select_x_line"
                )
                select_y = st.selectbox(
                    "please select y",
                    options=[None] + numeric + category + Time_dtype,
                    index=0, key="select_y_0i82"
                )
                select_color = st.selectbox(
                    "please select color",
                    options=[None] + numeric + category + Time_dtype,
                    index=0, key="select_color_e53"
                )
                select_log_x = st.selectbox(
                    "please select log x",
                    options=[False, True], index=0, key="select_log_2wd"
                )
                select_log_y = st.selectbox(
                    "please select log y",
                    options=[False, True], index=0, key="select_log_y_pl85"
                )
                select_template = st.selectbox(
                    "please select template",
                    options=[
                        "plotly", "plotly_white", "plotly_dark",
                        "ggplot2", "seaborn", "simple_white",
                        "presentation", "none"
                    ],
                    index=0, key="select_template_h824"
                )
                select_width = st.number_input("please enter width figure:", value=900, key="select_width0cv4")
                select_height = st.number_input("please enter height figure:", value=500, key="select_height0cod")
                select_xlabel = st.text_input("please enter xlabel:", key="select_xlabel_qw2d")
                select_ylabel = st.text_input("please enter ylabel:", key="select_ylabe_er43")
                select_main_title_fontsize = st.number_input(
                    "please enter main title fontsize:", value=15, key="select_main_title_fontsize13")
                select_xlabel_fontsize = st.number_input(
                    "please enter xlabel fontsize:", value=13, key="select_xlabel_fontsize13")
                select_ylabel_fontsize = st.number_input(
                    "please enter ylabel fontsize:", value=13, key="select_ylabel_fontsize13")
                select_tickfont_x = st.number_input(
                    "please enter x axis size:", value=13, key="select_tickfont_x_4")
                select_tickfont_y = st.number_input(
                    "please enter y axis size:", value=13, key="select_tickfont_y_4")
                if st.button("plotting", use_container_width=True, icon="📈", key="po932f"):
                    st.plotly_chart(
                        plotly_charts.lineplot(
                            data=df, x=select_x, y=select_y,
                            color=select_color, log_x=select_log_x,
                            log_y=select_log_y, template=select_template,
                            figsize=(select_width, select_height),
                            main_title=select_main_title, xlabel=select_xlabel, ylabel=select_ylabel,
                            main_title_fontsize=select_main_title_fontsize,
                            xlabel_fontsize=select_xlabel_fontsize, ylabel_fontsize=select_ylabel_fontsize,
                            tickfont_x=select_tickfont_x, tickfont_y=select_tickfont_y,
                        )
                    )
            with st.expander("heatmap"):
                pass
    if st.button("download data"):
        success, message = save_data(data=st.session_state['df'])
        if success:
            st.success(f"Data saved successfully at: '{message}'")
        else:
            st.error(f"Failed to save data: '{message}'")



else:
    st.warning(
        "Please upload the data to the first page to activate this page",
        icon="⚠️"
    )
