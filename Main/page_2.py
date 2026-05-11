import streamlit as st

from Core.preprocessor import MissingValue, EDA, handle_outliers
from components.charts import seaborn_chart

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
            if st.button("change to Datetime", icon="↘️", help="change data type"):
                df = EDA.change_dtype_datetime64(data=df)
                st.session_state['df'] = df
                st.success("✅ The date column was successfully updated!")

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
            options=EDA.detect_numeric_type(data=df), key="outliers_col_selectbox"
        )
        method_selectbox = st.selectbox(
            "Please select your preferred method",
            options=["IQR", "Z_score"], key="method_selectbox"
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
    with st.expander("Visualization"):
        numeric = EDA.detect_numeric_type(data=df)
        category = EDA.detect_object_type(data=df)
        select_plot_mode = st.selectbox(
            "please select plot mode:",
            options=["Simple", "interactive"]
        )
        if select_plot_mode == "Simple":
            with st.expander("histogram"):
                select_columns1 = st.selectbox(
                    "please select column",
                    options=numeric, key="st1"
                )
                select_columns2 = st.selectbox(
                    "please select column",
                    options=[None] + numeric, key="st2", index=0
                )
                select_hue = st.selectbox(
                    "please select hue",
                    options=[None] + category,
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
                    options=numeric, key="st7"
                )
                select_columns2 = st.selectbox(
                    "please select column",
                    options=[None] + numeric, key="st8", index=0
                )
                select_hue = st.selectbox(
                    "please select hue",
                    options=[None] + category,
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
                    options=[None] + category, key="select_columns_k3"
                )
                select_hue = st.selectbox(
                    "please select hue",
                    options=[None] + category,
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
                    options=[None] + category, index=0, key="select_columns_k4"
                )
                select_y = st.selectbox(
                    "please select y",
                    options=[None] + category, index=0, key="select_columns_k5"
                )
                st.info(
                    "You must choose either x or y; otherwise, an error will occur.",
                    icon="⚠️"
                )
                select_hue = st.selectbox(
                    "please select hue",
                    options=[None] + category,
                    key="hue3", index=0
                )
                select_saturation = st.number_input(
                    "please enter saturation",
                    value=0.75, key="saturation2"
                )
                select_fill_mode = st.selectbox(
                    "please select fill mode",
                    options=[False, True], index=0, key="fill_mode2"
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
                    "please select x",
                    options=[None] + numeric + category,
                    index=0, key="select_columns_k7"
                )
                select_hue = st.selectbox(
                    "please select hue",
                    options=[None] + category,
                    key="hue4", index=0
                )
                select_size = st.selectbox(
                    "please select size",
                    options=[None] + category,
                    index=0, key="select_size"
                )

                select_size_point = st.slider(
                    "please select size point",
                    min_value=50, max_value=500, step=5, key="select_size2"
                )
                select_style = st.selectbox(
                    "please select style",
                    options=[None] + category, index=0, key="select_style"
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
