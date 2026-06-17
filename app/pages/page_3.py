import streamlit as st
from sklearn.preprocessing import LabelEncoder

from Core.Preprocessor import DataPreprocessor
from Core.eda import EDA
from Core.evaluator import Evaluator
from Core.model_trainer import ModelParameterFactory, ModelPipelineBuilder
from components.metrics_plots import MetricPlot

st.markdown(
    """
    
# 🔧 Machine Learning Modeling

This page is the **Machine Learning Modeling** section of the Streamlit app, designed to build, train, evaluate, and compare machine learning models using the processed dataset.
The workflow guides users through preprocessing, model selection, training, validation, and performance evaluation.

### Sections

- **Learning Configuration**
  - Supports both supervised and unsupervised learning workflows.
  - Allows users to select the task type (Classification, Regression, Clustering, or Dimensionality Reduction).
  - Provides access to a variety of machine learning algorithms.

- **Target Selection**
  - Lets users choose the target column for supervised learning tasks.
  - Automatically separates features and target variables during training.

- **Data Preprocessing**
  - Supports multiple scaling methods:
    - No Scaling
    - Standard Scaler
    - MinMax Scaler
    - Robust Scaler
  - Supports feature encoding:
    - Ordinal Encoder
    - One Hot Encoder
  - Supports target encoding using Label Encoder.
  - Includes optional missing-value imputation for numerical and categorical features.

- **Train/Test Split**
  - Allows configuration of the test size ratio.
  - Supports stratified splitting for classification tasks.
  - Ensures reproducible data partitioning for model evaluation.

- **Cross Validation**
  - Optional K-Fold Cross Validation.
  - Configurable number of folds.
  - Provides more reliable performance estimates across multiple data splits.

- **Configuration Summary**
  - Displays a detailed overview of the selected preprocessing pipeline,
    dataset split, encoding methods, scaling configuration,
    and cross-validation settings before training.

- **Model Parameters**
  - Shows the hyperparameters used by the selected model.
  - Helps users understand and review model configurations before training.

- **Model Training**
  - Automatically builds a preprocessing and machine learning pipeline.
  - Trains the selected model using the configured settings.
  - Provides progress feedback during execution.

- **Model Evaluation**
  - Generates performance metrics after training.
  - Classification metrics may include:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - ROC-AUC
  - Regression metrics may include:
    - MAE
    - MSE
    - RMSE
    - R² Score

- **Classification Visualizations**
  - Confusion Matrix
  - ROC Curve
  - Learning Curve
  - Helps evaluate classification performance and detect overfitting or underfitting.

- **Regression Visualizations**
  - Actual vs Predicted Plot
  - Learning Curve
  - Helps assess prediction quality and model generalization.

- **Model Comparison**
  - Allows comparison between evaluation results and cross-validation scores.
  - Helps identify the most suitable model for the dataset.

### Supported Models

#### Classification
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Decision Tree
- Neural Network (MLP)
- AdaBoost
- XGBoost
- LightGBM
- Extra Trees

#### Regression
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Random Forest Regressor
- Gradient Boosting Regressor
- KNN Regressor
- Decision Tree Regressor
- Support Vector Regressor (SVR)
- Extra Trees Regressor
- AdaBoost Regressor
- XGBoost Regressor
- Neural Network Regressor
- LightGBM Regressor

#### Unsupervised Learning
- K-Means Clustering
- DBSCAN


### Pipeline Workflow

1. Select learning type and model.
2. Configure preprocessing options.
3. Choose train/test split settings.
4. Enable optional cross-validation.
5. Review configuration summary.
6. Train the model.
7. Evaluate performance.
8. Analyze visualizations and metrics.
"""

)
if 'df' in st.session_state:
    df = st.session_state['df']

    st.text("Model Configuration", text_alignment="center", width=1000)

    MODELS = {
        "supervised": {
            "classification": [
                "Logistic Regression", "Random Forest", "Support Vector Machine",
                "Knn", "Gaussian Naive Bayes", "Decision Tree", "Neural Network",
                "AdaBoost", "XGBoost", "LightGBM", "Extra Tree"
            ],
            "regression": [
                "Linear Regression", "Ridge Regression", "Random Forest Regressor",
                "Lasso Regression", "ElasticNet", "Gradient Boosting Regressor",
                "Knn Regressor",
                "Decision Tree Regressor", "SVR(support vector Regressor)",
                "ExtraTree Regressor", "AdaBoost Regressor", "XGBoost Regressor",
                "Neural Network(Regressor)", "LightGBM Regressor",

            ]
        },
        "unsupervised": {
            "clustering": ["K-Means", "DBSCAN"],
        }
    }
    SCALER = [
        "No Scaling",
        "MinMax Scaler",
        "Standard Scaler",
        "Robust Scaler"
    ]
    FEATURE_ENCODER = [
        "Ordinal Encoder",
        "One Hot Encoder",
    ]
    TARGET_ENCODER = [
        "None",
        "Label Encoder"
    ]

    select_learning_type = st.selectbox(
        "Select Learning Type",
        options=["supervised", "unsupervised"],
        index=0
    )

    if select_learning_type:
        task_options = list(MODELS[select_learning_type].keys())
        select_task_type = st.selectbox("Select Task Type", task_options)

        if select_task_type:
            model_options = MODELS[select_learning_type][select_task_type]
            select_model_type = st.selectbox("Select Model", model_options)
            if select_task_type == "clustering":
                pass
            else:
                select_target = st.selectbox(
                    "select target",
                    options=EDA.list_columns(data=df), key="0t42"
                )
            select_Scaler_type = st.selectbox(
                "select Scaler Type",
                options=SCALER, index=0, key="select_Scaler_type"
            )
            st.info(
                "Standard Scaling is generally recommended for most machine learning algorithms, "
                "while Min-Max Scaling is often preferred for neural networks.",
                icon="ℹ️"
            )
            select_feature_encoder = st.selectbox(
                "select Feature Encoder Type",
                options=[None] + FEATURE_ENCODER, index=0, key="select_feature_encoder"
            )
            if select_task_type != "clustering":
                select_target_encoder = st.selectbox(
                    "Select target encoder",
                    options=TARGET_ENCODER, key="select_target_encoder"
                )
            select_impute = st.checkbox(
                "Enable Imputation",
                value=False
            )
            select_num_impute_strategy = None
            select_cat_impute_strategy = None
            if select_impute:
                st.info(
                    "Whenever possible, it is recommended to remove missing values, "
                    "as imputation methods may not always accurately recover the information lost due to missing data.",
                    icon="ℹ️"
                )
                select_num_impute_strategy = st.selectbox(
                    "Numeric Imputation Strategy",
                    [None, "mean", "median", "most_frequent", "constant"],
                    key="num_impute"
                )

                select_cat_impute_strategy = st.selectbox(
                    "Categorical Imputation Strategy",
                    [None, "most_frequent", "constant"],
                    key="cat_impute"
                )
                num_fill_value = None
                cat_fill_value = None

                if select_num_impute_strategy == "constant":
                    num_fill_value = st.number_input(
                        "Numeric Fill Value",
                        value=0
                    )

                if select_cat_impute_strategy == "constant":
                    cat_fill_value = st.text_input(
                        "Categorical Fill Value",
                        value="Unknown"
                    )
            if (select_task_type == "regression") or (select_task_type == "classification"):
                select_test_size = st.slider(
                    "select test size(%)",
                    min_value=20, max_value=50, value=30, key="select_test_size",
                    step=1
                )
                select_stratify = st.checkbox(
                    "Enable Stratify",
                    value=False, key="select_stratify"
                )
                enable_cv = st.checkbox(
                    "Enable Cross Validation",
                    value=False
                )
                cv_folds = None
                if enable_cv:
                    cv_folds = st.slider(
                        "CV Folds",
                        min_value=2,
                        max_value=10,
                        value=5
                    )
            else:
                pass

            st.markdown(
                """
                
                ---
                
                """
            )
            st.text("configuration summary", text_alignment="center", width=1000)
            if st.button("Show configuration summary", use_container_width=True, icon="🔗"):
                num_cols = EDA.detect_numeric_type(data=df)
                cat_cols = EDA.detect_object_type(data=df)
                pr = DataPreprocessor(num_cols=num_cols, cat_cols=cat_cols)
                test_ratio = (select_test_size / 100) if (select_task_type == "classification") or (
                        select_task_type == "regression") else None

                st.dataframe(pr.get_config_df(
                    data=df,
                    model_name=select_model_type,
                    task_type=select_task_type,
                    target_col=select_target if (select_task_type == "classification") or (
                            select_task_type == "regression") else None,
                    test_size=test_ratio if (select_task_type == "classification") or (
                            select_task_type == "regression") else None,
                    train_size=1 - test_ratio if (select_task_type == "classification") or (
                            select_task_type == "regression") else None,
                    stratify=select_stratify if select_task_type != "clustering" else None,
                    n_test=int(len(df) * test_ratio) if select_task_type != "clustering" else None,
                    n_train=len(df) - int(len(df) * test_ratio) if select_task_type != "clustering" else len(df),
                    cross_validation=enable_cv if select_task_type != "clustering" else None,
                    cv_folds=cv_folds if select_task_type != "clustering" else None,
                    scaler_type=select_Scaler_type,
                    impute=select_impute,
                    num_impute_strategy=select_num_impute_strategy,
                    cat_impute_strategy=select_cat_impute_strategy,
                    encoder_feature_type=select_feature_encoder
                ))
            st.markdown(
                """
                ---
        
                """
            )
            st.text("Model Parameter", text_alignment="center", width=1000)
            model_params = ModelParameterFactory.get_params(
                select_model_type
            )
            if st.button("Show configuration model summary", use_container_width=True, icon="📅"):
                st.dataframe(model_params)
            train_btn = st.button(
                "🚀 Train Model",
                use_container_width=True
            )

            if train_btn:

                with st.spinner("Training model..."):

                    # ==================================
                    # Clustering
                    # ==================================

                    if select_task_type == "clustering":

                        X = df.copy()

                        num_cols = EDA.detect_numeric_type(X)
                        cat_cols = EDA.detect_object_type(X)

                        preprocessor = DataPreprocessor(
                            num_cols=num_cols,
                            cat_cols=cat_cols
                        )

                        transformer = preprocessor.get_transformer(
                            scaler_type=select_Scaler_type,
                            impute=select_impute,
                            num_impute_strategy=select_num_impute_strategy,
                            cat_impute_strategy=select_cat_impute_strategy,
                            encoder_feature_type=select_feature_encoder
                        )

                        pipeline = ModelPipelineBuilder(
                            preprocessor=transformer
                        ).build_pipeline(
                            model_type=select_model_type,
                            model_params=model_params
                        )

                        evaluator = Evaluator(
                            pipeline=pipeline,
                            task_type="clustering"
                        )

                        metrics_df = evaluator.clustering_report(X)

                        st.success("Model trained successfully!")

                        st.markdown("---")
                        st.text(
                            "Result",
                            text_alignment="center",
                            width=1010
                        )

                        st.dataframe(metrics_df)

                    # ==================================
                    # Classification / Regression
                    # ==================================

                    else:

                        X = df.drop(columns=[select_target])
                        y = df[select_target]

                        num_cols = EDA.detect_numeric_type(X)
                        cat_cols = EDA.detect_object_type(X)

                        X_train, X_test, y_train, y_test = (
                            DataPreprocessor.set_setting_split(
                                x=X,
                                y=y.ravel(),
                                test_size=select_test_size / 100,
                                stratify=select_stratify
                            )
                        )

                        if select_target_encoder == "Label Encoder":

                            label_encoder = LabelEncoder()

                            y_train_encode = label_encoder.fit_transform(
                                y_train
                            )

                            y_test_encode = label_encoder.transform(
                                y_test
                            )

                        else:

                            y_train_encode = y_train
                            y_test_encode = y_test

                        preprocessor = DataPreprocessor(
                            num_cols=num_cols,
                            cat_cols=cat_cols
                        )

                        transformer = preprocessor.get_transformer(
                            scaler_type=select_Scaler_type,
                            impute=select_impute,
                            num_impute_strategy=select_num_impute_strategy,
                            cat_impute_strategy=select_cat_impute_strategy,
                            encoder_feature_type=select_feature_encoder
                        )

                        pipeline = ModelPipelineBuilder(
                            preprocessor=transformer
                        ).build_pipeline(
                            model_type=select_model_type,
                            model_params=model_params
                        )

                        task_type = ModelPipelineBuilder.MODEL_INFO[
                            select_model_type
                        ]

                        evaluator = Evaluator(
                            pipeline=pipeline,
                            task_type=task_type
                        )

                        metrics_df = evaluator.evaluate(
                            X_train,
                            y_train_encode,
                            X_test,
                            y_test_encode
                        )

                        cv_df = None

                        if enable_cv:

                            if select_target_encoder == "Label Encoder":

                                y_cv = label_encoder.fit_transform(y)

                            else:

                                y_cv = y

                            cv_df = evaluator.cross_validation(
                                X=X,
                                y=y_cv,
                                cv=cv_folds
                            )

                        st.success("Model trained successfully!")

                        st.markdown("---")
                        st.text(
                            "Result",
                            text_alignment="center",
                            width=1010
                        )

                        st.dataframe(metrics_df)

                        if cv_df is not None:
                            st.dataframe(cv_df)

                        # =============================
                        # Classification Plots
                        # =============================

                        if task_type == "classification":

                            with st.expander(
                                    "Show Confusion Matrix"
                            ):

                                st.plotly_chart(
                                    MetricPlot.plot_confusion_matrix(
                                        pipeline=pipeline,
                                        X_test=X_test,
                                        y_test=y_test_encode
                                    )
                                )

                            with st.expander(
                                    "Show ROC Curve"
                            ):

                                st.plotly_chart(
                                    MetricPlot.plot_roc_curve(
                                        pipeline=pipeline,
                                        X_test=X_test,
                                        y_test=y_test_encode
                                    )
                                )

                            with st.expander(
                                    "Learning Curve"
                            ):

                                st.plotly_chart(
                                    MetricPlot.plot_learning_curve(
                                        pipeline=pipeline,
                                        X=X,
                                        y=y.ravel(),
                                        cv=5,
                                        scoring="accuracy"
                                    )
                                )

                        # =============================
                        # Regression Plots
                        # =============================

                        elif task_type == "regression":

                            with st.expander(
                                    "Actual vs Predicted"
                            ):

                                st.plotly_chart(
                                    MetricPlot.plot_regression_fit(
                                        pipeline=pipeline,
                                        X_test=X_test,
                                        y_test=y_test_encode
                                    ),
                                    use_container_width=True
                                )

                            with st.expander(
                                    "Learning Curve"
                            ):

                                st.plotly_chart(
                                    MetricPlot.plot_learning_curve(
                                        pipeline=pipeline,
                                        X=X,
                                        y=y.ravel(),
                                        cv=5
                                    )
                                )
                # if "save_model_clicked" not in st.session_state:
                #     st.session_state.save_model_clicked = False
                #
                #
                # def save_model_action():
                #     st.session_state.save_model_clicked = True

                # with st.expander("Save model", expanded=True):
                #
                #     st.button(
                #         "Save Model",
                #         use_container_width=True,
                #         on_click=save_model_action
                #     )
                #
                #     if st.session_state.save_model_clicked:
                #         with st.spinner("Saving model..."):
                #             save_model(pipeline=pipeline)
                #
                #         st.success("Model saved successfully ✅")
                #         st.session_state.save_model_clicked = False


else:
    st.warning(
        "Please upload the data to the first page to activate this page",
        icon="⚠️"
    )
