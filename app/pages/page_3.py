import streamlit as st

from Core.Preprocessor import DataPreprocessor
from Core.eda import EDA
from Core.evaluator import Evaluator
from Core.model_trainer import ModelParameterFactory, ModelPipelineBuilder

st.markdown(
    """
    # 🔧 Machine Learning Modeling

This page is the **Modeling** section of the Streamlit app, designed to train, evaluate, and compare various machine learning models on your processed dataset.
All tasks are organized as **accordions/expanders** to keep the interface clean and interactive.

## Sections

*   **Data Preparation & Splitting**
    *   Defines features (X) and target (y) variables.
    *   Allows configuration of test/train split ratio and random state for reproducibility.
*   **Model Selection**
    *   Dropdown menu to choose from various algorithms (e.g., Random Forest, XGBoost, Linear Regression, etc.).
    *   Hyperparameter tuning interface for selected models to optimize performance.
*   **Training Process**
    *   Triggers the model training pipeline.
    *   Displays real-time training progress and execution logs.
*   **Model Evaluation**
    *   Generates performance metrics based on the task (e.g., Accuracy, F1-Score, RMSE, R²).
    *   Includes confusion matrix and ROC-AUC curve visualization for classification tasks.
*   **Feature Importance**
    *   Visualizes which features have the most significant impact on model predictions.
*   **Save/Download Model**
    *   Allows you to export the trained model as a joblib file (`.joblib`) for deployment or future use.
*   **Compare Results**
    *   Provides a summary table to compare performance metrics across different runs or models.

---


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
                "AdaBoost", "XGBBoost", "LightGBM", "Extra Tree"
            ],
            "regression": [
                "Linear Regression", "Ridge Regression", "Random Forest Regressor",
                "Lasso Regression", "ElasticNet", "Gradient Boosting Regressor",
                "Knn Regressor",
                "Decision Tree Regressor", "SVR(support vector Regressor)",
                "ExtraTree Regressor", "AdaBoost Regressor", "XGBBoost Regressor",
                "Neural Network(Regressor)", "LightGBM Regressor",

            ]
        },
        "unsupervised": {
            "clustering": ["K-Means", "DBSCAN"],
            "dimensionality_reduction": ["PCA", "t-SNE"]
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
        "Target Encoder"
    ]
    TARGET_ENCODER = [
        "Label Encoder",
        "None"
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
            st.markdown(
                """
                ---
        
                """
            )
            st.text("Model Parameter", text_alignment="center", width=1000)
            model_params = ModelParameterFactory.get_params(
                select_model_type
            )
            train_btn = st.button(
                "🚀 Train Model",
                use_container_width=True
            )
            if train_btn:
                with st.spinner(
                        "Training model..."
                ):
                    # =========================
                    # 1. Split Data
                    # =========================
                    X = df.drop(columns=[select_target])
                    y = df[select_target]
                    num_cols = EDA.detect_numeric_type(data=df)
                    cat_cols = EDA.detect_object_type(data=df)

                    X_train, X_test, y_train, y_test = DataPreprocessor.set_setting_split(
                        data=df,
                        feature_cols=num_cols + cat_cols,
                        target_col=select_target,
                        test_size=select_test_size / 100,
                        stratify=select_stratify
                    )

                    # =========================
                    # 2. Preprocessor
                    # =========================

                    preprocessor = DataPreprocessor(
                        num_cols=num_cols,
                        cat_cols=cat_cols
                    )

                    transformer = preprocessor.get_transformer(
                        scaler_type=select_Scaler_type,
                        impute=select_impute,
                        num_impute_strategy=select_num_impute_strategy if select_num_impute_strategy else None,
                        cat_impute_strategy=select_cat_impute_strategy if select_cat_impute_strategy else None,
                        encoder_feature_type=select_feature_encoder
                    )

                    # =========================
                    # 3. Build Pipeline
                    # =========================

                    pipeline_builder = ModelPipelineBuilder(
                        preprocessor=transformer
                    )

                    pipeline = pipeline_builder.build_pipeline(
                        model_type=select_model_type,
                        model_params=model_params
                    )

                    # =========================
                    # 4. Task Type
                    # =========================

                    task_type = ModelPipelineBuilder.MODEL_INFO[
                        select_model_type
                    ]

                    # =========================
                    # 5. Evaluator
                    # =========================

                    evaluator = Evaluator(
                        pipeline=pipeline,
                        task_type=task_type
                    )

                    # =========================
                    # 6. Train + Evaluate
                    # =========================

                    metrics_df = evaluator.evaluate(
                        X_train,
                        y_train,
                        X_test,
                        y_test
                    )
                    # =========================
                    # 7. Cross Validation (optional)
                    # =========================

                    cv_df = None
                    if enable_cv:
                        cv_df = evaluator.cross_validation(
                            feature_cols=num_cols + cat_cols,
                            data=df,
                            target_cols=select_target,
                            cv=cv_folds
                        )

                    # =========================
                    # 8. Store Model
                    # =========================
                    trained_model = pipeline
                    st.success("Model trained successfully!")
                st.text("Result", text_alignment="center", width=1010)
                st.markdown(
                    """
                    ---
                    
                    """
                )
                st.dataframe(metrics_df)
                st.dataframe(cv_df)




else:
    st.warning(
        "Please upload the data to the first page to activate this page",
        icon="⚠️"
    )
