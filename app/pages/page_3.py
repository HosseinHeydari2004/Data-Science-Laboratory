import streamlit as st

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
    
    ----
    
    
    """
)

st.text("Model Configuration", text_alignment="center", width=700)

MODELS = {
    "supervised": {
        "classification": [
            "Logistic Regression", "Random Forest", "Support Vector Machine",
            "Knn", "Gaussian Naive Bayes", "Decision Tree", "Neural Network",
            "AdaBoost", "XGBBoost", "LightGBM", "Extra Tree", "CatBoost"
        ],
        "regression": [
            "Linear Regression", "Ridge Regression", "Random Forest Regressor",
            "XGBoost", "Lasso Regression", "ElasticNet", "Knn Regressor",
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
    options=[None, "supervised", "unsupervised"],
    index=0
)

if select_learning_type:
    task_options = list(MODELS[select_learning_type].keys())
    select_task_type = st.selectbox("Select Task Type", [None] + task_options)

    if select_task_type:
        model_options = MODELS[select_learning_type][select_task_type]
        select_model_type = st.selectbox("Select Model", [None] + model_options)
        select_Scaler_type = st.selectbox(
            "select Scaler Type",
            options=SCALER, index=0, key="select_Scaler_type"
        )
        select_feature_encoder = st.selectbox(
            "select Feature Encoder Type",
            options=[None] + FEATURE_ENCODER, index=0, key="select_feature_encoder"
        )
        select_target_encoder = st.selectbox(
            "Select target encoder",
            options=TARGET_ENCODER, key="select_target_encoder"
        )
        select_impute = st.selectbox(
            "Imputing",
            options=[False, True], index=0, key="select_impute"
        )
        if select_impute:
            select_num_impute_strategy = st.selectbox(
                "select num impute strategy",
                options=[
                    ""
                ]
            )


