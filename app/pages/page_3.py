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
    """
)