import streamlit as st
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier
from xgboost import XGBRegressor


class ModelPipelineBuilder:
    MODEL_FACTORY = {
        "Linear Regression": LinearRegression,
        "Ridge Regression": Ridge,
        "Lasso Regression": Lasso,
        "ElasticNet": ElasticNet,
        "Random Forest Regressor": RandomForestRegressor,
        "Gradient Boosting Regressor": GradientBoostingRegressor,
        "XGBoost Regressor": XGBRegressor,
        "LightGBM Regressor": LGBMRegressor,
        "Knn Regressor": KNeighborsRegressor,
        "Decision Tree Regressor": DecisionTreeRegressor,
        "SVR(support vector Regressor)": SVR,
        "ExtraTree Regressor": ExtraTreesRegressor,
        "AdaBoost Regressor": AdaBoostRegressor,
        "Neural Network(Regressor)": MLPRegressor,
        "Logistic Regression": LogisticRegression,
        "Random Forest": RandomForestClassifier,
        "Support Vector Machine": SVC,
        "Knn": KNeighborsClassifier,
        "Gaussian Naive Bayes": GaussianNB,
        "Decision Tree": DecisionTreeClassifier,
        "Neural Network": MLPClassifier,
        "AdaBoost": AdaBoostClassifier,
        "XGBBoost": XGBClassifier,
        "LightGBM": LGBMClassifier,
        "Extra Tree": ExtraTreesClassifier
    }
    MODEL_INFO = {

        # =====================
        # Regression
        # =====================

        "Linear Regression": "regression",
        "Ridge Regression": "regression",
        "Lasso Regression": "regression",
        "ElasticNet": "regression",
        "Random Forest Regressor": "regression",
        "Gradient Boosting Regressor": "regression",
        "XGBoost Regressor": "regression",
        "LightGBM Regressor": "regression",
        "Knn Regressor": "regression",
        "Decision Tree Regressor": "regression",
        "SVR(support vector Regressor)": "regression",
        "ExtraTree Regressor": "regression",
        "AdaBoost Regressor": "regression",
        "Neural Network(Regressor)": "regression",

        # =====================
        # Classification
        # =====================

        "Logistic Regression": "classification",
        "Random Forest": "classification",
        "Support Vector Machine": "classification",
        "Knn": "classification",
        "Gaussian Naive Bayes": "classification",
        "Decision Tree": "classification",
        "Neural Network": "classification",
        "AdaBoost": "classification",
        "XGBBoost": "classification",
        "LightGBM": "classification",
        "Extra Tree": "classification"
    }

    def __init__(self, preprocessor: ColumnTransformer):
        self.preprocessor: ColumnTransformer = preprocessor

    @classmethod
    def get_model(
            cls,
            model_type: str,
            model_params: dict | None = None
    ):

        if model_params is None:
            model_params = {}

        model_class = cls.MODEL_FACTORY.get(model_type)

        if model_class is None:
            raise ValueError(
                f"Unsupported model type: {model_type}"
            )

        return model_class(**model_params)

    def build_pipeline(
            self,
            model_type: str,
            model_params: dict | None = None
    ) -> Pipeline:

        model = self.get_model(
            model_type=model_type,
            model_params=model_params
        )

        return Pipeline([
            ("prep", self.preprocessor),
            ("model", model)
        ])


class ModelParameterFactory:
    @staticmethod
    def logistic_regression():
        return {
            "C": st.number_input(
                "enter C", value=1.0
            ),
            "max_iter": st.slider(
                "enter max iterations",
                value=100, max_value=1000, min_value=30
            ),
            "l1_ratio": st.slider(
                "enter l1 ratio",
                value=0.0, min_value=0.0, max_value=20.0
            ),
            "solver": st.selectbox(
                "select solver",
                options=["lbfgs", "liblinear",
                         "newton-cg", "newton-cholesky",
                         "sag", "saga"
                         ]
            ),
            "penalty": st.selectbox(
                "select penalty",
                options=["l2", "l1", "elasticnet", None], index=0
            ),
            "class_weight": st.selectbox(
                "select class weight",
                options=["balanced", None], index=1
            )
        }

    @staticmethod
    def random_forest():
        return {
            "n_estimators": st.slider(
                "enter n estimators",
                min_value=100, max_value=1000, value=100
            ),
            "max_depth": st.slider(
                "enter max depth",
                min_value=2, max_value=15, value=None
            ),
            "criterion": st.selectbox(
                "select criterion",
                options=["gini", "entropy", "log_loss"], index=0
            ),
            "min_samples_split": st.slider(
                "enter min samples split",
                min_value=2, value=2, max_value=50
            ),
            "min_samples_leaf": st.slider(
                "enter min samples leaf",
                min_value=1, value=1, max_value=50
            ),
            "max_features": st.selectbox(
                "select max features",
                options=["sqrt", "log2", None], index=0
            ),
            "max_leaf_nodes": st.slider(
                "enter max leaf nodes",
                min_value=5, value=None, max_value=50
            ),
            "bootstrap": st.checkbox(
                "bootstrap", value=False
            ),
            "n_jobs": st.checkbox(
                "n_jobs", value=-1
            )
        }

    @staticmethod
    def linear_regression():
        return {
            "fit_intercept": st.checkbox(
                "fit_intercept", value=True
            )
        }

    @staticmethod
    def ridge():
        return {
            "alpha": st.slider(
                "enter alpha",
                min_value=1.0, value=1.0, max_value=20
            ),
            "max_iter": st.slider(
                "enter max iterations",
                min_value=100, value=None, max_value=1000
            ),
            "solver": st.selectbox(
                "select solver",
                options=[
                    "auto",
                    "svd",
                    "cholesky",
                    "lsqr",
                    "sparse_cg",
                    "sag",
                    "saga",
                    "lbfgs"
                ], index=0
            ),

        }

    @staticmethod
    def lasso():
        return {
            "alpha": st.slider(
                "Enter alpha",
                min_value=1.0, max_value=10.0, value=1.0
            ),
            "max_iter": st.slider(
                "Enter max iteration",
                min_value=100, max_value=1000, value=300
            ),
            "selection": st.selectbox(
                "select selection",
                options=[
                    "cyclic",
                    "random"
                ], index=0
            )
        }

    @staticmethod
    def elasticnet():
        return {
            "alpha": st.slider(
                "Enter alpha",
                min_value=1.0, max_value=10.0, value=1.0
            ),
            "max_iter": st.slider(
                "Enter max iteration",
                min_value=100, max_value=1000, value=300
            ),
            "selection": st.selectbox(
                "select selection",
                options=[
                    "cyclic",
                    "random"
                ], index=0
            ),
            "l1_ratio": st.slider(
                "enter L1 ratio",
                value=0.0, min_value=0.0, max_value=20.0
            )
        }

    @staticmethod
    def random_forest_regressor():
        return {
            "n_estimators": st.slider(
                "enter n estimators",
                min_value=100, max_value=1000, value=100
            ),
            "max_depth": st.slider(
                "enter max depth",
                min_value=2, max_value=15, value=None
            ),
            "criterion": st.selectbox(
                "select criterion",
                options=["gini", "entropy", "log_loss"], index=0
            ),
            "min_samples_split": st.slider(
                "enter min samples split",
                min_value=2, value=2, max_value=50
            ),
            "min_samples_leaf": st.slider(
                "enter min samples leaf",
                min_value=1, value=1, max_value=50
            ),
            "max_features": st.selectbox(
                "select max features",
                options=["sqrt", "log2", None], index=0
            ),
            "max_leaf_nodes": st.slider(
                "enter max leaf nodes",
                min_value=5, value=None, max_value=50
            ),
            "bootstrap": st.checkbox(
                "bootstrap", value=False
            ),
            "n_jobs": st.checkbox(
                "n_jobs", value=-1
            )
        }

    @staticmethod
    def gradient_boosting_regressor():
        return {
            "loss": st.selectbox(
                "select loss function",
                options=[
                    "squared_error",
                    "absolute_error",
                    "huber",
                    "quantile"
                ], index=0
            ),
            "learning_rate": st.slider(
                "Enter learning rate",
                min_value=0.1, value=0.1, max_value=10.0
            ),
            "n_estimators": st.slider(
                "enter n estimators",
                min_value=100, max_value=1000, value=100, step=10
            ),
            "max_depth": st.slider(
                "enter max depth",
                min_value=3, max_value=15, value=3
            ),
            "criterion": st.selectbox(
                "select criterion",
                options=[
                    "friedman_mse",
                    "squared_error"
                ], index=0
            ),
            "min_samples_split": st.slider(
                "enter min samples split",
                min_value=2, value=2, max_value=50
            ),
            "min_samples_leaf": st.slider(
                "enter min samples leaf",
                min_value=1, value=1, max_value=50
            ),
            "max_leaf_nodes": st.slider(
                "enter max leaf nodes",
                min_value=5, value=None, max_value=50
            ),
            "max_features": st.selectbox(
                "select max features",
                options=["sqrt", "log2", None], index=2
            ),
            "alpha": st.slider(
                "Enter alpha",
                min_value=0.9, max_value=10.0, value=0.9
            ),
            "ccp_alpha": st.slider(
                "Enter ccp alpha",
                min_value=0.0, value=0.0, max_value=10.0
            )
        }

    @staticmethod
    def knn_regressor():
        return {
            "n_neighbors": st.slider(
                "select n_neighbors",
                min_value=2, max_value=25, value=2
            ),
            "weights": st.selectbox(
                "select weights",
                options=[
                    "uniform",
                    "distance"
                ], index=0
            ),
            "algorithm": st.selectbox(
                "select algorithm",
                options=[
                    "auto",
                    "ball_tree",
                    "kd_tree",
                    "brute"
                ], index=0
            )
        }

    @staticmethod
    def lightgbm():
        return {
            "boosting_type": st.selectbox(
                "select boosting type",
                options=[
                    "gbdt",
                    "dart",
                    "rf"
                ], index=0,
            ),
            "num_leaves": st.slider(
                "Enter num leaves",
                min_value=10, max_value=50, value=31
            ),
            "max_depth": (st.slider(
                "Enter max depth",
                min_value=-1, value=-1, max_value=20
            ),
                          st.info(
                              "When max_depth is set to -1, "
                              "the algorithm explores the tree completely, "
                              "down to the deepest leaf nodes.",
                              icon="ℹ️"
                          )
            ),
            "learning_rate": st.slider(
                "Enter learning rate",
                min_value=0.001, max_value=5.0, value=0.1, step=0.001
            ),
            "n_estimators": st.slider(
                "Enter n estimators",
                min_value=20, value=100, max_value=3000, step=5
            ),
            "class_weight": st.selectbox(
                "select class weight",
                options=[
                    None,
                    "balanced"
                ], index=0
            ),
            "reg_alpha": st.slider(
                "Enter reg alpha",
                min_value=0.0, value=0.0, max_value=10.0, key="0x23dd"
            ),
            "reg_lambda": st.slider(
                "Enter reg alpha",
                min_value=0.0, value=0.0, max_value=10.0, key="0xrttv"
            )
        }

    @staticmethod
    def Decision_Tree_regressor():
        return {
            "criterion": st.selectbox(
                "select criterion",
                options=[
                    "squared_error",
                    "friedman_mse",
                    "absolute_error",
                    "poisson"
                ], index=0, key="0x9ow3"
            ),
            "splitter": st.selectbox(
                "select splitter",
                options=[
                    "best",
                    "random"
                ], index=0, key="0xcvd2"
            ),
            "max_depth": st.slider(
                "Enter max depth",
                min_value=2, value=None, max_value=20
            ),
            "min_samples_split": st.slider(
                "Enter min samples split",
                min_value=2, value=2, max_value=50
            ),
            "min_samples_leaf": st.slider(
                "Enter min samples leaf",
                min_value=1, value=1, max_value=50
            ),
            "max_features": st.selectbox(
                "select max features",
                options=[
                    None,
                    "sqrt",
                    "log2"
                ], index=0, key="0xase1"
            ),
            "ccp_alpha": st.slider(
                "Enter ccp alpha",
                min_value=0.0, max_value=10.0, value=0.0
            )

        }

    @staticmethod
    def svr():
        return {
            "kernel": st.selectbox(
                "select kernel",
                options=[
                    "linear",
                    "poly",
                    "rbf",
                    "sigmoid"
                ], index=2, key="0x4562lp"
            ),
            "degree": (st.slider(
                "Enter degree",
                min_value=2, value=2, max_value=10, key="0xcvf4"
            ),
                       st.info(
                           "The degree parameter is only applicable when the poly kernel is selected.",
                           icon="ℹ️"
                       )
            ),
            "gamma": st.selectbox(
                "select gamma",
                options=[
                    "scale",
                    "auto"
                ], index=0, key="0x.09"
            ),
            "C": st.slider(
                "Enter C",
                min_value=1.0, value=1.0, max_value=20.0, key="0x./l"
            ),
            "shrinking": st.checkbox(
                "shrinking", value=False
            )
        }

    @staticmethod
    def xgboost():
        return {
            "n_estimators": st.slider(
                "Enter n estimators",
                min_value=20, value=100, max_value=3000, key="0cdf"
            ),
            "max_depth": st.slider(
                "Enter max depth",
                min_value=2, value=None, max_value=20, key="02ws"
            ),
            "learning_rate": st.slider(
                "Enter learning rate",
                min_value=0.001, max_value=5.0, value=0.1, step=0.001, key="0l223"
            ),
            "subsample": st.slider(
                "Enter subsample",
                min_value=0.2, max_value=1.0, step=0.1, value=0.5, key="0w23"
            ),
            "gamma": st.slider(
                "Enter gamma",
                min_value=0.0, max_value=10.0, step=0.1, value=0.0
            )
        }

    @staticmethod
    def ExtraTree_regressor():
        return {
            "n_estimators": st.slider(
                "Enter n estimators",
                min_value=10, value=10, max_value=1000, step=5, key="0o232"
            ),
            "max_features": st.selectbox(
                "Select max features",
                options=[
                    "auto",
                    "sqrt",
                    "log2",
                    None
                ], index=0, key="0t422"
            ),
            "max_depth": st.slider(
                "Enter max depth",
                min_value=2, value=None, max_value=20, key="02ws"
            ),
            "min_samples_split": st.slider(
                "Enter min samples split",
                min_value=2, value=2, max_value=50
            ),
            "bootstrap": st.checkbox(
                "bootstrap",
                value=False, key="0uw2"
            ),
            "oob_score": st.checkbox(
                "oob score",
                value=False, key="0p92"
            ),
            "n_jobs": st.checkbox(
                "n jobs", value=-1, key="0ore"
            )
        }

    @staticmethod
    def AdaBoost_regressor():
        return {
            "n_estimators": st.slider(
                "Enter n estimators",
                min_value=10, value=50, max_value=1000, step=5, key="0o232"
            ),
            "learning_rate": st.slider(
                "Enter learning rate",
                min_value=0.001, max_value=5.0, value=0.1, step=0.001, key="0l223"
            ),
            "loss": st.selectbox(
                "Select loss Function",
                options=[
                    "linear",
                    "square",
                    "exponential"
                ], index=0, key="0k232"
            ),
        }

    @staticmethod
    def Neural_Network_regressor():
        return {
            "loss": st.selectbox(
                "Select loss",
                options=[
                    "squared_error",
                    "poisson"
                ], index=0, key="0u29"
            ),
            "hidden_layer_sizes": st.slider(
                "Enter hidden layer sizes",
                min_value=5, value=100, max_value=500, key="0u322s"
            ),
            "activation": st.selectbox(
                "Select activation",
                options=[
                    "identity",
                    "logistic",
                    "tanh",
                    "relu"
                ]
            ),
            "solver": st.selectbox(
                "Select solver",
                options=[
                    "lbfgs",
                    "sgd",
                    "adam"
                ], index=2, key="0urw"
            ),
            "alpha": st.slider(
                "Enter alpha",
                min_value=0.0001, max_value=10.0, value=0.0001, key=""
            ),
            "learning_rate": st.selectbox(
                "Select learning rate",
                options=[
                    "constant",
                    "invscaling",
                    "adaptive"
                ], index=0, key="0er2"
            ),
            "learning_rate_init": st.slider(
                "Enter learning rate",
                min_value=0.001, max_value=5.0, value=0.001
            ),
            "max_iter": st.slider(
                "Enter max iteration",
                min_value=10, max_value=2000, value=200, key="li92"
            ),
            "early_stopping": st.checkbox(
                "early stopping", value=False, key="0po2"
            )
        }

    @staticmethod
    def Knn():
        return {
            "n_neighbors": st.slider(
                "select n_neighbors",
                min_value=2, max_value=25, value=2
            ),
            "weights": st.selectbox(
                "select weights",
                options=[
                    "uniform",
                    "distance"
                ], index=0
            ),
            "algorithm": st.selectbox(
                "select algorithm",
                options=[
                    "auto",
                    "ball_tree",
                    "kd_tree",
                    "brute"
                ], index=0
            )
        }

    @staticmethod
    def gaussian_naive_bayes():
        return {
            "var_smoothing": st.number_input(
                "Enter var smoothing",
                min_value=1e-9, value=1e-9
            )
        }

    @staticmethod
    def support_vector_machine():
        return {
            "kernel": st.selectbox(
                "select kernel",
                options=[
                    "linear",
                    "poly",
                    "rbf",
                    "sigmoid"
                ], index=2, key="0x4562lp"
            ),
            "degree": (st.slider(
                "Enter degree",
                min_value=2, value=2, max_value=10, key="0xcvf4"
            ),
                       st.info(
                           "The degree parameter is only applicable when the poly kernel is selected.",
                           icon="ℹ️"
                       )
            ),
            "gamma": st.selectbox(
                "select gamma",
                options=[
                    "scale",
                    "auto"
                ], index=0, key="0x.09"
            ),
            "C": st.slider(
                "Enter C",
                min_value=1.0, value=1.0, max_value=20.0, key="0x./l"
            ),
            "shrinking": st.checkbox(
                "shrinking", value=False
            )
        }

    @staticmethod
    def decision_tree():
        return {
            "criterion": st.selectbox(
                "select criterion",
                options=[
                    "gini",
                    "entropy",
                    "log_loss"
                ], index=0, key="0o2s"
            ),
            "splitter": st.selectbox(
                "select splitter",
                options=[
                    "best",
                    "random"
                ], index=0, key="0iu7"
            ),
            "max_depth": st.slider(
                "Enter max depth",
                min_value=2, value=None, max_value=20
            ),
            "min_samples_split": st.slider(
                "Enter min samples split",
                min_value=2, value=2, max_value=50
            ),
            "min_samples_leaf": st.slider(
                "Enter min samples leaf",
                min_value=1, value=1, max_value=50
            ),
            "max_features": st.selectbox(
                "select max features",
                options=[
                    None,
                    "sqrt",
                    "log2"
                ], index=0, key="0ye72"
            ),
            "ccp_alpha": st.slider(
                "Enter ccp alpha",
                min_value=0.0, max_value=10.0, value=0.0
            )

        }

    @staticmethod
    def neural_network():
        return {
            "hidden_layer_sizes": st.slider(
                "Enter hidden layer sizes",
                min_value=5, value=100, max_value=500, key="0u322s"
            ),
            "activation": st.selectbox(
                "Select activation",
                options=[
                    "identity",
                    "logistic",
                    "tanh",
                    "relu"
                ]
            ),
            "solver": st.selectbox(
                "Select solver",
                options=[
                    "lbfgs",
                    "sgd",
                    "adam"
                ], index=2, key="0urw"
            ),
            "alpha": st.slider(
                "Enter alpha",
                min_value=0.0001, max_value=10.0, value=0.0001, key=""
            ),
            "learning_rate": st.selectbox(
                "Select learning rate",
                options=[
                    "constant",
                    "invscaling",
                    "adaptive"
                ], index=0, key="0er2"
            ),
            "learning_rate_init": st.slider(
                "Enter learning rate",
                min_value=0.001, max_value=5.0, value=0.001
            ),
            "max_iter": st.slider(
                "Enter max iteration",
                min_value=10, max_value=2000, value=200, key="li92"
            ),
            "early_stopping": st.checkbox(
                "early stopping", value=False, key="0po2"
            )
        }

    @staticmethod
    def adaboost():
        return {
            "n_estimators": st.slider(
                "Enter n estimators",
                min_value=10, value=50, max_value=1000, step=5, key="0o232"
            ),
            "learning_rate": st.slider(
                "Enter learning rate",
                min_value=0.001, max_value=5.0, value=0.1, step=0.001, key="0l223"
            )
        }

    @staticmethod
    def extra_tree():
        return {
            "n_estimators": st.slider(
                "Enter n estimators",
                min_value=10, value=10, max_value=1000, step=5, key="0o232"
            ),
            "criterion": st.selectbox(
                "select criterion",
                options=["gini", "entropy", "log_loss"], index=0
            ),
            "max_features": st.selectbox(
                "Select max features",
                options=[
                    "auto",
                    "sqrt",
                    "log2",
                    None
                ], index=0, key="0t422"
            ),
            "max_depth": st.slider(
                "Enter max depth",
                min_value=2, value=None, max_value=20, key="02ws"
            ),
            "min_samples_split": st.slider(
                "Enter min samples split",
                min_value=2, value=2, max_value=50
            ),
            "bootstrap": st.checkbox(
                "bootstrap",
                value=False, key="0uw2"
            ),
            "oob_score": st.checkbox(
                "oob score",
                value=False, key="0p92"
            ),
            "n_jobs": st.checkbox(
                "n jobs", value=-1, key="0ore"
            )
        }

    PARAMS_FACTORY = {
        "Logistic Regression": logistic_regression,
        "Random Forest": random_forest,
        "Linear Regression": linear_regression,
        "Lasso Regression": lasso,
        "ElasticNet": elasticnet,
        "Random Forest Regressor": random_forest_regressor,
        "Gradient Boosting Regressor": gradient_boosting_regressor,
        "XGBBoost Regressor": xgboost,
        "Knn Regressor": knn_regressor,
        "LightGBM Regressor": lightgbm,
        "Decision Tree Regressor": Decision_Tree_regressor,
        "SVR(support vector Regressor)": svr,
        "ExtraTree Regressor": ExtraTree_regressor,
        "AdaBoost Regressor": AdaBoost_regressor,
        "Neural Network(Regressor)": Neural_Network_regressor,
        "Knn": Knn,
        "Gaussian Naive Bayes": gaussian_naive_bayes,
        "Support Vector Machine": support_vector_machine,
        "Decision Tree": decision_tree,
        "Neural Network": neural_network,
        "AdaBoost": adaboost,
        "XGBBoost": xgboost,
        "LightGBM": lightgbm

    }

    @classmethod
    def get_params(cls, model_name):
        func = cls.PARAMS_FACTORY.get(model_name)

        if func is None:
            return {}

        return func()
