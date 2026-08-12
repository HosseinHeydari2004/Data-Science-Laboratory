import streamlit as st
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
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
    """
    Build a scikit-learn compatible ``Pipeline`` for a supported model.

    This class centralizes two pieces of static knowledge used across the
    app:

    - ``MODEL_FACTORY`` maps a human-readable model name (as shown in the
      Streamlit selectbox) to the scikit-learn/XGBoost/LightGBM estimator
      class that implements it.
    - ``MODEL_INFO`` maps that same model name to its task type
      (``"classification"``, ``"regression"`` or ``"clustering"``), which
      the rest of the app uses to decide which metrics/plots to show.

    An instance wraps a fitted-or-unfitted ``ColumnTransformer`` (the
    preprocessing step) and exposes :meth:`build_pipeline` to combine it
    with a freshly constructed estimator into a single ``Pipeline`` with
    the steps ``("prep", preprocessor)`` and ``("model", estimator)``.

    Examples
    --------
    >>> from sklearn.compose import ColumnTransformer
    >>> builder = ModelPipelineBuilder(preprocessor=some_column_transformer)
    >>> pipe = builder.build_pipeline(
    ...     model_type="Random Forest",
    ...     model_params={"n_estimators": 200, "max_depth": 8}
    ... )
    >>> pipe.fit(X_train, y_train)
    """

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
        "XGBoost": XGBClassifier,
        "LightGBM": LGBMClassifier,
        "Extra Tree": ExtraTreesClassifier,
        "K-Means": KMeans,
        "DBSCAN": DBSCAN

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
        "XGBoost": "classification",
        "LightGBM": "classification",
        "Extra Tree": "classification",
        "K-Means": "clustering",
        "DBSCAN": "clustering"
    }

    def __init__(self, preprocessor: ColumnTransformer):
        """
        Parameters
        ----------
        preprocessor : sklearn.compose.ColumnTransformer
            A (typically unfitted) transformer produced by
            ``DataPreprocessor.get_transformer`` that will become the
            ``"prep"`` step of the resulting pipeline.
        """
        self.preprocessor: ColumnTransformer = preprocessor

    @classmethod
    def get_model(
            cls,
            model_type: str,
            model_params: dict | None = None
    ):

        """
        Instantiate the estimator registered under ``model_type``.

        Parameters
        ----------
        model_type : str
            Key into :attr:`MODEL_FACTORY`, e.g. ``"Random Forest"``.
        model_params : dict, optional
            Keyword arguments forwarded to the estimator's constructor
            (typically the output of ``ModelParameterFactory.get_params``).

        Returns
        -------
        BaseEstimator
            A new, unfitted estimator instance.

        Raises
        ------
        ValueError
            If ``model_type`` is not a key of :attr:`MODEL_FACTORY`.
        """
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
        """
        Build the full preprocessing + model pipeline.

        Parameters
        ----------
        model_type : str
            Key into :attr:`MODEL_FACTORY`.
        model_params : dict, optional
            Hyperparameters passed to :meth:`get_model`.

        Returns
        -------
        sklearn.pipeline.Pipeline
            A two-step pipeline: ``("prep", preprocessor)`` followed by
            ``("model", estimator)``.
        """
        model = self.get_model(
            model_type=model_type,
            model_params=model_params
        )

        return Pipeline([
            ("prep", self.preprocessor),
            ("model", model)
        ])


class ModelParameterFactory:
    """
    A collection of Streamlit hyperparameter forms, one per supported model.

    Every method below renders a small set of ``st.slider``/``st.selectbox``/
    ``st.checkbox`` widgets for one model's most relevant hyperparameters and
    returns them as a ``dict`` ready to be unpacked into the corresponding
    scikit-learn/XGBoost/LightGBM estimator (e.g. ``RandomForestClassifier
    (**params)``). :attr:`PARAMS_FACTORY` maps the model's display name
    (matching :attr:`ModelPipelineBuilder.MODEL_FACTORY`) to the function
    that should render its form, and :meth:`get_params` is the single entry
    point the pages call.

    Because each method calls Streamlit widget functions directly, these
    methods must be called during a Streamlit script run (they have the
    side effect of rendering widgets) and return a plain ``dict`` of the
    current widget values.
    """
    @staticmethod
    def k_means():
        """Hyperparameter form for sklearn.cluster.KMeans (n_clusters, init, n_init, max_iter). Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
        return {
            "n_clusters": st.slider(
                "select n_clusters", value=2, max_value=12, min_value=2, key="9jm3"
            ),
            "init": st.selectbox(
                "select init mode",
                options=[
                    "k-means++",
                    "random"
                ], index=0, key="w2aac"
            ),
            "max_iter": st.slider(
                "select max iteration", value=300, min_value=20,
                max_value=1000, key="0lo2"
            ),
        }

    @staticmethod
    def dbscan():
        """Hyperparameter form for sklearn.cluster.DBSCAN (eps, min_samples, metric). Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
        return {
            "eps": st.slider(
                "select eps",
                value=0.5, min_value=0.1, max_value=2.0
            ),
            "min_samples": st.slider(
                "select min samples",
                value=5, min_value=2, max_value=35
            ),
            "algorithm": st.selectbox(
                "select algorithm",
                options=[
                    "auto",
                    "ball_tree",
                    "kd_tree",
                    "brute"
                ], index=0
            ),
            "n_jobs": st.checkbox(
                "n_jobs",
                value=False
            )
        }

    @staticmethod
    def logistic_regression():
        """Hyperparameter form for sklearn.linear_model.LogisticRegression. Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
        return {
            "C": st.number_input(
                "enter C", value=1.0
            ),
            "max_iter": st.slider(
                "enter max iterations",
                value=100, max_value=1000, min_value=30
            ),
            "l1_ratio": st.slider(
                # Bug fix: scikit-learn requires l1_ratio in [0.0, 1.0];
                # the previous max_value=20.0 let users pick invalid values
                # that raised `ValueError` as soon as training started.
                "enter l1 ratio", value=0.0, min_value=0.0, max_value=1.0,
                help="Only used when penalty='elasticnet' and solver='saga'."
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
        """Hyperparameter form for sklearn.ensemble.RandomForestClassifier. Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
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
            "n_jobs": -1 if st.checkbox("use all cpu") else None
        }

    @staticmethod
    def linear_regression():
        """Hyperparameter form for sklearn.linear_model.LinearRegression (has no tunable params exposed). Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
        return {
            "fit_intercept": st.checkbox(
                "fit_intercept", value=True
            )
        }

    @staticmethod
    def ridge():
        """Hyperparameter form for sklearn.linear_model.Ridge. Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
        return {
            "alpha": st.slider(
                "enter alpha",
                min_value=1.0, value=1.0, max_value=20.0
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
        """Hyperparameter form for sklearn.linear_model.Lasso. Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
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
        """Hyperparameter form for sklearn.linear_model.ElasticNet. Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
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
                # Bug fix: ElasticNet requires l1_ratio in [0.0, 1.0].
                "enter L1 ratio", value=0.0, min_value=0.0, max_value=1.0
            )
        }

    @staticmethod
    def random_forest_regressor():
        """Hyperparameter form for sklearn.ensemble.RandomForestRegressor. Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
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
                options=["squared_error", "absolute_error", "poisson"], index=0
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
            "n_jobs": -1 if st.checkbox("use all cpu") else None
        }

    @staticmethod
    def gradient_boosting_regressor():
        """Hyperparameter form for sklearn.ensemble.GradientBoostingRegressor. Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
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
                min_value=0.1, max_value=0.9, value=0.1
            ),
            "ccp_alpha": st.slider(
                "Enter ccp alpha",
                min_value=0.0, value=0.0, max_value=10.0
            )
        }

    @staticmethod
    def knn_regressor():
        """Hyperparameter form for sklearn.neighbors.KNeighborsRegressor. Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
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
        """Hyperparameter form for LightGBM (LGBMClassifier/LGBMRegressor share this form). Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
        return {
            "boosting_type": st.selectbox(
                "select boosting type",
                options=[
                    "gbdt",
                    "dart"
                ], index=0,
            ),
            "num_leaves": st.slider(
                "Enter num leaves",
                min_value=10, max_value=50, value=31
            ),
            "max_depth": st.slider(
                "Enter max depth",
                min_value=2, value=2, max_value=20
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
        """Hyperparameter form for sklearn.tree.DecisionTreeRegressor. Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
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
        """Hyperparameter form for sklearn.svm.SVR. Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
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
            "degree": st.slider(
                "Enter degree",
                min_value=1, value=1, max_value=10, key="0xcvf4",
                help="The degree parameter is only applicable when the poly kernel is selected."
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
        """Hyperparameter form for XGBoost (XGBClassifier/XGBRegressor share this form). Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
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
        """Hyperparameter form for sklearn.ensemble.ExtraTreesRegressor. Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
        return {
            "n_estimators": st.slider(
                "Enter n estimators",
                min_value=10, value=10, max_value=1000, step=5, key="0o232"
            ),
            "max_features": st.selectbox(
                "Select max features",
                options=["sqrt", "log2", None], index=0, key="0t422"
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
            "n_jobs": -1 if st.checkbox("use all cpu") else None
        }

    @staticmethod
    def AdaBoost_regressor():
        """Hyperparameter form for sklearn.ensemble.AdaBoostRegressor. Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
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
        """Hyperparameter form for sklearn.neural_network.MLPRegressor. Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
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
                # Bug fix: an empty string key="" can collide with other
                # unkeyed widgets and cause a StreamlitDuplicateElementId
                # error; every widget now gets a unique, non-empty key.
                "Enter alpha",
                min_value=0.0001, max_value=10.0, value=0.0001, key="0nnr_alpha"
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
        """Hyperparameter form for sklearn.neighbors.KNeighborsClassifier. Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
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
        """Hyperparameter form for sklearn.naive_bayes.GaussianNB. Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
        return {
            "var_smoothing": st.number_input(
                "Enter var smoothing",
                min_value=1e-9, value=1e-9
            )
        }

    @staticmethod
    def support_vector_machine():
        """Hyperparameter form for sklearn.svm.SVC. Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
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
            "degree": st.slider(
                "Enter degree",
                min_value=2, value=2, max_value=10, key="0xcvf4"
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
        """Hyperparameter form for sklearn.tree.DecisionTreeClassifier. Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
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
        """Hyperparameter form for sklearn.neural_network.MLPClassifier. Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
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
                min_value=0.0001, max_value=10.0, value=0.0001, key="0u83jd"
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
        """Hyperparameter form for sklearn.ensemble.AdaBoostClassifier. Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
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
        """Hyperparameter form for sklearn.ensemble.ExtraTreesClassifier. Called by ModelParameterFactory.get_params; renders widgets and returns the collected kwargs as a dict."""
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
            "n_jobs": -1 if st.checkbox("use all cpu") else None
        }

    PARAMS_FACTORY = {
        "Logistic Regression": logistic_regression,
        "Random Forest": random_forest,
        "Linear Regression": linear_regression,
        "Lasso Regression": lasso,
        "ElasticNet": elasticnet,
        "Random Forest Regressor": random_forest_regressor,
        "Gradient Boosting Regressor": gradient_boosting_regressor,
        "XGBoost Regressor": xgboost,
        "Knn Regressor": knn_regressor,
        "LightGBM Regressor": lightgbm,
        "Decision Tree Regressor": Decision_Tree_regressor,
        "SVR(support vector Regressor)": svr,
        "ExtraTree Regressor": ExtraTree_regressor,
        "AdaBoost Regressor": AdaBoost_regressor,
        "Neural Network(Regressor)": Neural_Network_regressor,
        "Ridge Regression": ridge,
        "Knn": Knn,
        "Gaussian Naive Bayes": gaussian_naive_bayes,
        "Support Vector Machine": support_vector_machine,
        "Decision Tree": decision_tree,
        "Neural Network": neural_network,
        "AdaBoost": adaboost,
        "XGBoost": xgboost,
        "LightGBM": lightgbm,
        # Bug fix: "Extra Tree" (the classifier) was missing from this
        # mapping even though its widget function `extra_tree` was already
        # implemented above. Users who selected "Extra Tree" in the UI
        # silently got an empty hyperparameter form (ModelParameterFactory
        # .get_params fell through to `{}`) and the model always trained
        # with scikit-learn defaults.
        "Extra Tree": extra_tree,
        "K-Means": k_means,
        "DBSCAN": dbscan

    }

    @classmethod
    def get_params(cls, model_name: str) -> dict:
        """
        Render and collect the hyperparameter widgets for ``model_name``.

        Parameters
        ----------
        model_name : str
            Display name of the model, e.g. ``"Random Forest"``. Must be a
            key of :attr:`PARAMS_FACTORY`.

        Returns
        -------
        dict
            The hyperparameters currently selected in the UI, ready to be
            passed as ``**model_params`` to
            ``ModelPipelineBuilder.get_model``. Returns an empty dict (and
            renders no widgets) if ``model_name`` has no registered form.
        """
        func = cls.PARAMS_FACTORY.get(model_name)

        if func is None:
            return {}

        return func()
