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
