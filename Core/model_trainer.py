from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


class ModelPipelineBuilder:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
    @staticmethod
    def get_model(
            model_type: str,
            model_params: dict | None = None
    ):

        if model_params is None:
            model_params = {}
        if model_type == "Linear Regression":
            model = LinearRegression(**model_params)
        elif model_type == "Random Forest":
            model = RandomForestRegressor(**model_params)
        elif model_type == "XGBoost":
            model = XGBRegressor(**model_params)
        elif model_type == "GradientBoostingRegressor":
            model = GradientBoostingRegressor(**model_params)
        elif model_type == "Ridge":
            model = Ridge(**model_params)
        elif model_type == "Lasso":
            model = Lasso(**model_params)
        elif model_type == "KNeighborsRegressor":
            model = KNeighborsRegressor(**model_params)
        elif model_type == "DecisionTreeRegressor":
            model = DecisionTreeRegressor(**model_params)
        elif model_type == "XGBRegressor":
            model = XGBRegressor(**model_params)
        elif model_type == "LGBMRegressor":
            model = LGBMRegressor(**model_params)
        elif model_type == "ElasticNet":
            model = ElasticNet(**model_params)
        elif model_type == "SVR(support vector Regressor)":
            model = SVR(**model_params)
        elif model_type == "ExtraTree Regressor":
            model = ExtraTreesRegressor(**model_params)
        elif model_type == "AdaBoost Regressor":
            model = AdaBoostRegressor(**model_params)
        elif model_type == "Neural Network(Regressor)":
            model = MLPRegressor(**model_params)
        else:
            raise ValueError("Invalid model type")

        return model

    def build_pipeline(
            self,
            model_type: str,
            model_params: dict | None = None
    ):

        model = self.get_model(
            model_type=model_type,
            model_params=model_params
        )

        pipeline = Pipeline([
            ("prep", self.preprocessor),
            ("model", model)
        ])

        return pipeline

