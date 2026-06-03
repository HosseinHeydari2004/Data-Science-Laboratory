from category_encoders import TargetEncoder
from pandas import Series, DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder
)


class DataPreprocessor:

    def __init__(
            self,
            num_cols: list,
            cat_cols: list
    ):

        self.num_cols = num_cols
        self.cat_cols = cat_cols

    def get_transformer(

            self,

            scaler_type: str = "Standard Scaler",

            impute: bool = False,

            num_impute_strategy: str = "mean",

            cat_impute_strategy: str = "most_frequent",

            encoder_feature_type: str = "One Hot Encoder"

    ) -> ColumnTransformer:

        # =========================
        # Scalers
        # =========================

        scalers = {

            "Standard Scaler": StandardScaler(),

            "MinMax Scaler": MinMaxScaler(),

            "Robust Scaler": RobustScaler()
        }

        scaler = scalers.get(scaler_type)

        if scaler is None:
            raise ValueError("Invalid scaler type")

        # =========================
        # Numeric Pipeline
        # =========================

        numeric_steps = []

        if impute:
            if num_impute_strategy is not None:
                numeric_steps.append(
                    (
                        "imputer",
                        SimpleImputer(
                            strategy=num_impute_strategy
                        )
                    )
                )
            else:
                numeric_steps = []

        numeric_steps.append(
            ("scaler", scaler)
        )

        numeric_transformer = Pipeline(
            steps=numeric_steps
        )

        # =========================
        # Encoders
        # =========================

        encoders = {

            "One Hot Encoder":

                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),

            "Ordinal Encoder":

                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ),

            "Target Encoder":

                TargetEncoder()
        }

        encoder_feature = encoders.get(
            encoder_feature_type
        )

        if encoder_feature is None:
            raise ValueError("Invalid encoder type")

        # =========================
        # Categorical Pipeline
        # =========================

        categorical_steps = []

        if impute:
            if cat_impute_strategy is not None:
                categorical_steps.append(
                    (
                        "imputer",
                        SimpleImputer(
                            strategy=cat_impute_strategy
                        )
                    )
                )
            else:
                categorical_steps = []

        categorical_steps.append(
            ("encoder", encoder_feature)
        )

        categorical_transformer = Pipeline(
            steps=categorical_steps
        )

        # =========================
        # Final ColumnTransformer
        # =========================

        prep = ColumnTransformer(

            transformers=[

                (
                    "num",
                    numeric_transformer,
                    self.num_cols
                ),

                (
                    "cat",
                    categorical_transformer,
                    self.cat_cols
                )
            ]
        )

        return prep

    @classmethod
    def set_setting_split(
            cls,
            data: DataFrame,
            feature_cols: list[str],
            target_col: str,
            test_size: float = 0.2,
            stratify: bool = False,
    ) -> tuple[DataFrame, DataFrame, Series, Series]:

        x = data[feature_cols]
        y = data[target_col]
        if stratify:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=test_size,
                random_state=42,
                stratify=y
            )
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=test_size,
                random_state=42
            )

        return x_train, x_test, y_train, y_test
