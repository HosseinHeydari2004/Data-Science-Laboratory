import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from sklearn.model_selection import cross_validate


class Evaluator:

    def __init__(
            self,
            pipeline,
            task_type: str
    ):

        self.pipeline = pipeline
        self.task_type = task_type.lower()

    def fit(
            self,
            X_train,
            y_train
    ):

        self.pipeline.fit(
            X_train,
            y_train
        )

        return self

    def predict(
            self,
            X
    ):

        return self.pipeline.predict(X)

    def evaluate(
            self,
            X_train,
            y_train,
            X_test,
            y_test
    ) -> pd.DataFrame:

        self.fit(
            X_train,
            y_train
        )

        y_pred = self.predict(
            X_test
        )

        if self.task_type == "regression":

            mse = mean_squared_error(
                y_test,
                y_pred
            )

            results = {

                "R2": r2_score(
                    y_test,
                    y_pred
                ),

                "MAE": mean_absolute_error(
                    y_test,
                    y_pred
                ),

                "MSE": mse,

                "RMSE": mse ** 0.5
            }

        elif self.task_type == "classification":

            results = {

                "Accuracy": accuracy_score(
                    y_test,
                    y_pred
                ),

                "Precision": precision_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0
                ),

                "Recall": recall_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0
                ),

                "F1 Score": f1_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            }

        else:

            raise ValueError(
                f"Unknown task_type: {self.task_type}"
            )

        return pd.DataFrame(
            [results]
        )

    def cross_validation(
            self,
            data,
            feature_cols,
            target_cols,
            cv: int = 5
    ) -> pd.DataFrame:

        if self.task_type == "regression":

            scoring = {
                "r2": "r2",
                "mae": "neg_mean_absolute_error",
                "mse": "neg_mean_squared_error"
            }

        elif self.task_type == "classification":

            scoring = {
                "accuracy": "accuracy",
                "precision": "precision_weighted",
                "recall": "recall_weighted",
                "f1": "f1_weighted"
            }

        else:

            raise ValueError(
                f"Unknown task_type: {self.task_type}"
            )
        X = data[feature_cols]
        y = data[target_cols]
        scores = cross_validate(
            estimator=self.pipeline,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            return_train_score=True
        )

        results = {}

        for metric_name, values in scores.items():

            if metric_name.startswith("train_"):

                metric = metric_name.replace(
                    "train_",
                    ""
                ).upper()

                results[
                    f"Train {metric}"
                ] = abs(
                    values.mean()
                )

            elif metric_name.startswith("test_"):

                metric = metric_name.replace(
                    "test_",
                    ""
                ).upper()

                results[
                    f"CV {metric}"
                ] = abs(
                    values.mean()
                )
        result = pd.DataFrame(
            [results]
        )
        result["total cv"] = cv
        return result

    def full_report(
            self,
            X_train,
            y_train,
            X_test,
            y_test,
            X,
            y,
            cv: int = 5
    ) -> pd.DataFrame:

        eval_df = self.evaluate(
            X_train,
            y_train,
            X_test,
            y_test
        )

        cv_df = self.cross_validation(
            X,
            y,
            cv=cv
        )

        return pd.concat(
            [eval_df, cv_df],
            axis=1
        )
