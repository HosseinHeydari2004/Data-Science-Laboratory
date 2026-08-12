import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

from sklearn.model_selection import cross_validate


class Evaluator:
    """
    Fit a pipeline and compute evaluation metrics for one of three tasks.

    ``Evaluator`` is a thin, task-aware wrapper around a scikit-learn
    ``Pipeline``. Depending on ``task_type`` it knows which metrics to
    compute after fitting:

    - ``"classification"``: Accuracy, weighted Precision/Recall/F1 and
      ROC-AUC.
    - ``"regression"``: R2, MAE, MSE and RMSE.
    - ``"clustering"``: Silhouette, Calinski-Harabasz and Davies-Bouldin
      scores plus the number of clusters found.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The (unfitted) pipeline to train and evaluate.
    task_type : str
        One of ``"classification"``, ``"regression"`` or ``"clustering"``
        (case-insensitive).
    """

    def __init__(self, pipeline, task_type: str):
        """Store the pipeline to fit/evaluate and normalize ``task_type`` to lowercase."""

        self.pipeline = pipeline
        self.task_type = task_type.lower()

    def fit(self, X_train, y_train) -> "Evaluator":
        """Fit ``self.pipeline`` on the training data and return ``self``."""

        y_train = pd.Series(y_train).squeeze()

        self.pipeline.fit(
            X_train,
            y_train
        )

        return self

    def predict(self, X):
        """Return ``self.pipeline.predict(X)``."""

        return self.pipeline.predict(X)

    def evaluate(
            self,
            X_train,
            y_train,
            X_test,
            y_test,
            label=None
    ) -> pd.DataFrame:
        """
        Fit on train data, predict on test data and return metrics.

        Parameters
        ----------
        X_train, y_train : array-like
            Training features/target.
        X_test, y_test : array-like
            Held-out features/target used to compute the reported metrics.
        label : optional
            Unused; kept for backward/forward compatibility with callers.

        Returns
        -------
        pandas.DataFrame
            A single-row DataFrame whose columns are the metrics for
            ``self.task_type`` (see class docstring).

        Raises
        ------
        ValueError
            If ``self.task_type`` is not one of ``"classification"``,
            ``"regression"`` or ``"clustering"``.
        """

        y_train = pd.Series(y_train).squeeze()
        y_test = pd.Series(y_test).squeeze()

        self.fit(
            X_train,
            y_train
        )

        y_pred = self.predict(
            X_test
        )
        if self.task_type == "classification":
            y_probs = self.pipeline.predict_proba(X_test)
            # Bug fix: the previous code always used `y_probs[:, 1]`, which
            # only makes sense for binary classification. For a target with
            # more than 2 classes this produced the probability of an
            # arbitrary class and made `roc_auc_score` raise
            # `ValueError: multi_class must be in ('ovo', 'ovr')`. We now
            # branch on the number of classes and use the appropriate
            # scikit-learn call for each case.
            n_classes = y_probs.shape[1]

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
                ),
                "AUC Score": (
                    roc_auc_score(y_test, y_probs[:, 1])
                    if n_classes == 2
                    else roc_auc_score(
                        y_test, y_probs, multi_class="ovr", average="weighted"
                    )
                )
            }
        elif self.task_type == "clustering":
            labels = self.pipeline.fit_predict(X_test)
            results = {
                "Silhouette Score": silhouette_score(
                    X_test,
                    labels
                ),
                "Calinski Harabasz Score": calinski_harabasz_score(
                    X_test,
                    labels
                ),
                "Davies Bouldin Score": davies_bouldin_score(
                    X_test,
                    labels
                ),
                "Number Of Clusters": len(set(labels))
            }

        else:
            raise ValueError(
                f"Unknown task_type: {self.task_type}"
            )

        return pd.DataFrame([results])

    def cross_validation(
            self,
            X,
            y,
            cv: int = 5
    ) -> pd.DataFrame:
        """
        Run k-fold cross-validation and summarize train/test scores.

        Parameters
        ----------
        X, y : array-like
            Full feature matrix and target (not pre-split).
        cv : int, default=5
            Number of folds.

        Returns
        -------
        pandas.DataFrame
            A single-row DataFrame with one column per aggregated metric
            (``"Train <METRIC>"``, ``"CV <METRIC>"``), one column per fold
            (``"<METRIC> Fold <n>"``) and a ``"Total CV"`` column recording
            ``cv``. MAE/MSE are reported as positive values even though
            scikit-learn's scorers are "neg_*" internally.

        Raises
        ------
        ValueError
            If ``self.task_type`` is ``"clustering"`` (cross-validation is
            not defined for unsupervised tasks here) or any other
            unsupported value.
        """
        y = pd.Series(y).squeeze()

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
                "f1": "f1_weighted",

            }

        else:

            raise ValueError(
                f"Unknown task_type: {self.task_type}"
            )

        scores = cross_validate(
            estimator=self.pipeline,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            error_score="raise"
        )

        results = {}

        for metric_name, values in scores.items():

            if metric_name.startswith("train_"):

                metric = metric_name.replace(
                    "train_",
                    ""
                ).upper()

                if metric in ["MAE", "MSE"]:
                    mean_value = abs(values.mean())
                else:
                    mean_value = values.mean()

                results[
                    f"Train {metric}"
                ] = mean_value

            elif metric_name.startswith("test_"):

                metric = metric_name.replace(
                    "test_",
                    ""
                ).upper()

                if metric in ["MAE", "MSE"]:
                    mean_value = abs(values.mean())
                else:
                    mean_value = values.mean()

                results[
                    f"CV {metric}"
                ] = mean_value

                for fold_idx, score in enumerate(
                        values,
                        start=1
                ):

                    if metric in ["MAE", "MSE"]:
                        score = abs(score)

                    results[
                        f"{metric} Fold {fold_idx}"
                    ] = score

        results["Total CV"] = cv

        return pd.DataFrame([results])

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
        """
        Convenience wrapper combining :meth:`evaluate` and
        :meth:`cross_validation` into a single row DataFrame
        (concatenated column-wise).
        """
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

    def clustering_report(self, X) -> pd.DataFrame:
        """
        Fit-predict the clustering pipeline on ``X`` and report metrics.

        Parameters
        ----------
        X : array-like
            Data to cluster.

        Returns
        -------
        pandas.DataFrame
            A single-row DataFrame with Silhouette / Calinski-Harabasz /
            Davies-Bouldin scores, the fitted model's inertia (if it has
            one, e.g. K-Means) and the number of clusters found. The three
            quality scores are ``None`` when fewer than 2 clusters are
            found (e.g. DBSCAN labeling everything as noise).
        """

        labels = self.pipeline.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters > 1:
            silhouette = silhouette_score(X, labels)
            calinski = calinski_harabasz_score(X, labels)
            davies = davies_bouldin_score(X, labels)
        else:
            silhouette = None
            calinski = None
            davies = None

        results = {
            "Silhouette Score": silhouette,
            "Calinski Harabasz": calinski,
            "Davies Bouldin": davies,
            "Inertia": getattr(self.pipeline['model'], "inertia_", None),
            "Clusters": n_clusters,

        }

        return pd.DataFrame([results])
