import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline


class MetricPlot:
    """
    Plotly figure builders used on the "Model Training" page.

    Every method here takes a (fitted or fittable) scikit-learn ``Pipeline``
    plus some data and returns a ready-to-render ``plotly.graph_objs.Figure``
    — nothing is drawn to screen directly, so these can be reused outside
    of Streamlit too (e.g. saved to HTML/PNG).
    """
    @classmethod
    def plot_confusion_matrix(cls, pipeline: Pipeline, X_test, y_test):
        """
        Generate an interactive confusion matrix visualization.

        Parameters
        ----------
        pipeline : Pipeline
            A trained scikit-learn Pipeline object with a `predict` method.
        X_test : array-like
            Feature matrix for testing. Accepts numpy arrays, pandas DataFrames, or similar.
        y_test : array-like
            True labels for the test set. Accepts numpy arrays, pandas Series, or lists.

        Returns
        -------
        plotly.graph_objs.Figure
            A Plotly Figure object containing the interactive confusion matrix.

        Raises
        ------
        AttributeError
            If the pipeline does not have a `predict` method.
        ValueError
            If X_test and y_test have incompatible shapes or types.

        Examples
        --------
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>>
        >>> X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>>
        >>> pipeline = Pipeline([('classifier', RandomForestClassifier())])
        >>> pipeline.fit(X_train, y_train)
        >>>
        >>> fig = MetricPlot.plot_confusion_matrix(pipeline, X_test, y_test)
        >>> fig.show()

        Notes
        -----
        - The plot uses the "viridis" color scale by default.
        - Labels on both axes are automatically sorted based on unique values in `y_test`.
        - The plot is interactive, allowing zoom, pan, and hover details.
        - This method is Prefect-compatible and can be used in Flow tasks for monitoring model performance.
        """
        y_pred = pipeline.predict(X_test)
        fig = px.imshow(
            confusion_matrix(y_test, y_pred),
            text_auto=True,
            color_continuous_scale="viridis",
            labels=dict(x="Predicted", y="Actual"),
            x=sorted(set(y_test)),
            y=sorted(set(y_test))
        )

        fig.update_layout(title_text="Confusion Matrix", title_font_size=24, title_x=0.35)
        fig.update_xaxes(type='category')
        fig.update_yaxes(type='category')

        return fig

    @classmethod
    def plot_roc_curve(cls, pipeline: Pipeline, X_test, y_test):
        """
            Generate an interactive ROC (Receiver Operating Characteristic) curve plot.

            Parameters
            ----------
            pipeline : Pipeline
                A trained scikit-learn Pipeline object with a `predict_proba` method.
            X_test : array-like
                Feature matrix for testing. Accepts numpy arrays, pandas DataFrames, or similar.
            y_test : array-like
                True binary labels for the test set. Accepts numpy arrays, pandas Series, or lists.

            Returns
            -------
            plotly.graph_objs.Figure
                A Plotly Figure object containing the interactive ROC curve.

            Raises
            ------
            AttributeError
                If the pipeline does not have a `predict_proba` method.
            ValueError
                If y_test is not binary or X_test and y_test have incompatible shapes.

            Examples
            --------
            >>> from sklearn.pipeline import Pipeline
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn.datasets import make_classification
            >>> from sklearn.model_selection import train_test_split
            >>>
            >>> X, y = make_classification(n_samples=100, n_features=4, random_state=42)
            >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            >>>
            >>> pipeline = Pipeline([('classifier', RandomForestClassifier())])
            >>> pipeline.fit(X_train, y_train)
            >>>
            >>> fig = MetricPlot.plot_roc_curve(pipeline, X_test, y_test)
            >>> fig.show()

            Notes
            -----
            - The plot displays the ROC curve with the Area Under the Curve (AUC) score.
            - A diagonal dashed line represents random guessing (AUC = 0.5).
            - The optimal threshold is calculated using Youden's J statistic (J = tpr - fpr).
            - The optimal threshold point is marked in red with its value displayed.
            - This method is Prefect-compatible and can be used in Flow tasks for model evaluation.
        """
        y_probs = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        J = tpr - fpr
        ix = np.argmax(J)
        best_threshold = thresholds[ix]
        best_fpr = fpr[ix]
        best_tpr = tpr[ix]
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC curve (AUC = {roc_auc:.2f})',
            line=dict(color='darkorange', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Guess',
            line=dict(color='navy', width=2, dash='dash'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[best_fpr],
            y=[best_tpr],
            mode='markers+text',
            marker=dict(size=12, color='red'),
            text=[f'Threshold = {best_threshold:.3f}'],
            textposition="top center",
            name='Optimal Threshold'
        ))

        fig.update_layout(
            title=dict(text=f'ROC Curve (AUC: {roc_auc:.4f})', font=dict(size=24), x=0.35),
            xaxis=dict(title='False Positive Rate', gridcolor='lightgray'),
            yaxis=dict(title='True Positive Rate', gridcolor='lightgray'),
            width=700, height=600,
            legend=dict(x=0.7, y=0.1, bgcolor='rgba(255,255,255,0.5)')
        )

        return fig

    @classmethod
    def plot_regression_fit(cls, pipeline: Pipeline, X_test, y_test):
        """
        Generate a regression fit plot comparing actual vs predicted values.

        Parameters
        ----------
        pipeline : Pipeline
            A trained scikit-learn Pipeline object with a `predict` method.
        X_test : array-like
            Feature matrix for testing. Accepts numpy arrays, pandas DataFrames, or similar.
        y_test : array-like
            True target values for the test set. Accepts numpy arrays, pandas Series, or lists.

        Returns
        -------
        plotly.graph_objs.Figure
            A Plotly Figure object containing the regression fit scatter plot.

        Raises
        ------
        AttributeError
            If the pipeline does not have a `predict` method.
        ValueError
            If X_test and y_test have incompatible shapes or types.

        Examples
        --------
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.model_selection import train_test_split
        >>>
        >>> X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>>
        >>> pipeline = Pipeline([('regressor', RandomForestRegressor())])
        >>> pipeline.fit(X_train, y_train)
        >>>
        >>> fig = MetricPlot.plot_regression_fit(pipeline, X_test, y_test)
        >>> fig.show()

        Notes
        -----
        - The plot displays actual values on the x-axis and predicted values on the y-axis.
        - A red dashed diagonal line (y = x) represents the ideal perfect prediction.
        - Points closer to the diagonal indicate better model performance.
        - Points above the diagonal indicate over-prediction, below indicate under-prediction.
        - The plot automatically adjusts axis limits based on the data range.
        - This method is Prefect-compatible and can be used in Flow tasks for regression model evaluation.
        """
        y_pred = pipeline.predict(X_test)
        y_test = np.array(y_test).ravel()
        y_pred = np.array(y_pred).ravel()
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())

        fig = go.Figure()

        # Scatter: Actual vs Predicted
        fig.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode="markers",
            marker=dict(
                color="royalblue",
                size=8,
                opacity=0.6
            ),
            name="Predictions"
        ))

        # Ideal Line (y = x)
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="red", dash="dash", width=3),
            name="Ideal Fit"
        ))

        fig.update_layout(
            title=dict(
                text="Regression Fit (Actual vs Predicted)",
                font=dict(size=24),
                x=0.35
            ),
            xaxis=dict(title="Actual Values"),
            yaxis=dict(title="Predicted Values"),
            width=750,
            height=600
        )

        return fig

    @classmethod
    def plot_learning_curve(
            cls,
            pipeline,
            X,
            y,
            cv=5,
            scoring="r2",
            n_jobs=-1
    ):
        """
        Generate a learning curve plot showing training and validation scores.

        Parameters
        ----------
        pipeline : Pipeline
            A scikit-learn Pipeline object to be evaluated.
        X : array-like
            Feature matrix. Accepts numpy arrays, pandas DataFrames, or similar.
        y : array-like
            Target values. Accepts numpy arrays, pandas Series, or lists.
        cv : int, default=5
            Number of cross-validation folds.
        scoring : str, default="r2"
            Scoring metric to evaluate. Common options: "r2", "neg_mean_squared_error",
            "accuracy", "roc_auc", etc. See sklearn metrics for more options.
        n_jobs : int, default=-1
            Number of parallel jobs to run. -1 means using all processors.

        Returns
        -------
        plotly.graph_objs.Figure
            A Plotly Figure object containing the learning curve plot.

        Raises
        ------
        ValueError
            If the pipeline is not fitted or invalid parameters are provided.

        Examples
        --------
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from sklearn.datasets import make_regression
        >>>
        >>> X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        >>> pipeline = Pipeline([('regressor', RandomForestRegressor())])
        >>>
        >>> fig = MetricPlot.plot_learning_curve(
        ...     pipeline, X, y,
        ...     cv=5,
        ...     scoring="r2",
        ...     n_jobs=-1
        ... )
        >>> fig.show()

        Notes
        -----
        - The plot displays training scores and cross-validation scores as functions of training set size.
        - Training scores are shown with markers and lines in the default color.
        - Validation scores are shown with markers and lines in a contrasting color.
        - The y-axis label is automatically set to the uppercase scoring metric name.
        - The plot helps diagnose bias-variance tradeoff:
            - High training score, low validation score: Overfitting
            - Both scores low: Underfitting
            - Both scores high and converging: Good fit
        - This method is Prefect-compatible and can be used in Flow tasks for model diagnostics.
        """
        train_sizes, train_scores, val_scores = learning_curve(
            estimator=pipeline,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=n_jobs
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)

        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        fig = go.Figure()

        # Bug fix / new feature: `train_std` and `val_std` were already
        # being computed but never used, so the learning curve gave no
        # sense of how noisy each score was across folds. We now draw a
        # shaded +/-1 std-dev band around each curve, which is the
        # standard way to visualize this (matplotlib's own learning-curve
        # example does the same with `fill_between`).
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([train_sizes, train_sizes[::-1]]),
                y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
                fill="toself",
                fillcolor="rgba(31,119,180,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([train_sizes, train_sizes[::-1]]),
                y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
                fill="toself",
                fillcolor="rgba(255,127,14,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False
            )
        )

        # Train Curve
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=train_mean,
                mode="lines+markers",
                name="Training Score",
                line=dict(color="rgb(31,119,180)")
            )
        )

        # Validation Curve
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=val_mean,
                mode="lines+markers",
                name="Validation Score",
                line=dict(color="rgb(255,127,14)")
            )
        )

        fig.update_layout(
            title=dict(
                text="Learning Curve",
                x=0.43,
                font=dict(size=24)
            ),
            xaxis_title="Training Examples",
            yaxis_title=scoring.upper(),
            width=800,
            height=600,
            legend=dict(
                x=0.01,
                y=0.99
            )
        )

        return fig

    @classmethod
    def clustering_visualization(cls, X, labels):
        """
        Visualize cluster assignments as a 2D scatter plot.

        This was previously an empty stub (``pass``) — selecting K-Means or
        DBSCAN on the Model Training page produced a metrics table but no
        way to actually *see* the clusters. This implementation projects
        ``X`` down to 2 dimensions with PCA (when it has more than 2
        features) and colors each point by its cluster label, with
        DBSCAN's noise points (label ``-1``) shown in gray.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The (already preprocessed/transformed) data that was clustered.
        labels : array-like of shape (n_samples,)
            Cluster label for each row of ``X``, e.g. the output of
            ``pipeline.fit_predict(X)``.

        Returns
        -------
        plotly.graph_objs.Figure
            A 2D scatter plot, colored by cluster label. If ``X`` has more
            than 2 columns, axes are the first two principal components
            and are labeled with the fraction of variance they explain.
        """
        X = np.asarray(X)
        labels = np.asarray(labels).astype(str)
        labels = np.where(labels == "-1", "Noise", labels)

        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X)
            explained = pca.explained_variance_ratio_
            x_label = f"PC1 ({explained[0]:.1%} variance)"
            y_label = f"PC2 ({explained[1]:.1%} variance)"
        else:
            coords = X[:, :2]
            x_label, y_label = "Feature 1", "Feature 2"

        fig = px.scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            color=labels,
            labels={"x": x_label, "y": y_label, "color": "Cluster"},
            color_discrete_map={"Noise": "lightgray"}
        )
        fig.update_layout(
            title=dict(text="Cluster Visualization", font=dict(size=24), x=0.35),
            width=750,
            height=600
        )
        return fig
