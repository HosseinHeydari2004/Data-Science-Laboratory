import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline


class MetricPlot:
    @classmethod
    def plot_confusion_matrix(cls, pipeline: Pipeline, X_test, y_test):
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

        # Train Curve
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=train_mean,
                mode="lines+markers",
                name="Training Score"
            )
        )

        # Validation Curve
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=val_mean,
                mode="lines+markers",
                name="Validation Score"
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
