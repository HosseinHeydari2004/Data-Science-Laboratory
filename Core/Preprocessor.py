import numpy as np
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
    """
    Build the scikit-learn preprocessing pipeline for the modeling page.

    Given the list of numeric and categorical column names, this class
    assembles a ``ColumnTransformer`` that scales numeric features and
    encodes categorical ones (with optional imputation for either), plus a
    couple of small helpers for train/test splitting and building a
    human-readable configuration summary table.

    Parameters
    ----------
    num_cols : list[str]
        Names of the numeric feature columns.
    cat_cols : list[str]
        Names of the categorical feature columns.
    """

    def __init__(self, num_cols: list, cat_cols: list):
        """Store the numeric/categorical column names used by every other method."""
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
        """
        Assemble a ``ColumnTransformer`` for the configured columns.

        Parameters
        ----------
        scaler_type : {"Standard Scaler", "MinMax Scaler", "Robust Scaler"}
            Which scikit-learn scaler to apply to ``num_cols``.
        impute : bool, default=False
            Whether to add an imputation step before scaling/encoding.
        num_impute_strategy : str, default="mean"
            Strategy passed to ``SimpleImputer`` for numeric columns
            (used only when ``impute=True``).
        cat_impute_strategy : str, default="most_frequent"
            Strategy passed to ``SimpleImputer`` for categorical columns
            (used only when ``impute=True``).
        encoder_feature_type : {"One Hot Encoder", "Ordinal Encoder"}
            Which encoder to apply to ``cat_cols``.

        Returns
        -------
        sklearn.compose.ColumnTransformer
            Ready to be embedded as the ``"prep"`` step of a
            ``ModelPipelineBuilder`` pipeline.

        Raises
        ------
        ValueError
            If ``scaler_type`` is not recognized, or if there are
            categorical columns but ``encoder_feature_type`` is not
            recognized.
        """

        # =========================
        # Scalers
        # =========================

        scalers = {
            "Standard Scaler": StandardScaler(),
            "MinMax Scaler": MinMaxScaler(feature_range=(0, 1)),
            "Robust Scaler": RobustScaler()
        }

        scaler = scalers.get(scaler_type)

        if scaler is None:
            raise ValueError("Invalid scaler type")

        # =========================
        # Numeric Pipeline
        # =========================

        numeric_steps = []

        if impute and num_impute_strategy:
            numeric_steps.append(
                ("imputer", SimpleImputer(strategy=num_impute_strategy))
            )

        numeric_steps.append(("scaler", scaler))

        numeric_transformer = Pipeline(steps=numeric_steps)

        # =========================
        # Encoders
        # =========================

        encoders = {

            "One Hot Encoder":
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64
                ),

            "Ordinal Encoder":
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.float64
                )
        }

        encoder_feature = encoders.get(encoder_feature_type)

        if encoder_feature is None and self.cat_cols:
            raise ValueError(
                "Encoder type must be specified when categorical columns are present."
            )

        # =========================
        # Categorical Pipeline
        # =========================

        categorical_steps = []

        if impute and cat_impute_strategy:
            categorical_steps.append(
                ("imputer", SimpleImputer(strategy=cat_impute_strategy))
            )

        categorical_steps.append(("encoder", encoder_feature))

        categorical_transformer = Pipeline(steps=categorical_steps)

        # =========================
        # ColumnTransformer
        # =========================

        transformers = []

        if self.num_cols:
            transformers.append(
                ("num", numeric_transformer, self.num_cols)
            )

        if self.cat_cols:
            transformers.append(
                ("cat", categorical_transformer, self.cat_cols)
            )

        prep = ColumnTransformer(transformers=transformers)

        return prep

    @classmethod
    def set_setting_split(
            cls,
            x: np.ndarray,
            y: np.ndarray,
            test_size: float = 0.2,
            stratify: bool = False,
    ) -> tuple[DataFrame, DataFrame, Series, Series]:
        """
        Wrapper around ``sklearn.model_selection.train_test_split``.

        Parameters
        ----------
        x, y : array-like
            Features and target to split.
        test_size : float, default=0.2
            Fraction of the data reserved for the test set.
        stratify : bool, default=False
            If True, splits are stratified by ``y`` (classification only).

        Returns
        -------
        tuple
            ``(x_train, x_test, y_train, y_test)``, with a fixed
            ``random_state=42`` for reproducibility.
        """

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

    def get_config_df(
            self,
            data: DataFrame = None,
            model_name: str = None,
            task_type: str = None,
            target_col: str = None,
            test_size: float = None,
            train_size: float = None,
            stratify: bool = None,
            n_train: int = None,
            n_test: int = None,
            scaler_type=None,
            impute=None,
            num_impute_strategy=None,
            cat_impute_strategy=None,
            encoder_feature_type=None,
            cross_validation: bool = False,
            cv_folds: int | None = None
    ) -> DataFrame:
        """
        Build a two-column (``component``/``value``) summary table.

        This is purely a display helper used by the "configuration
        summary" expander on the modeling page; every argument is optional
        and simply gets echoed back (with light formatting) into the
        output table so the user can review their choices before training.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            The full dataset, used only to compute the class distribution
            when ``task_type == "classification"``.
        model_name, task_type, target_col, scaler_type, encoder_feature_type,
        num_impute_strategy, cat_impute_strategy : str, optional
            Human-readable labels echoed into the summary.
        test_size, train_size : float, optional
            Split ratios, rendered as percentages.
        stratify, impute, cross_validation : bool, optional
            Flags echoed into the summary.
        n_train, n_test, cv_folds : int, optional
            Row counts / fold counts echoed into the summary.

        Returns
        -------
        pandas.DataFrame
            A two-column DataFrame with one row per configuration item.
        """
        if task_type == "classification":
            class_counts = data[target_col].value_counts().sort_index()
            class_distribution_str = " | ".join(
                [f"{cls}: {count}" for cls, count in class_counts.items()]
            )
        else:
            class_counts = None
            class_distribution_str = None
        num_val = ", ".join(self.num_cols) if self.num_cols else np.nan
        cat_val = ", ".join(self.cat_cols) if self.cat_cols else np.nan

        # Note: this used to be named `data`, shadowing the `data`
        # parameter above. It happened to work (Python evaluates the RHS
        # dict literal, including `len(data)` below, before rebinding the
        # name) but was confusing and fragile to future edits, so it is
        # now named `config_rows`.
        config_rows = {
            "component": [
                "target_column",
                "class_distribution_str",
                "numeric_features",
                "categorical_features",
                "n_train_samples",
                "n_test_samples",
                "train_size",
                "test_size",
                "stratify_enabled",
                "task_type",
                "model_name",
                "scaler",
                "imputation_enabled",
                "numeric_imputation_strategy",
                "categorical_imputation_strategy",
                "encoder",
                "cross_validation_enabled",
                "cv_folds"
            ],
            "value": [
                target_col if (task_type == "classification") or (task_type == "regression") else None,
                class_distribution_str if task_type == "classification" else None,
                num_val,
                cat_val,
                n_train,
                n_test,
                f"{train_size * 100:.0f}%" if task_type != "clustering" else len(data) if data is not None else None,
                f"{test_size * 100:.0f}%" if task_type != "clustering" else None,
                stratify if task_type != "clustering" else None,
                task_type,
                model_name,
                scaler_type if self.num_cols else np.nan,
                impute,
                num_impute_strategy if impute and self.num_cols else np.nan,
                cat_impute_strategy if impute and self.cat_cols else np.nan,
                encoder_feature_type if self.cat_cols else np.nan,
                cross_validation,
                cv_folds if cross_validation else np.nan
            ]
        }

        return DataFrame(config_rows)
