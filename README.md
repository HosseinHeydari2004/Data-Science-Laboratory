# ML_Lablator

A Streamlit app for interactive **Exploratory Data Analysis** and
**Machine Learning model training** on any tabular dataset you upload —
no code required.

## Features

- **Upload** a CSV/Excel file and browse it (Home page).
- **EDA page**: missing-value / duplicate / outlier detection and cleanup,
  dtype fixes (including automatic date-column detection), manual data
  selection (by row/column slice, filter, text search, or pandas query),
  descriptive statistics, and a large library of Seaborn/Plotly charts.
- **Model Training page**: build a preprocessing + model pipeline
  (scaling, encoding, imputation), train any of 24 classification /
  regression / clustering algorithms with a UI-generated hyperparameter
  form, evaluate with train/test split and optional k-fold cross
  validation, and inspect confusion matrix / ROC curve / learning curve /
  actual-vs-predicted / cluster scatter plots.

## Project layout

```
main_page.py                 # st.navigation entry point
app/pages/
    home_page.py              # file upload / landing page
    page_2.py                 # Exploratory Data Analysis
    page_3.py                 # Model Training
Core/
    eda.py                     # EDA, handle_MissingValue, handle_outliers, data_manipulation
    Preprocessor.py            # DataPreprocessor: scaling/encoding/imputation + train/test split
    model_trainer.py           # ModelPipelineBuilder + ModelParameterFactory (hyperparameter UI)
    evaluator.py                # Evaluator: fit/predict/evaluate/cross-validate any task type
components/
    charts.py                  # seaborn_chart, plotly_charts — reusable plot builders
    metrics_plots.py           # MetricPlot — confusion matrix / ROC / learning curve / clusters
Untils/
    Until.py                    # save_data / save_model helpers
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run main_page.py
```

## Changelog — bug-fix & refactor pass (2026-08-09)

This pass focused on: (1) fixing real, verified bugs, (2) documenting every
public class/function that lacked a docstring, and (3) turning two
already-half-built features into working ones, without changing the
app's overall structure or UI flow.

### Bugs fixed

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | `Untils/Until.py` | `"Data\Proccessed_data"` is a Windows-only literal path (backslash isn't a separator on Linux/macOS), so `save_data` silently wrote to the wrong folder off Windows. | Rebuilt with `os.path.join`; also removed unused imports. |
| 2 | `Core/eda.py` — `change_dtype_datetime64` | `check_date_object` returns either `False` or `(True, col_name)`. The caller did `ch, data_col = check_date_object(...)` unconditionally, which raises `TypeError: cannot unpack non-iterable bool object` whenever no date column is found. | Now checks `isinstance(result, tuple)` before unpacking, and always returns the DataFrame (previously fell through to `None` in that branch). |
| 3 | `Core/eda.py` — `check_dtype_column` | Fell through to an implicit `return None` for any dtype that isn't `object` or `number` (e.g. `bool`, `datetime64`). Callers treat the result as a 0/1 flag (`== 0` → text filter, else → numeric filter), so `None` silently took the *numeric* branch and broke on boolean/date columns. | Non-object dtypes now explicitly return `1`. |
| 4 | `Core/model_trainer.py` — `ModelParameterFactory.PARAMS_FACTORY` | The **"Extra Tree"** classifier had no entry in this mapping (only its regressor twin did), even though its hyperparameter function `extra_tree` was already implemented. Selecting it in the UI silently trained with library defaults and no visible controls. | Added the missing `"Extra Tree": extra_tree` entry. |
| 5 | `Core/model_trainer.py` — `logistic_regression`, `elasticnet` | `l1_ratio` sliders allowed `0.0–20.0`; scikit-learn requires `[0.0, 1.0]` and raises `ValueError` above that. | Sliders now capped at `1.0`. |
| 6 | `Core/model_trainer.py` — `Neural_Network_regressor` | An `alpha` slider was given `key=""` (empty string), risking a duplicate/empty widget key collision. | Given a unique key. |
| 7 | `Core/evaluator.py` — `Evaluator.evaluate` | AUC was always computed as `roc_auc_score(y_test, predict_proba[:, 1])`, which only works for binary classification; any 3+-class target raised `ValueError`. | Branches on the number of classes: binary keeps the original behavior, multiclass uses `roc_auc_score(..., multi_class="ovr", average="weighted")`. |
| 8 | `app/pages/page_3.py` | `y.ravel()` was called on a `pandas.Series` in three places. `Series.ravel()` was removed in modern pandas (crashes with `AttributeError` on pandas ≥ 2.x, confirmed against pandas 3.0). | Replaced with `y.to_numpy().ravel()`. |
| 9 | `app/pages/page_2.py` | A caught exception (`except Exception as E`) discarded `E` and showed a generic message, unlike every other error handler on the page, making failures hard to diagnose. | The message now includes `{E}`. |
| 10 | `app/pages/page_2.py` | 4 f-strings had no placeholders (harmless but misleading — flagged by `ruff` as F541). | Cleaned up. |
| 11 | `Core/Preprocessor.py` — `get_config_df` | The local dict was named `data`, shadowing the `data: DataFrame` parameter. It happened to work (Python evaluates the right-hand side before rebinding the name) but was confusing and fragile to future edits. | Renamed to `config_rows`. |

All fixes were verified with a small functional test script (constructing
real pipelines, evaluating a 3-class classifier, converting a date column,
etc.) — see the bottom of this section for the kind of checks that were run.

### Previously unfinished features, now completed

- **`MetricPlot.clustering_visualization`** was an empty `pass` stub — the
  Model Training page showed a metrics *table* for K-Means/DBSCAN but no
  plot. It's now implemented: a 2D scatter of the clustered points
  (PCA-projected when there are more than 2 features), colored by cluster,
  with DBSCAN's noise points (label `-1`) shown separately. Wired into
  `page_3.py` under a new "Show Cluster Visualization" expander.
- **Learning curve confidence bands**: `plot_learning_curve` was already
  computing `train_std`/`val_std` (flagged as unused by `ruff`) but never
  plotting them. It now draws a shaded ±1 standard-deviation band around
  each curve, which is the standard way to show cross-validation noise on
  a learning curve.

### Known limitation intentionally left as-is

`Core/eda.py`'s `handle_MissingValue.fill_SimpleImputer` (a fully
implemented, documented method for imputing missing values with
scikit-learn's `SimpleImputer`) is not called anywhere in `page_2.py` —
the EDA page currently only offers row/column *deletion* for missing
values, not imputation. This wasn't touched in this pass since it's a
feature gap rather than a bug, but see the suggestions below.

## Suggested next features

1. **Wire up `fill_SimpleImputer` in the EDA page** — it already exists
   and is documented; add a third option next to "delete by row/column"
   for "impute" with a strategy selector.
2. **Model persistence** — `Untils/Until.save_model` is fully implemented
   but the "Save model" section in `page_3.py` is commented out. Re-enable
   it so trained pipelines can be downloaded as `.joblib` and reloaded
   later for inference on new data.
3. **Batch/inference page** — a new page that loads a saved `.joblib`
   pipeline and lets the user upload new data (or fill a form) to get
   predictions, closing the loop from "train" to "use."
4. **Feature importance / SHAP view** — for tree-based models
   (RandomForest, XGBoost, LightGBM, ExtraTrees), show built-in feature
   importances; optionally add SHAP summary plots for deeper explainability.
5. **Multiclass ROC** — `plot_roc_curve` currently only handles binary
   targets (`predict_proba[:, 1]`); extend it to one-vs-rest curves per
   class for multiclass problems, mirroring the AUC fix already made in
   `Evaluator`.
6. **Session-based experiment tracking** — keep a running table (in
   `st.session_state`) of every model trained in the session with its
   metrics, so users can compare runs side by side instead of only seeing
   the latest one.
7. **Config export/import** — let a user export the full
   pipeline configuration (`get_config_df` output) as JSON/YAML and
   re-import it to reproduce a run without re-clicking every widget.
8. **Data profiling report export** — a "Download EDA report" button that
   renders the key EDA findings (missing values, duplicates, outliers,
   describe table) to a PDF/HTML, useful for sharing with non-technical
   stakeholders.
9. **Class imbalance handling** — given your recurring interest in
   `RandomUnderSampler`/`SMOTE`, this would fit naturally as an optional
   preprocessing step alongside scaling/encoding on the Model Training page.
