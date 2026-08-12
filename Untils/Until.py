"""
Untils.Until
=============

Small, cross-cutting helper functions used by the Streamlit pages.

This module intentionally has no dependency on ``Core`` so it can be
imported from any page without risking circular imports.

Functions
---------
save_data(data, file_name="processed_data.csv")
    Persist a cleaned/edited DataFrame to disk under ``Data/Proccessed_data``.
save_model(pipeline, name_model="trained_model")
    Serialize a fitted scikit-learn ``Pipeline`` with joblib and expose it
    to the user through a Streamlit download button.
download_matplotlib_figure(fig, file_name="chart.png", ...)
    Render a Streamlit download button that exports a Matplotlib figure
    as an image (PNG/JPEG/SVG/PDF).
"""

import io
import os

import joblib
from matplotlib.figure import Figure
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from streamlit import download_button

# Folder where processed datasets are written. Built with os.path.join so it
# works the same way on Windows, Linux and macOS.
PROCESSED_DATA_DIR = os.path.join("Data", "Proccessed_data")


def save_data(data: DataFrame, file_name: str = "processed_data.csv") -> tuple[bool, str]:
    """
    Save a DataFrame to the local ``Data/Proccessed_data`` folder.

    The output format is inferred from the extension of ``file_name``:
    ``.csv`` and any unrecognized extension are written as CSV,
    ``.xlsx``/``.xls`` are written as Excel.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to persist.
    file_name : str, default="processed_data.csv"
        Name of the output file (extension controls the format).

    Returns
    -------
    tuple[bool, str]
        ``(True, path)`` on success where ``path`` is the file that was
        written, or ``(False, error_message)`` if saving failed for any
        reason (e.g. permission errors, invalid characters in the name).

    Notes
    -----
    Previously this function built the target folder with the Windows-only
    literal ``"Data\\Proccessed_data"``. On Linux/macOS a backslash is not a
    path separator, so ``os.path.dirname`` returned an empty string and the
    file was written into the current working directory instead of a
    dedicated subfolder. This is now built with ``os.path.join`` so it is
    portable across operating systems.
    """
    try:
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        save_path = os.path.join(PROCESSED_DATA_DIR, file_name)

        if file_name.lower().endswith(".csv"):
            data.to_csv(save_path, index=False)
        elif file_name.lower().endswith((".xlsx", ".xls")):
            data.to_excel(save_path, index=False)
        else:
            data.to_csv(save_path, index=False)

        return True, save_path
    except Exception as e:
        return False, str(e)


def save_model(pipeline: Pipeline, name_model: str = "trained_model") -> None:
    """
    Serialize a fitted pipeline with joblib and offer it for download.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The fitted pipeline (preprocessing + model) to serialize.
    name_model : str, default="trained_model"
        Base file name (without extension) used for the downloaded
        ``.joblib`` file.

    Returns
    -------
    None
        This function has a UI side effect only: it renders a Streamlit
        download button in the current app context.
    """
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    download_button(
        label="💾 Download Model",
        data=buffer.getvalue(),
        file_name=f"{name_model}.joblib",
        mime="application/octet-stream"
    )


# Maps a file extension to the MIME type Streamlit needs for its
# download button, and to the keyword arguments Matplotlib's
# `Figure.savefig` needs to produce that format.
_MATPLOTLIB_EXPORT_FORMATS = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "svg": "image/svg+xml",
    "pdf": "application/pdf",
}


def download_matplotlib_figure(
        fig: Figure,
        file_name: str = "chart.png",
        dpi: int = 120,
        key: str | None = None,
        label: str = "⬇️ Download chart",
) -> None:
    """
    Render a Streamlit download button that exports a Matplotlib figure.

    This is the Matplotlib counterpart of :func:`save_model`: instead of
    writing anything to disk on the server, the figure is rendered into an
    in-memory buffer and handed straight to the user's browser via
    ``st.download_button`` — the same pattern already used for models.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to export. Every ``seaborn_chart`` method in
        ``components/charts.py`` already returns one of these.
    file_name : str, default="chart.png"
        Name of the downloaded file. The extension (``.png``, ``.jpg``,
        ``.svg`` or ``.pdf``) selects the export format; anything else
        falls back to PNG.
    dpi : int, default=300
        Resolution used when rasterizing (ignored for the vector ``svg``
        and ``pdf`` formats, but harmless to pass).
    key : str, optional
        Streamlit widget key. Pass a unique value whenever this function
        is called more than once on the same page (e.g. one download
        button per chart type), otherwise Streamlit raises a
        ``StreamlitDuplicateElementId`` error.
    label : str, default="⬇️ Download chart"
        Text shown on the button.

    Returns
    -------
    None
        Side effect only: renders a download button in the current
        Streamlit app context.

    Examples
    --------
    >>> fig = seaborn_chart.histogram(data=df, x="age")
    >>> st.pyplot(fig)
    >>> download_matplotlib_figure(fig, file_name="age_histogram.png", key="hist_dl")
    """
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "png"
    if ext not in _MATPLOTLIB_EXPORT_FORMATS:
        ext = "png"
        file_name = f"{file_name}.png"

    buffer = io.BytesIO()
    # bbox_inches="tight" trims excess whitespace around the plot, which
    # matters more for downloads than for the inline st.pyplot preview.
    fig.savefig(buffer, format=ext, dpi=dpi, bbox_inches="tight")
    buffer.seek(0)

    download_button(
        label=label,
        data=buffer,
        file_name=file_name,
        mime=_MATPLOTLIB_EXPORT_FORMATS[ext],
        key=key,
    )
