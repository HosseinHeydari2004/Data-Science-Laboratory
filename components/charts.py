import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class seaborn_chart:
    """
    This class contains functions for plotting Seaborn charts.
    """

    @classmethod
    def histogram(
            cls, data: pd.DataFrame | pd.Series = None, x: pd.Series = None,
            y: pd.Series = None, hue: pd.Series = None, kde: bool = False, fill: bool = False,
            stat: str = "count", figsize: tuple[int, int] = (5, 5), dpi: int = 80,
            main_title: str = "", xlabel: str = "", ylabel: str = "", main_title_fontsize: int = 15,
            xlabel_fontsize: int = 12, ylabel_fontsize: int = 12, ax_fontsize: int = 11,
            ax_mode: str = "both"

    ) -> plt.Figure:
        """
        Plot a histogram using Seaborn and Matplotlib with multiple customization options.

        This method creates a histogram for the provided data (either a DataFrame or Series),
        allowing fine control over the appearance, statistical interpretation, and labeling.
        It supports kernel density estimate (KDE) overlay and can group data by a hue variable.

        Parameters
        ----------
        data : pd.DataFrame or pd.Series, optional
            Input data structure containing the variables to be plotted.

        x : pd.Series, optional
            Data to be plotted on the x-axis.

        y : pd.Series, optional
            Data to be plotted on the y-axis (for 2D histograms).

        hue : pd.Series, optional
            Variable that defines subsets within data, used for color grouping.

        kde : bool, default=False
            If True, a Kernel Density Estimate curve is plotted along with the histogram.

        fill : bool, default=False
            If True, hist bars are filled with color.

        stat : str, default="count"
            Defines the statistic to compute: options include "count", "frequency",
            "density", or "probability" (as supported by seaborn.histplot).

        figsize : tuple[int, int], default=(5, 5)
            Figure size in inches.

        dpi : int, default=80
            Resolution of the figure.

        main_title : str, optional
            The figure’s main title.

        xlabel : str, optional
            Label for the X-axis.

        ylabel : str, optional
            Label for the Y-axis.

        main_title_fontsize : int, default=15
            Font size for the main title.

        xlabel_fontsize : int, default=12
            Font size for the X-axis label.

        ylabel_fontsize : int, default=12
            Font size for the Y-axis label.

        ax_fontsize : int, default=11
            Font size for axis tick labels.

        ax_mode : str, default="both"
            Axis selection for tick parameter adjustment — one of {"x", "y", "both"}.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plotted histogram.

        Example
        -------
        >>> df = pd.DataFrame({'values': [1, 2, 2, 3, 4, 4, 5]})
        >>> fig = seaborn_chart.histogram(data=df, x=df['values'], kde=True, fill=True,
        ...     main_title="Value Distribution", xlabel="Values", ylabel="Frequency")
        >>> fig.show()

        Notes
        -----
        - The method uses `sns.histplot()` internally.
        - Tick labels and font sizes can be adjusted via parameters.
        - Designed for quick visualization during data exploration.
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sns.histplot(
            data=data, x=x, y=y, hue=hue, stat=stat, ax=ax, kde=kde, fill=fill
        )
        ax.set_title(label=main_title, fontsize=main_title_fontsize)
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        plt.tick_params(axis=ax_mode, labelsize=ax_fontsize)
        return fig

    histogram.__doc__ = """
        Plot a histogram using Seaborn and Matplotlib with multiple customization options.

        This method creates a histogram for the provided data (either a DataFrame or Series),
        allowing fine control over the appearance, statistical interpretation, and labeling.
        It supports kernel density estimate (KDE) overlay and can group data by a hue variable.

        Parameters
        ----------
        data : pd.DataFrame or pd.Series, optional
            Input data structure containing the variables to be plotted.

        x : pd.Series, optional
            Data to be plotted on the x-axis.

        y : pd.Series, optional
            Data to be plotted on the y-axis (for 2D histograms).

        hue : pd.Series, optional
            Variable that defines subsets within data, used for color grouping.

        kde : bool, default=False
            If True, a Kernel Density Estimate curve is plotted along with the histogram.

        fill : bool, default=False
            If True, hist bars are filled with color.

        stat : str, default="count"
            Defines the statistic to compute: options include "count", "frequency",
            "density", or "probability" (as supported by seaborn.histplot).

        figsize : tuple[int, int], default=(5, 5)
            Figure size in inches.

        dpi : int, default=80
            Resolution of the figure.

        main_title : str, optional
            The figure’s main title.

        xlabel : str, optional
            Label for the X-axis.

        ylabel : str, optional
            Label for the Y-axis.

        main_title_fontsize : int, default=15
            Font size for the main title.

        xlabel_fontsize : int, default=12
            Font size for the X-axis label.

        ylabel_fontsize : int, default=12
            Font size for the Y-axis label.

        ax_fontsize : int, default=11
            Font size for axis tick labels.

        ax_mode : str, default="both"
            Axis selection for tick parameter adjustment — one of {"x", "y", "both"}.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plotted histogram.

        Example
        -------
        >>> df = pd.DataFrame({'values': [1, 2, 2, 3, 4, 4, 5]})
        >>> fig = seaborn_chart.histogram(data=df, x=df['values'], kde=True, fill=True,
        ...     main_title="Value Distribution", xlabel="Values", ylabel="Frequency")
        >>> fig.show()

        Notes
        -----
        - The method uses `sns.histplot()` internally.
        - Tick labels and font sizes can be adjusted via parameters.
        - Designed for quick visualization during data exploration.
        """

    @classmethod
    def kde_plot(
            cls,
            data: pd.DataFrame = None,
            x: pd.Series = None,
            y: pd.Series = None,
            hue: pd.Series = None,
            fill: bool = False,
            multiple: str = 'layer',
            figsize: tuple[int, int] = (5, 5), dpi: int = 80,
            main_title: str = "", xlabel: str = "",
            ylabel: str = "", main_title_fontsize: int = 15,
            xlabel_fontsize: int = 12, ylabel_fontsize: int = 12,
            ax_fontsize: int = 11,
            ax_mode: str = "both"
    ) -> plt.Figure:
        """
            Generates a Kernel Density Estimate (KDE) plot using Seaborn.

            Parameters
            ----------
            data : pd.DataFrame, optional
                The DataFrame containing the data to plot.  Defaults to None.
            x : pd.Series, optional
                The Series representing the x-axis data. Defaults to None.
            y : pd.Series, optional
                The Series representing the y-axis data. Defaults to None.
            hue : pd.Series, optional
                The Series representing the categorical variable to use for coloring
                the KDE plots. Defaults to None.
            fill : bool, optional
                Whether to fill the KDE plots with color. Defaults to False.
            multiple : str, optional
                Specifies how multiple KDE plots should be drawn ('layer', 'stack', or
                'dodge'). Defaults to 'layer'.
            figsize : tuple[int, int], optional
                The figure size (width, height) in inches. Defaults to (5, 5).
            dpi : int, optional
                The resolution of the plot in dots per inch. Defaults to 80.
            main_title : str, optional
                The title of the plot. Defaults to "".
            xlabel : str, optional
                The label for the x-axis. Defaults to "".
            ylabel : str, optional
                The label for the y-axis. Defaults to "".
            main_title_fontsize : int, optional
                The font size for the main title. Defaults to 15.
            xlabel_fontsize : int, optional
                The font size for the x-axis label. Defaults to 12.
            ylabel_fontsize : int, optional
                The font size for the y-axis label. Defaults to 12.
            ax_fontsize : int, optional
                The font size for the axis tick labels. Defaults to 11.
            ax_mode : str, optional
                Specifies which axis to apply the fontsize to ('x', 'y', or 'both').
                Defaults to "both".

            Returns
            -------
            plt.Figure
                The matplotlib Figure object containing the generated KDE plot.
            """

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sns.kdeplot(
            data=data, x=x, y=y, hue=hue,
            fill=fill, multiple=multiple, ax=ax
        )
        ax.set_title(label=main_title, fontsize=main_title_fontsize)
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        plt.tick_params(axis=ax_mode, labelsize=ax_fontsize)
        return fig

    @classmethod
    def boxplot(cls,
                data: pd.DataFrame = None,
                x: pd.Series = None,
                y: pd.Series = None,
                hue: pd.Series = None,
                saturation: float = 0.75,
                fill: bool = True,
                gap: int | float = 0,
                dpi: int = 80,
                figsize: tuple[int, int] = (5, 5),
                main_title: str = "", xlabel: str = "",
                ylabel: str = "", main_title_fontsize: int = 15,
                xlabel_fontsize: int = 12, ylabel_fontsize: int = 12,
                ax_fontsize: int = 11,
                ax_mode: str = "both"
                ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sns.boxplot(
            data=data, x=x, y=y,
            hue=hue,
            saturation=saturation, fill=fill,
            gap=gap, ax=ax
        )
        ax.set_title(label=main_title, fontsize=main_title_fontsize)
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        plt.tick_params(axis=ax_mode, labelsize=ax_fontsize)
        return fig

    @classmethod
    def countplot(
            cls, data: pd.DataFrame = None, x: pd.Series = None, y: pd.Series = None,
            hue: pd.Series = None, stat: str = "count", orient: str = "v", saturation: float = 0.75,
            width: float = 1, log_scale: bool = False, dpi: int = 80,
            figsize: tuple[int, int] = (5, 5),
            main_title: str = "", xlabel: str = "",
            ylabel: str = "", main_title_fontsize: int = 15,
            xlabel_fontsize: int = 12, ylabel_fontsize: int = 12,
            ax_fontsize: int = 11,
            ax_mode: str = "both", gap: float = 0.8, fill: bool = False

    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sns.countplot(
            data=data, x=x, y=y, hue=hue, stat=stat, orient=orient,
            saturation=saturation, width=width, log_scale=log_scale, ax=ax, gap=gap,
            fill=fill
        )
        ax.set_title(label=main_title, fontsize=main_title_fontsize)
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        plt.tick_params(axis=ax_mode, labelsize=ax_fontsize)
        return fig

    @classmethod
    def scatterplot(
            cls, data: pd.DataFrame = None, x: pd.Series = None, y: pd.Series = None,
            hue: pd.Series = None, size: int | pd.Series = None, s: int = 50, style: pd.Series = None,
            figsize: tuple[int, int] = (5, 5), dpi: int = 80,
            main_title: str = "", xlabel: str = "",
            ylabel: str = "", main_title_fontsize: int = 15,
            xlabel_fontsize: int = 12, ylabel_fontsize: int = 12,
            ax_fontsize: int = 11,
            ax_mode: str = "both",
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sns.scatterplot(
            data=data, x=x, y=y, size=size, s=s, hue=hue, style=style, ax=ax
        )
        ax.set_title(label=main_title, fontsize=main_title_fontsize)
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        plt.tick_params(axis=ax_mode, labelsize=ax_fontsize)
        return fig

    @classmethod
    def violin_plot(
            cls, data: pd.DataFrame = None, x: pd.Series = None, y: pd.Series = None,
            hue: pd.Series = None, orient: str = None, fill: bool = False,
            gap: float = None, figsize: tuple[int, int] = (5, 5), dpi: int = 80,
            main_title: str = "", xlabel: str = "",
            ylabel: str = "", main_title_fontsize: int = 15,
            xlabel_fontsize: int = 12, ylabel_fontsize: int = 12,
            ax_fontsize: int = 11,
            ax_mode: str = "both",
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sns.violinplot(
            data=data, x=x, y=y, hue=hue, orient=orient, fill=fill, gap=gap, ax=ax
        )
        ax.set_title(label=main_title, fontsize=main_title_fontsize)
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        plt.tick_params(axis=ax_mode, labelsize=ax_fontsize)
        return fig

    @classmethod
    def lineplot(
            cls, data: pd.DataFrame = None, x: pd.Series = None, y: pd.Series = None,
            hue: pd.Series = None, size: pd.Series = None, style: pd.Series = None,
            sort: bool = False, figsize: tuple[int, int] = (5, 5), dpi: int = 80,
            main_title: str = "", xlabel: str = "",
            ylabel: str = "", main_title_fontsize: int = 15,
            xlabel_fontsize: int = 12, ylabel_fontsize: int = 12,
            ax_fontsize: int = 11,
            ax_mode: str = "both",
    ):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sns.lineplot(
            data=data, x=x, y=y, hue=hue, size=size, style=style, sort=sort
        )
        ax.set_title(label=main_title, fontsize=main_title_fontsize)
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        plt.tick_params(axis=ax_mode, labelsize=ax_fontsize)
        return fig

    @classmethod
    def heatmap(
            cls, data: pd.DataFrame, annot: bool = False, fmt: str = ".2f",
            figsize: tuple[int, int] = (5, 5), dpi: int = 80,
            main_title: str = "", xlabel: str = "",
            ylabel: str = "", main_title_fontsize: int = 15,
            xlabel_fontsize: int = 12, ylabel_fontsize: int = 12,
            ax_fontsize: int = 11,
            ax_mode: str = "both", annot_kws: dict[str, int] = {"size": 12}
    ):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sns.heatmap(data=data, annot=annot, fmt=fmt, annot_kws=annot_kws)
        ax.set_title(label=main_title, fontsize=main_title_fontsize)
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        plt.tick_params(axis=ax_mode, labelsize=ax_fontsize)
        return fig
