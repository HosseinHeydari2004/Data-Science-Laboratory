from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
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
        """
        Create a boxplot visualization using seaborn with customizable styling options.

    This method generates a boxplot from the provided data, allowing for grouping
    by hue, adjusting visual properties, and customizing axis labels and titles.

    Parameters
    ----------
    data : pd.DataFrame, optional
        DataFrame containing the variables to plot. If provided, x and y should
        be column names from this DataFrame.
    x : pd.Series, optional
        Data for the x-axis (typically categorical variable). Can be a column
        name if data is provided, or a Series directly.
    y : pd.Series, optional
        Data for the y-axis (typically numeric variable). Can be a column name
        if data is provided, or a Series directly.
    hue : pd.Series, optional
        Categorical variable to group the data by, creating multiple boxplots
        side by side for each category.
    saturation : float, default=0.75
        Proportion of the original saturation to draw colors at. Between 0 and 1.
    fill : bool, default=True
        Whether to fill the boxes with color.
    gap : int | float, default=0
        Gap between boxes when hue nesting is used. Size of the gap as a
        proportion of the box width.
    dpi : int, default=80
        Dots per inch for the figure resolution.
    figsize : tuple[int, int], default=(5, 5)
        Figure size in inches as (width, height).
    main_title : str, default=""
        Main title text for the plot.
    xlabel : str, default=""
        Label for the x-axis.
    ylabel : str, default=""
        Label for the y-axis.
    main_title_fontsize : int, default=15
        Font size for the main title.
    xlabel_fontsize : int, default=12
        Font size for the x-axis label.
    ylabel_fontsize : int, default=12
        Font size for the y-axis label.
    ax_fontsize : int, default=11
        Font size for the axis tick labels.
    ax_mode : str, default="both"
        Which axis ticks to modify. Options: 'x', 'y', 'both'.
        Passed to plt.tick_params(axis=ax_mode).

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the boxplot.

    Examples
    --------
    >>> # Basic usage with a DataFrame
    >>> fig = MyClass.boxplot(
    ...     data=df,
    ...     x='category',
    ...     y='value',
    ...     main_title='Distribution by Category',
    ...     xlabel='Categories',
    ...     ylabel='Values'
    ... )
    >>> plt.show()

    >>> # With hue grouping
    >>> fig = MyClass.boxplot(
    ...     data=df,
    ...     x='category',
    ...     y='value',
    ...     hue='group',
    ...     figsize=(8, 6),
    ...     main_title='Distribution by Category and Group'
    ... )

    >>> # Using Series directly
    >>> fig = MyClass.boxplot(
    ...     x=pd.Series(['A', 'B', 'A', 'B']),
    ...     y=pd.Series([1, 2, 3, 4]),
    ...     fill=False,
    ...     saturation=0.5
    ... )

    Notes
    -----
    This method is a wrapper around seaborn.boxplot that adds customizable
    figure styling and axis formatting. It creates a single figure with
    one subplot.

    See Also
    --------
    seaborn.boxplot : The underlying plotting function.
    matplotlib.pyplot.subplots : Used to create the figure and axes.
        """
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
        """
            Create a count plot showing the frequency of categorical variable observations.

    This method generates a bar chart that displays the count (or proportion) of
    observations in each categorical bin, with options for grouping by hue,
    orientation, and statistical aggregation.

    Parameters
    ----------
    data : pd.DataFrame, optional
        DataFrame containing the variables to plot. If provided, x and y should
        be column names from this DataFrame.
    x : pd.Series, optional
        Data for the x-axis (categorical variable). Can be a column name
        if data is provided, or a Series directly. Mutually exclusive with y.
    y : pd.Series, optional
        Data for the y-axis (categorical variable). Can be a column name
        if data is provided, or a Series directly. Mutually exclusive with x.
    hue : pd.Series, optional
        Categorical variable to group the data by, creating grouped bars
        for each category.
    stat : str, default="count"
        Statistical measure to compute. Options: 'count' (number of observations),
        'percent' (percentage of total), 'probability' (proportion), or
        other seaborn-supported statistics.
    orient : str, default="v"
        Orientation of the plot. Options: 'v' for vertical bars, 'h' for
        horizontal bars.
    saturation : float, default=0.75
        Proportion of the original saturation to draw colors at. Between 0 and 1.
    width : float, default=1
        Width of the bars as a proportion of the bin width.
    log_scale : bool, default=False
        Whether to use a logarithmic scale for the count axis.
    dpi : int, default=80
        Dots per inch for the figure resolution.
    figsize : tuple[int, int], default=(5, 5)
        Figure size in inches as (width, height).
    main_title : str, default=""
        Main title text for the plot.
    xlabel : str, default=""
        Label for the x-axis.
    ylabel : str, default=""
        Label for the y-axis.
    main_title_fontsize : int, default=15
        Font size for the main title.
    xlabel_fontsize : int, default=12
        Font size for the x-axis label.
    ylabel_fontsize : int, default=12
        Font size for the y-axis label.
    ax_fontsize : int, default=11
        Font size for the axis tick labels.
    ax_mode : str, default="both"
        Which axis ticks to modify. Options: 'x', 'y', 'both'.
        Passed to plt.tick_params(axis=ax_mode).
    gap : float, default=0.8
        Gap between groups when hue nesting is used. Size of the gap as a
        proportion of the bar width.
    fill : bool, default=False
        Whether to fill the bars with color.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the count plot.

    Examples
    --------
    >>> # Basic count plot of a categorical variable
    >>> fig = MyClass.countplot(
    ...     data=df,
    ...     x='category',
    ...     main_title='Distribution of Categories',
    ...     xlabel='Category',
    ...     ylabel='Count'
    ... )
    >>> plt.show()

    >>> # Horizontal count plot with hue grouping
    >>> fig = MyClass.countplot(
    ...     data=df,
    ...     y='category',
    ...     hue='group',
    ...     orient='h',
    ...     figsize=(8, 6),
    ...     main_title='Category Distribution by Group',
    ...     gap=0.5
    ... )

    >>> # Showing percentages instead of counts
    >>> fig = MyClass.countplot(
    ...     data=df,
    ...     x='category',
    ...     stat='percent',
    ...     fill=True,
    ...     saturation=0.9,
    ...     main_title='Percentage Distribution'
    ... )

    >>> # Using Series directly
    >>> fig = MyClass.countplot(
    ...     x=pd.Series(['A', 'B', 'A', 'C', 'B', 'B']),
    ...     log_scale=True,
    ...     width=0.7
    ... )

    Notes
    -----
    This method is a wrapper around seaborn.countplot that adds customizable
    figure styling and axis formatting. At least one of x or y must be provided.
    If using hue, the hue variable will create separate bars for each category
    within each bin.

    The stat parameter allows flexible aggregation: 'count' shows raw frequencies,
    'percent' shows percentages of the total, and 'probability' shows proportions
    (values between 0 and 1).

    See Also
    --------
    seaborn.countplot : The underlying plotting function.
    matplotlib.pyplot.subplots : Used to create the figure and axes.
        """
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
        """
            Create a scatter plot to visualize the relationship between two numeric variables.

    This method generates a scatter plot that displays the relationship between
    two continuous variables, with additional options for encoding data through
    color (hue), size, and marker style to reveal patterns, clusters, and
    correlations in the data.

    Parameters
    ----------
    data : pd.DataFrame, optional
        DataFrame containing the variables to plot. If provided, x and y should
        be column names from this DataFrame.
    x : pd.Series, optional
        Data for the x-axis (numeric variable). Can be a column name if data is
        provided, or a Series directly.
    y : pd.Series, optional
        Data for the y-axis (numeric variable). Can be a column name if data is
        provided, or a Series directly.
    hue : pd.Series, optional
        Categorical or numeric variable that determines the color of the points,
        allowing for additional grouping or dimension encoding.
    size : int | pd.Series, optional
        Variable that determines the size of the points. Can be a numeric Series
        for continuous size mapping or a categorical variable for discrete sizes.
    s : int, default=50
        Base point size in points. If size parameter is provided, this value
        serves as the base size for scaling.
    style : pd.Series, optional
        Categorical variable that determines the marker style of the points,
        allowing for visual differentiation of categories.
    figsize : tuple[int, int], default=(5, 5)
        Figure size in inches as (width, height).
    dpi : int, default=80
        Dots per inch for the figure resolution.
    main_title : str, default=""
        Main title text for the plot.
    xlabel : str, default=""
        Label for the x-axis.
    ylabel : str, default=""
        Label for the y-axis.
    main_title_fontsize : int, default=15
        Font size for the main title.
    xlabel_fontsize : int, default=12
        Font size for the x-axis label.
    ylabel_fontsize : int, default=12
        Font size for the y-axis label.
    ax_fontsize : int, default=11
        Font size for the axis tick labels.
    ax_mode : str, default="both"
        Which axis ticks to modify. Options: 'x', 'y', 'both'.
        Passed to plt.tick_params(axis=ax_mode).

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the scatter plot.

    Examples
    --------
    >>> # Basic scatter plot
    >>> fig = MyClass.scatterplot(
    ...     data=df,
    ...     x='height',
    ...     y='weight',
    ...     main_title='Height vs Weight Relationship',
    ...     xlabel='Height (cm)',
    ...     ylabel='Weight (kg)'
    ... )
    >>> plt.show()

    >>> # Scatter plot with color encoding
    >>> fig = MyClass.scatterplot(
    ...     data=df,
    ...     x='height',
    ...     y='weight',
    ...     hue='gender',
    ...     figsize=(8, 6),
    ...     main_title='Height vs Weight by Gender'
    ... )

    >>> # Scatter plot with size and style encoding
    >>> fig = MyClass.scatterplot(
    ...     data=df,
    ...     x='height',
    ...     y='weight',
    ...     hue='age_group',
    ...     size='income',
    ...     style='region',
    ...     s=100,
    ...     main_title='Height vs Weight with Multiple Encodings'
    ... )

    >>> # Using Series directly
    >>> fig = MyClass.scatterplot(
    ...     x=pd.Series([1, 2, 3, 4, 5]),
    ...     y=pd.Series([2, 4, 1, 5, 3]),
    ...     main_title='Simple Scatter Plot'
    ... )

    Notes
    -----
    This method is a wrapper around seaborn.scatterplot that adds customizable
    figure styling and axis formatting. The scatter plot is one of the most
    effective visualizations for exploring relationships between variables and
    identifying patterns such as:

    - Correlation (positive, negative, or none)
    - Clusters or groupings in the data
    - Outliers or anomalies
    - Distribution trends

    The hue, size, and style parameters can be used together to encode up to
    four dimensions of data in a single visualization (x, y, color, size, style),
    making it a powerful exploratory data analysis tool.

    See Also
    --------
    seaborn.scatterplot : The underlying plotting function.
    matplotlib.pyplot.subplots : Used to create the figure and axes.
        """
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
        """
        Create a violin plot to visualize the distribution of numeric data across categories.

    This method generates a violin plot that combines aspects of box plots and
    kernel density plots, showing the full distribution of the data. The width
    of the violin represents the density of data points at different values,
    providing a rich visualization of the distribution's shape, spread, and
    multimodality.

    Parameters
    ----------
    data : pd.DataFrame, optional
        DataFrame containing the variables to plot. If provided, x and y should
        be column names from this DataFrame.
    x : pd.Series, optional
        Data for the x-axis (typically categorical variable). Can be a column name
        if data is provided, or a Series directly. Mutually exclusive with y.
    y : pd.Series, optional
        Data for the y-axis (typically numeric variable). Can be a column name
        if data is provided, or a Series directly. Mutually exclusive with x.
    hue : pd.Series, optional
        Categorical variable to group the data by, creating multiple violins
        side by side for each category.
    orient : str, optional
        Orientation of the plot. Options: 'v' for vertical violins, 'h' for
        horizontal violins. If not specified, it will be inferred from the
        data structure.
    fill : bool, default=False
        Whether to fill the violins with color. If False, only the outline
        of the violin will be displayed.
    gap : float, optional
        Gap between violins when hue nesting is used. Size of the gap as a
        proportion of the violin width. If None, seaborn's default will be used.
    figsize : tuple[int, int], default=(5, 5)
        Figure size in inches as (width, height).
    dpi : int, default=80
        Dots per inch for the figure resolution.
    main_title : str, default=""
        Main title text for the plot.
    xlabel : str, default=""
        Label for the x-axis.
    ylabel : str, default=""
        Label for the y-axis.
    main_title_fontsize : int, default=15
        Font size for the main title.
    xlabel_fontsize : int, default=12
        Font size for the x-axis label.
    ylabel_fontsize : int, default=12
        Font size for the y-axis label.
    ax_fontsize : int, default=11
        Font size for the axis tick labels.
    ax_mode : str, default="both"
        Which axis ticks to modify. Options: 'x', 'y', 'both'.
        Passed to plt.tick_params(axis=ax_mode).

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the violin plot.

    Examples
    --------
    >>> # Basic violin plot
    >>> fig = MyClass.violin_plot(
    ...     data=df,
    ...     x='category',
    ...     y='value',
    ...     main_title='Distribution by Category',
    ...     xlabel='Category',
    ...     ylabel='Value'
    ... )
    >>> plt.show()

    >>> # Violin plot with hue grouping
    >>> fig = MyClass.violin_plot(
    ...     data=df,
    ...     x='category',
    ...     y='value',
    ...     hue='group',
    ...     fill=True,
    ...     figsize=(10, 6),
    ...     main_title='Distribution by Category and Group'
    ... )

    >>> # Horizontal violin plot
    >>> fig = MyClass.violin_plot(
    ...     data=df,
    ...     y='category',
    ...     x='value',
    ...     orient='h',
    ...     main_title='Horizontal Violin Plot'
    ... )

    >>> # With custom gap between groups
    >>> fig = MyClass.violin_plot(
    ...     data=df,
    ...     x='category',
    ...     y='value',
    ...     hue='group',
    ...     gap=0.5,
    ...     fill=True,
    ...     main_title='Violin Plot with Custom Gap'
    ... )

    >>> # Using Series directly
    >>> fig = MyClass.violin_plot(
    ...     x=pd.Series(['A', 'B', 'A', 'B', 'C']),
    ...     y=pd.Series([1, 2, 3, 4, 5]),
    ...     main_title='Simple Violin Plot'
    ... )

    Notes
    -----
    This method is a wrapper around seaborn.violinplot that adds customizable
    figure styling and axis formatting. Violin plots are particularly useful for:

    - Comparing distributions across multiple categories
    - Identifying multimodality (multiple peaks) in the data
    - Visualizing the full shape of the distribution
    - Detecting skewness and outliers

    A violin plot shows:
    - The median (typically a white dot in the center)
    - The interquartile range (thicker central line)
    - The full distribution shape through the width of the violin
    - Potential outliers (extensions of the violin)

    Advantages over box plots:
    - Shows the full distribution, not just summary statistics
    - Better at identifying multimodality and skewness
    - More visually appealing for presentation

    See Also
    --------
    seaborn.violinplot : The underlying plotting function.
    matplotlib.pyplot.subplots : Used to create the figure and axes.
        """
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
        """
            Create a line plot to visualize trends and relationships in sequential data.

    This method generates a line plot that connects data points in order along
    the x-axis, making it ideal for visualizing time series data, trends over
    continuous variables, and relationships where the sequence of observations
    matters. The plot supports multiple grouping dimensions through hue, size,
    and style parameters.

    Parameters
    ----------
    data : pd.DataFrame, optional
        DataFrame containing the variables to plot. If provided, x and y should
        be column names from this DataFrame.
    x : pd.Series, optional
        Data for the x-axis (typically ordered or continuous variable). Can be a
        column name if data is provided, or a Series directly.
    y : pd.Series, optional
        Data for the y-axis (typically numeric variable). Can be a column name
        if data is provided, or a Series directly.
    hue : pd.Series, optional
        Categorical or numeric variable that determines the color of the lines,
        allowing for multiple lines to be plotted on the same axes.
    size : pd.Series, optional
        Variable that determines the width of the lines. Can be a numeric Series
        for continuous mapping or a categorical variable for discrete sizes.
    style : pd.Series, optional
        Categorical variable that determines the line style (solid, dashed,
        dotted, etc.), allowing for additional visual differentiation.
    sort : bool, default=False
        Whether to sort the data by the x variable before plotting. If True,
        points will be connected in sorted order. If False, points are connected
        in the order they appear in the data.
    figsize : tuple[int, int], default=(5, 5)
        Figure size in inches as (width, height).
    dpi : int, default=80
        Dots per inch for the figure resolution.
    main_title : str, default=""
        Main title text for the plot.
    xlabel : str, default=""
        Label for the x-axis.
    ylabel : str, default=""
        Label for the y-axis.
    main_title_fontsize : int, default=15
        Font size for the main title.
    xlabel_fontsize : int, default=12
        Font size for the x-axis label.
    ylabel_fontsize : int, default=12
        Font size for the y-axis label.
    ax_fontsize : int, default=11
        Font size for the axis tick labels.
    ax_mode : str, default="both"
        Which axis ticks to modify. Options: 'x', 'y', 'both'.
        Passed to plt.tick_params(axis=ax_mode).

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the line plot.

    Examples
    --------
    >>> # Basic line plot
    >>> fig = MyClass.lineplot(
    ...     data=df,
    ...     x='date',
    ...     y='temperature',
    ...     main_title='Temperature Over Time',
    ...     xlabel='Date',
    ...     ylabel='Temperature (°C)'
    ... )
    >>> plt.show()

    >>> # Line plot with hue for multiple categories
    >>> fig = MyClass.lineplot(
    ...     data=df,
    ...     x='date',
    ...     y='sales',
    ...     hue='product_category',
    ...     figsize=(10, 6),
    ...     main_title='Sales Trends by Product Category'
    ... )

    >>> # Time series with size and style encoding
    >>> fig = MyClass.lineplot(
    ...     data=df,
    ...     x='month',
    ...     y='revenue',
    ...     hue='region',
    ...     style='quarter',
    ...     size='confidence',
    ...     sort=True,
    ...     main_title='Revenue Trends by Region and Quarter'
    ... )

    >>> # Using Series directly
    >>> fig = MyClass.lineplot(
    ...     x=pd.Series([1, 2, 3, 4, 5]),
    ...     y=pd.Series([2, 4, 3, 5, 7]),
    ...     main_title='Simple Line Plot'
    ... )

    >>> # With confidence intervals (aggregated data)
    >>> fig = MyClass.lineplot(
    ...     data=df,
    ...     x='time',
    ...     y='measurement',
    ...     hue='group',
    ...     main_title='Measurements Over Time with CI'
    ... )

    Notes
    -----
    This method is a wrapper around seaborn.lineplot that adds customizable
    figure styling and axis formatting. Line plots are particularly useful for:

    - Visualizing trends over time (time series analysis)
    - Showing relationships where x is ordered
    - Comparing multiple series on the same axes
    - Displaying aggregated statistics with confidence intervals

    When working with grouped data (using hue), lineplot automatically
    aggregates data points with the same x value for each hue category,
    showing the mean and a confidence interval by default.

    Important considerations:
    - For time series data, ensure your date/time data is properly formatted
      and sorted appropriately
    - Use sort=True when your data isn't already ordered by x
    - The hue parameter is useful for comparing groups, but avoid using too
      many distinct values (typically <= 8-10 for readability)
    - For large datasets, consider aggregating or sampling before plotting

    See Also
    --------
    seaborn.lineplot : The underlying plotting function.
    matplotlib.pyplot.subplots : Used to create the figure and axes.
        """
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


class plotly_charts:
    @classmethod
    def scatterplot(
            cls, data: pd.DataFrame | pd.Series = None, x: pd.Series = None,
            y: pd.Series = None, color: str = None, size: pd.Series = None,
            symbol: pd.Series = None, size_max_symbol: int = 15,
            hover_data: dict[Any] | list[Any] = None, log_x: bool = False,
            log_y: bool = True, figsize: tuple[int, int] = (900, 600),
            main_title: str = "", xlabel: str = "",
            ylabel: str = "", main_title_fontsize: int = 15,
            xlabel_fontsize: int = 12, ylabel_fontsize: int = 12,
            tickfont_x: int = 14, tickfont_y: int = 14, size_marker: int = 12,
            template: str = None
    ) -> plt.Figure:
        fig = px.scatter(
            data_frame=data, x=x, y=y,
            color=color, size=size,
            symbol=symbol, size_max=size_max_symbol,
            hover_data=hover_data, log_x=log_x, log_y=log_y,
            width=figsize[0], height=figsize[1],
        )
        fig.update_traces(marker=dict(size=size_marker))
        fig.update_layout(
            title=main_title,
            title_font=dict(size=main_title_fontsize),
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            xaxis=dict(tickfont=dict(size=tickfont_x), title_font=dict(size=xlabel_fontsize)),
            yaxis=dict(tickfont=dict(size=tickfont_y), title_font=dict(size=ylabel_fontsize)),
            template=template
        )
        return fig

    @classmethod
    def histogram(
            cls, data: pd.DataFrame, x: pd.Series | pd.DataFrame | str | object = None,
            y: pd.DataFrame | pd.Series | str | object = None,
            color: pd.DataFrame | pd.Series | str | object = None,
            log_x: bool | object = False, nbins: int = None,
            log_y: bool | object = True, figsize: tuple[int, int] = (900, 600),
            main_title: str = "", xlabel: str = "",
            ylabel: str = "", main_title_fontsize: int = 15,
            xlabel_fontsize: int = 12, ylabel_fontsize: int = 12,
            tickfont_x: int = 14, tickfont_y: int = 14,
            template: str | object = None, bargap: float = 0.1
    ) -> plt.Figure:
        fig = px.histogram(
            data_frame=data, x=x, y=y, color=color,
            log_x=log_x, log_y=log_y,
            nbins=nbins, width=figsize[0], height=figsize[1],

        )
        fig.update_layout(
            title=main_title,
            title_font=dict(size=main_title_fontsize),
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            xaxis=dict(tickfont=dict(size=tickfont_x), title_font=dict(size=xlabel_fontsize)),
            yaxis=dict(tickfont=dict(size=tickfont_y), title_font=dict(size=ylabel_fontsize)),
            template=template,
            bargap=bargap
        )
        return fig

    @classmethod
    def boxplot(
            cls, data: pd.DataFrame = None, x: pd.Series | str | object = None,
            y: pd.Series | str | object = None, color: pd.Series | str | object = None,
            hover_data: dict[Any] | list[Any] = None, log_x: bool | object = False,
            log_y: bool | object = False, figsize: tuple[int, int] = (900, 600),
            main_title: str = "", xlabel: str = "",
            ylabel: str = "", main_title_fontsize: int = 15,
            xlabel_fontsize: int = 12, ylabel_fontsize: int = 12,
            tickfont_x: int = 14, tickfont_y: int = 14,
            template: str | object = None
    ) -> plt.Figure:
        fig = px.box(
            data_frame=data, x=x, y=y,
            color=color, hover_data=hover_data, log_x=log_x,
            log_y=log_y, width=figsize[0], height=figsize[1],
        )
        fig.update_layout(
            title=main_title,
            title_font=dict(size=main_title_fontsize),
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            xaxis=dict(tickfont=dict(size=tickfont_x), title_font=dict(size=xlabel_fontsize)),
            yaxis=dict(tickfont=dict(size=tickfont_y), title_font=dict(size=ylabel_fontsize)),
            template=template
        )
        return fig

    @classmethod
    def lineplot(
            cls,
            data: pd.DataFrame | pd.Series | object | str = None,
            x: pd.DataFrame | pd.Series | object | str = None,
            y: pd.DataFrame | pd.Series | object | str = None,
            color: pd.Series | str | object = None,
            hover_data: dict[Any] | list[Any] = None, log_x: bool | object = False,
            log_y: bool | object = False, figsize: tuple[int, int] = (900, 600),
            main_title: str = "", xlabel: str = "",
            ylabel: str = "", main_title_fontsize: int = 15,
            xlabel_fontsize: int = 12, ylabel_fontsize: int = 12,
            tickfont_x: int = 14, tickfont_y: int = 14,
            template: str | object = None
    ):
        x_sorted = x
        y_sorted = y
        if isinstance(x, str) and isinstance(x, str):
            x_sorted = data[x].sort_values()
            y_sorted = data[y].sort_values()
        fig = px.line(
            data_frame=data, x=x_sorted, y=y_sorted,
            color=color, hover_data=hover_data, log_x=log_x,
            log_y=log_y, width=figsize[0], height=figsize[1]
        )
        fig.update_layout(
            title=main_title,
            title_font=dict(size=main_title_fontsize),
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            xaxis=dict(tickfont=dict(size=tickfont_x), title_font=dict(size=xlabel_fontsize)),
            yaxis=dict(tickfont=dict(size=tickfont_y), title_font=dict(size=ylabel_fontsize)),
            template=template
        )
        return fig

    @classmethod
    def violin_plot(
            cls,
            data: pd.DataFrame | pd.Series | object | str = None,
            x: pd.DataFrame | pd.Series | object | str = None,
            y: pd.DataFrame | pd.Series | object | str = None,
            color: pd.Series | str | object = None,
            hover_data: dict[Any] | list[Any] = None, log_x: bool | object = False,
            box: bool = False, violin_mode: str = 'group',
            log_y: bool | object = False, figsize: tuple[int, int] = (900, 600),
            main_title: str = "", xlabel: str = "",
            ylabel: str = "", main_title_fontsize: int = 15,
            xlabel_fontsize: int = 12, ylabel_fontsize: int = 12,
            tickfont_x: int = 14, tickfont_y: int = 14,
            template: str | object = None
    ):
        fig = px.violin(
            data_frame=data, x=x, y=y, color=color,
            hover_data=hover_data, log_x=log_x, log_y=log_y,
            box=box, violinmode=violin_mode, width=figsize[0], height=figsize[1],
        )
        fig.update_layout(
            title=main_title,
            title_font=dict(size=main_title_fontsize),
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            xaxis=dict(tickfont=dict(size=tickfont_x), title_font=dict(size=xlabel_fontsize)),
            yaxis=dict(tickfont=dict(size=tickfont_y), title_font=dict(size=ylabel_fontsize)),
            template=template
        )
        return fig

