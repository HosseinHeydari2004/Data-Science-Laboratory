import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class seaborn_chart:
    @classmethod
    def histogram(
            cls, data: pd.DataFrame | pd.Series = None, x: pd.Series = None,
            y: pd.Series = None, hue: pd.Series = None, kde: bool = False, fill: bool = False,
            stat: str = "count", figsize: tuple[int, int] = (5, 5), dpi: int = 80,
            main_title: str = "", xlabel: str = "", ylabel: str = "", main_title_fontsize: int = 15,
            xlabel_fontsize: int = 12, ylabel_fontsize: int = 12, ax_fontsize: int = 11,
            ax_mode: str = "both"

    ):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sns.histplot(
            data=data, x=x, y=y, hue=hue, stat=stat, ax=ax, kde=kde, fill=fill
        )
        ax.set_title(label=main_title, fontsize=main_title_fontsize)
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        plt.tick_params(axis=ax_mode, labelsize=ax_fontsize)
        return fig

    @classmethod
    def kde_plot(
            cls,
            data: pd.DataFrame = None,
            x: pd.Series = None,
            y: pd.Series = None,
            hue: pd.Series = None,
            palette=None,
            color: str = None,
            fill: bool = False,
            multiple: str = 'layer',
    ):
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.kdeplot(
            data=data, x=x, y=y, hue=hue,
            palette=palette, color=color,
            fill=fill, multiple=multiple, ax=ax
        )
        return fig
