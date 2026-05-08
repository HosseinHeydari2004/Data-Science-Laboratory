import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class seaborn_chart:
    @classmethod
    def histogram(
            cls, data: pd.DataFrame | pd.Series = None, x: pd.Series = None,
            y: pd.Series = None, hue: pd.Series = None, kde: bool = False, fill: bool = False,
            stat: str = "count"

    ):
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(
            data=data, x=x, y=y, hue=hue, stat=stat, ax=ax, kde=kde, fill=fill
        )
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
