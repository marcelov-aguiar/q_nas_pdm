import numpy as np
from typing import List
import matplotlib.pyplot as plt
import util
from ds_library.graphs.plotter import Plotter


class MatPlotter(Plotter):
    def __init__(self):
        self._title = 'Binary Classification'

    def plot(self,
             model_names: List[str],
             metrics: List[float],
             std_devs: List[float] = None,
             title: str = None):
        """Plots a bar chart with colored bars representing
        model metrics and error lines indicating standard deviations.

        Parameters
        ----------
        model_names : List[str]
            List containing model names.
        metrics : List[float]
            List containing metric values for each model.
        std_devs : List[float], optional
            List containing standard deviations of metrics for each model, by default None
        title : str, optional
            Title of the plot, by default None
        """
        colors = plt.cm.tab10(np.arange(len(model_names)))

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(model_names,
                      metrics,
                      yerr=std_devs,
                      capsize=8,
                      color=colors,
                      edgecolor='black',
                      linewidth=1.5)

        ax.set_ylabel('Metric')
        if title is None:
            ax.set_title(self._title)
        else:
            ax.set_title(title)
        ax.set_ylim(0.7, 1.0) 
        ax.set_xticklabels(model_names, rotation=45, ha='right')

        # Adding labels with bar values
        for bar, metric in zip(bars, metrics):
            height = bar.get_height()
            ax.annotate(f'{metric:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(-20, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        if std_devs is not None:
            for bar, std_dev in zip(bars, std_devs):
                ax.plot([bar.get_x() + bar.get_width() / 2, bar.get_x() + bar.get_width() / 2],
                        [bar.get_height() - std_dev, bar.get_height() + std_dev],
                        color='black', linewidth=1)

        # Display the plot
        plt.tight_layout()
        plt.show()
