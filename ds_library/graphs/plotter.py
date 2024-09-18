from abc import ABC, abstractmethod
from typing import List


class Plotter(ABC):
    """Abstract class that contains operations of a Plotter."""

    @abstractmethod
    def plot(self,
             model_names: List[str],
             metrics: List[float],
             std_devs: List[float] = None,
             title: str = None):
        """Method that plots the graph."""
        pass
