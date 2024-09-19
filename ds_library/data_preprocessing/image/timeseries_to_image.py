from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

class TimeSeriesToImage(BaseEstimator, TransformerMixin):
    def __init__(self,
                 features_names: List[str],
                 target: str,
                 window_size: int) -> None:
        self._features_names = features_names
        self._target = target
        self._window_size = window_size
    
    def fit(self, X:pd.DataFrame=None, y=None):
        return self

    def transform(self,
                  df_data: pd.DataFrame) -> Tuple[np.array, np.array]:
        images = []
        targets = []

        if self._target in self._features_names:
            self._features_names.remove(self._target)

        for start in range(len(df_data) - self._window_size + 1):
            end = start + self._window_size
            window_df = df_data.iloc[start:end][self._features_names]

            image = window_df.T.values
            images.append(image)

            # Ajustar o índice para evitar o erro de indexação
            if end - 1 < len(df_data):
                target = df_data.iloc[end - 1][self._target]
                targets.append(target)
            else:
                # Adicione um tratamento de erro, se necessário
                raise IndexError(f'End index {end-1} is out of bounds for the DataFrame of length {len(df_data)}')

        return np.array(images), np.array(targets)

"""
time    sensor_1    sensor_2    sensor_3    RUL
1       0.1         0.2         0.11        40
2       0.3         0.4         0.13        39
3       0.5         0.6         0.15        38
4       0.7         0.8         0.17        37
5       0.9         0.10        0.19        36
6       0.11        0.12        0.21        35
7       0.13        0.14        0.23        34



Imagem
n_features=4
Window=4
features_name = time, sensor_1, sensor_2, sensor_3
TARGET=RUL

imagem_1:
            windows1    window2     windows3    window4
time        1           2           3           4
sensor_1    0.1         0.3         0.5         0.7         
sensor_2    0.2         0.4         0.6         0.8
sensor_3    0.11        0.13        0.15        0.17

target_value=37

imagem_2:
            windows1    window2     windows3    window4
time        2           3           4           5
sensor_1    0.3         0.5         0.7         0.9
sensor_2    0.4         0.6         0.8         0.10
sensor_3    0.13        0.15        0.17        0.19

target_value=36

imagem_3:
            windows1    window2     windows3    window4
time        3           4           5           6
sensor_1    0.5         0.7         0.9         0.11
sensor_2    0.6         0.8         0.10        0.12
sensor_3    0.15        0.17        0.19        0.21

target_value=35

.
.
.


"""

if __name__ == "__main__":
    data = {
    'time': [1, 2, 3, 4, 5, 6, 7],
    'sensor_1': [0.1, 0.3, 0.5, 0.7, 0.9, 0.11, 0.13],
    'sensor_2': [0.2, 0.4, 0.6, 0.8, 0.10, 0.12, 0.14],
    'sensor_3': [0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23],
    'RUL': [40, 39, 38, 37, 36, 35, 34]
    }
    df = pd.DataFrame(data)

    window_size = 4
    features_names = ['time', 'sensor_1', 'sensor_2', 'sensor_3']
    target_column = 'RUL'
    images, targets = TimeSeriesToImage(features_names=features_names,
                                    target=target_column,
                                    window_size=window_size).fit_transform(df)
    print("Finished")