import pytest
import pandas as pd
import numpy as np
from ds_library.data_preprocessing.image.timeseries_to_image import TimeSeriesToImage

@pytest.fixture
def sample_data():
    data = {
        'time': [1, 2, 3, 4, 5, 6, 7],
        'sensor_1': [0.1, 0.3, 0.5, 0.7, 0.9, 0.11, 0.13],
        'sensor_2': [0.2, 0.4, 0.6, 0.8, 0.10, 0.12, 0.14],
        'sensor_3': [0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23],
        'RUL': [40, 39, 38, 37, 36, 35, 34]
    }
    df = pd.DataFrame(data)
    return df

def test_series_to_image(sample_data):
    window_size = 4
    features_names = ['time', 'sensor_1', 'sensor_2', 'sensor_3']
    target_column = 'RUL'

    transformer = TimeSeriesToImage(features_names=features_names,
                                    target=target_column,
                                    window_size=window_size)
    images, targets = transformer.fit_transform(sample_data)

    # Verifique se o número correto de imagens e alvos foi gerado
    expected_images_shape = (len(sample_data) - window_size + 1, len(features_names), window_size)
    expected_targets_shape = (len(sample_data) - window_size + 1,)
    
    assert images.shape == expected_images_shape, f"Expected images shape {expected_images_shape}, but got {images.shape}"
    assert targets.shape == expected_targets_shape, f"Expected targets shape {expected_targets_shape}, but got {targets.shape}"

    # Verifique os valores das primeiras imagens e alvos esperados
    expected_images = [
        [
            [1, 2, 3, 4],
            [0.1, 0.3, 0.5, 0.7],
            [0.2, 0.4, 0.6, 0.8],
            [0.11, 0.13, 0.15, 0.17]
        ],
        [
            [2, 3, 4, 5],
            [0.3, 0.5, 0.7, 0.9],
            [0.4, 0.6, 0.8, 0.10],
            [0.13, 0.15, 0.17, 0.19]
        ],
        [
            [3, 4, 5, 6],
            [0.5, 0.7, 0.9, 0.11],
            [0.6, 0.8, 0.10, 0.12],
            [0.15, 0.17, 0.19, 0.21]
        ]
    ]
    
    expected_targets = [37, 36, 35]

    for i, (expected_image, image) in enumerate(zip(expected_images, images)):
        assert np.allclose(expected_image, image), f"Image {i} does not match the expected values"

    for i, (expected_target, target) in enumerate(zip(expected_targets, targets)):
        assert expected_target == target, f"Target {i} does not match the expected value"

if __name__ == "__main__":
    pytest.main()
