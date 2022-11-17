import warnings

from galaxy_ml.preprocessors import TDMScaler

import numpy as np


warnings.simplefilter('ignore')


X = [
    [1., -2., 2.],
    [-2., 1., 3.],
    [4., 1., -2.],
]


def test_self_transform():
    scaler = TDMScaler()
    scaler.fit(X)
    got = scaler.transform(X)

    assert np.array_equal(X, got), got
