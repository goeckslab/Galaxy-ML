"""Regression coverage for accuracy hidden behind Keras 3 compile_metrics."""

import numpy as np
import pytest
from sklearn.metrics import check_scoring

import keras

from galaxy_ml.keras_galaxy_models import (
    KerasGBatchClassifier, KerasGClassifier, KerasGRegressor,
)


class _ArrayBatches(keras.utils.PyDataset):
    def __init__(self, X, y, batch_size):
        super().__init__()
        self.X, self.y, self.batch_size = X, y, batch_size

    def __len__(self):
        return (len(self.X) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        start = index * self.batch_size
        stop = start + self.batch_size
        return self.X[start:stop], self.y[start:stop]


class _ArrayGenerator:
    def flow(self, X, y, batch_size):
        return _ArrayBatches(X, y, batch_size)


@pytest.mark.parametrize('classifier_type', [
    KerasGClassifier, KerasGBatchClassifier, KerasGRegressor,
])
@pytest.mark.parametrize('metric_name', ['acc', 'accuracy'])
@pytest.mark.parametrize('default_scorer', [False, True],
                         ids=['score', 'sklearn-default-score'])
def test_estimator_score_with_keras_metrics(
        classifier_type, metric_name, default_scorer):
    # Fixed weights predict class 1 for every sample: exactly 75% accuracy.
    # No training, random initialization, or external data is needed.
    X = np.array([[0.], [1.], [2.], [3.]], dtype='float32')
    y = np.array([0, 1, 1, 1])
    model = keras.Sequential([
        keras.Input(shape=(1,)),
        keras.layers.Dense(1, activation='sigmoid',
                           kernel_initializer='zeros',
                           bias_initializer='ones'),
    ])
    model.compile(loss='binary_crossentropy', metrics=[metric_name])

    classifier = classifier_type(
        model.get_config(), loss='binary_crossentropy',
        metrics=[metric_name], batch_size=4, verbose=0)
    # Supply the fitted state directly to isolate scoring from training.
    classifier.model_ = model
    if classifier_type is not KerasGRegressor:
        classifier.classes_ = np.array([0, 1])
        if classifier_type is KerasGBatchClassifier:
            classifier.data_generator_ = _ArrayGenerator()

    results = model.evaluate(X, y, verbose=0, return_dict=True)
    assert model.metrics_names == ['loss', 'compile_metrics']
    assert np.isclose(0.75, results[metric_name])

    if default_scorer:
        actual = check_scoring(classifier, scoring=None)(classifier, X, y)
    else:
        actual = classifier.score(X, y)

    if classifier_type is KerasGRegressor:
        expected = -results['loss']
    else:
        # Keras 3 hides accuracy behind the ``compile_metrics`` placeholder
        # in ``metrics_names``, while preserving it in the result dictionary.
        expected = results[metric_name]

    assert np.isclose(actual, expected)
