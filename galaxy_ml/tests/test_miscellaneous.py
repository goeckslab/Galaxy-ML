"""Behavioral coverage for modern APIs in the pinned dependencies."""

import h5py
from joblib import Parallel, delayed
import keras
import numpy as np
import pandas as pd
import pytest
from scipy.stats import Normal
from sklearn import config_context
from sklearn.base import BaseEstimator
from sklearn.calibration import FrozenEstimator
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.utils.validation import validate_data
import tensorflow as tf
from xgboost import XGBClassifier


def test_numpy_vector_norm_supports_batched_arrays():
    values = np.array([[3.0, 4.0], [5.0, 12.0]])

    norms = np.linalg.vector_norm(values, axis=1)

    np.testing.assert_allclose(norms, [5.0, 13.0])


def test_pandas_expression_columns_and_copy_on_write():
    frame = pd.DataFrame({'left': [1, 2], 'right': [3, 4]})
    result = frame.assign(total=pd.col('left') + pd.col('right'))
    shallow_copy = result.copy(deep=False)

    shallow_copy.loc[0, 'total'] = 99

    assert result['total'].tolist() == [4, 6]
    assert shallow_copy['total'].tolist() == [99, 6]


def test_scipy_distribution_new_class_api():
    normal = Normal(mu=0.0, sigma=1.0)

    assert normal.cdf(0.0) == pytest.approx(0.5)
    assert normal.ccdf(0.0) == pytest.approx(0.5)
    assert normal.icdf(0.5) == pytest.approx(0.0)


def test_sklearn_public_validate_data_tracks_feature_count():
    estimator = BaseEstimator()
    X = np.ones((3, 2))

    validated = validate_data(estimator, X, reset=True)

    assert validated.shape == (3, 2)
    assert estimator.n_features_in_ == 2
    with pytest.raises(ValueError, match='2 features'):
        validate_data(estimator, np.ones((3, 3)), reset=False)


def test_h5py_native_numpy_string_dtype_round_trip(tmp_path):
    path = tmp_path / 'native-strings.h5'
    strings = np.asarray(
        ['alpha', 'beta'], dtype=np.dtypes.StringDType())

    with h5py.File(path, 'w') as handle:
        handle.create_dataset('values', data=strings, dtype='T')

    with h5py.File(path, 'r') as handle:
        restored = handle['values'].astype('T')[:]

    assert isinstance(restored.dtype, np.dtypes.StringDType)
    assert restored.tolist() == ['alpha', 'beta']


def test_keras_backend_neutral_linalg_ops():
    values = keras.ops.convert_to_tensor([[3.0, 4.0], [5.0, 12.0]])

    norms = keras.ops.linalg.norm(values, axis=1)

    np.testing.assert_allclose(keras.ops.convert_to_numpy(norms), [5.0, 13.0])


def test_joblib_streams_unordered_results():
    results = Parallel(
        n_jobs=2, prefer='threads', return_as='generator_unordered')(
            delayed(pow)(value, 2) for value in range(4))

    assert not isinstance(results, list)
    assert sorted(results) == [0, 1, 4, 9]


def test_xgboost_sklearn_metadata_routing():
    with config_context(enable_metadata_routing=True):
        classifier = XGBClassifier().set_fit_request(sample_weight=True)
        routing = classifier.get_metadata_routing()

    assert routing.consumes('fit', ['sample_weight']) == {'sample_weight'}


def test_sklearn_root_mean_squared_error():
    actual = root_mean_squared_error([0.0, 0.0], [3.0, 4.0])

    assert actual == pytest.approx(np.sqrt(12.5))


def test_sklearn_frozen_estimator_preserves_fitted_state():
    X, y = make_classification(n_samples=30, n_features=4, random_state=0)
    fitted = LogisticRegression(random_state=0).fit(X, y)
    coefficients = fitted.coef_.copy()
    frozen = FrozenEstimator(fitted)

    returned = frozen.fit(X, 1 - y)

    assert returned is frozen
    np.testing.assert_array_equal(frozen.estimator.coef_, coefficients)


def test_sklearn_tuned_threshold_classifier():
    X, y = make_classification(
        n_samples=60, n_features=5, weights=[0.75, 0.25], random_state=0)
    classifier = TunedThresholdClassifierCV(
        LogisticRegression(random_state=0), cv=3, thresholds=10)

    classifier.fit(X, y)

    assert 0.0 <= classifier.best_threshold_ <= 1.0
    assert classifier.predict(X).shape == y.shape


def test_sklearn_estimator_metadata_routing():
    with config_context(enable_metadata_routing=True):
        estimator = LogisticRegression().set_fit_request(sample_weight=True)
        routing = estimator.get_metadata_routing()

    assert routing.consumes('fit', ['sample_weight']) == {'sample_weight'}


def test_keras_seed_generator_advances_reproducibly():
    def samples():
        seed = keras.random.SeedGenerator(42)
        return [
            keras.ops.convert_to_numpy(keras.random.normal((3,), seed=seed)),
            keras.ops.convert_to_numpy(keras.random.normal((3,), seed=seed)),
        ]

    first_run = samples()
    second_run = samples()

    assert not np.array_equal(first_run[0], first_run[1])
    np.testing.assert_array_equal(first_run, second_run)


def test_keras_layer_stateless_call_does_not_mutate_weights():
    layer = keras.layers.Dense(1, use_bias=False, kernel_initializer='ones')
    inputs = keras.ops.ones((2, 2))
    layer(inputs)
    original_weights = [keras.ops.copy(value) for value in layer.weights]

    outputs, non_trainable = layer.stateless_call(
        layer.trainable_variables, layer.non_trainable_variables, inputs)

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(outputs), [[2.], [2.]]
    )
    assert non_trainable == []
    for current, original in zip(layer.weights, original_weights):
        np.testing.assert_array_equal(current, original)


def test_keras_export_and_tfsm_layer_round_trip(tmp_path):
    model = keras.Sequential([
        keras.Input(shape=(2,)),
        keras.layers.Dense(1, use_bias=False, kernel_initializer='ones'),
    ])
    export_path = tmp_path / 'saved_model'

    model.export(export_path, verbose=False)
    restored = keras.layers.TFSMLayer(export_path, call_endpoint='serve')

    output = restored(keras.ops.array([[2.0, 3.0]]))
    np.testing.assert_allclose(keras.ops.convert_to_numpy(output), [[5.0]])


def test_tensorflow_dataset_rebatch():
    dataset = tf.data.Dataset.range(6).batch(3).rebatch(2)

    batches = [batch.numpy().tolist() for batch in dataset]

    assert batches == [[0, 1], [2, 3], [4, 5]]


def test_tensorflow_random_split_and_fold_in_are_deterministic():
    seed = tf.constant([1, 2], dtype=tf.int32)

    split_seeds = tf.random.split(seed, 2)
    folded_seed = tf.random.fold_in(seed, 7)

    np.testing.assert_array_equal(split_seeds, tf.random.split(seed, 2))
    np.testing.assert_array_equal(folded_seed, tf.random.fold_in(seed, 7))
    assert split_seeds.shape == (2, 2)


def test_tensorflow_extension_type_as_dict():
    class Pair(tf.experimental.ExtensionType):
        left: tf.Tensor
        right: tf.Tensor

    pair = Pair(tf.constant(1), tf.constant(2))
    values = tf.experimental.extension_type.as_dict(pair)

    assert values['left'].numpy() == 1
    assert values['right'].numpy() == 2


def test_tensorflow_saved_model_fingerprint(tmp_path):
    export_path = tmp_path / 'module'
    module = tf.Module()
    module.value = tf.Variable(3.0)
    tf.saved_model.save(module, export_path)

    fingerprint = tf.saved_model.experimental.read_fingerprint(
        str(export_path))

    assert fingerprint.singleprint()
