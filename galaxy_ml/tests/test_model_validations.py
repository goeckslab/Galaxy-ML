"""Tests for `model_validations` module.
"""
import warnings
import pytest
import pandas as pd
import numpy as np

from galaxy_ml.model_validations import (
    train_test_split, OrderedKFold, RepeatedOrderedKFold)
from galaxy_ml.model_validations import _fit_and_score
from galaxy_ml.keras_galaxy_models import KerasGClassifier

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from sklearn.ensemble import RandomForestRegressor
from six.moves import zip
from sklearn.model_selection import _search
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import SCORERS
from sklearn.utils.mocking import MockDataFrame
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import ignore_warnings


setattr(_search, '_fit_and_score', _fit_and_score)

warnings.simplefilter('ignore')

train_test_split.__test__ = False

df = pd.read_csv('./tools/test-data/pima-indians-diabetes.csv', sep=',')
X = df.iloc[:, 0:8].values.astype(float)
y = df.iloc[:, 8].values

train_model = Sequential()
train_model.add(Dense(12, input_dim=8, activation='relu'))
train_model.add(Dense(1, activation='sigmoid'))


def test_train_test_split_errors():
    pytest.raises(ValueError, train_test_split)

    pytest.raises(ValueError, train_test_split, range(3), train_size=1.1)

    pytest.raises(ValueError, train_test_split, range(3), test_size=0.6,
                  train_size=0.6)
    pytest.raises(ValueError, train_test_split, range(3),
                  test_size=np.float32(0.6), train_size=np.float32(0.6))
    pytest.raises(ValueError, train_test_split, range(3),
                  test_size="wrong_type")
    pytest.raises(ValueError, train_test_split, range(3), test_size=2,
                  train_size=4)
    pytest.raises(TypeError, train_test_split, range(3),
                  some_argument=1.1)
    pytest.raises(ValueError, train_test_split, range(3), range(42))
    pytest.raises(ValueError, train_test_split, range(10),
                  shuffle=True)
    pytest.raises(ValueError, train_test_split, range(10),
                  shuffle='group', labels=None)
    pytest.raises(TypeError, train_test_split, range(10),
                  shuffle='stratified', labels=None)

    with pytest.raises(ValueError,
                       match=r'train_size=11 should be either positive and '
                             r'smaller than the number of samples 10 or a '
                             r'float in the \(0, 1\) range'):
        train_test_split(range(10), train_size=11, test_size=1)


def test_train_test_split():
    X = np.arange(100).reshape((10, 10))
    X_s = coo_matrix(X)
    y = np.arange(10)

    # simple test
    split = train_test_split(X, y, test_size=None, train_size=.5)
    X_train, X_test, y_train, y_test = split
    assert_equal(len(y_test), len(y_train))
    # test correspondence of X and y
    assert_array_equal(X_train[:, 0], y_train * 10)
    assert_array_equal(X_test[:, 0], y_test * 10)

    # don't convert lists to anything else by default
    split = train_test_split(X, X_s, y.tolist())
    X_train, X_test, X_s_train, X_s_test, y_train, y_test = split
    assert isinstance(y_train, list)
    assert isinstance(y_test, list)

    # allow nd-arrays
    X_4d = np.arange(10 * 5 * 3 * 2).reshape(10, 5, 3, 2)
    y_3d = np.arange(10 * 7 * 11).reshape(10, 7, 11)
    split = train_test_split(X_4d, y_3d)
    assert_equal(split[0].shape, (7, 5, 3, 2))
    assert_equal(split[1].shape, (3, 5, 3, 2))
    assert_equal(split[2].shape, (7, 7, 11))
    assert_equal(split[3].shape, (3, 7, 11))

    # test shuffle='stratified' option
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    for test_size, exp_test_size in zip([2, 4, 0.25, 0.5, 0.75],
                                        [2, 4, 2, 4, 6]):
        train, test = train_test_split(y, test_size=test_size,
                                       shuffle='stratified',
                                       labels=y,
                                       random_state=0)
        assert_equal(len(test), exp_test_size)
        assert_equal(len(test) + len(train), len(y))
        # check the 1:1 ratio of ones and twos in the data is preserved
        assert_equal(np.sum(train == 1), np.sum(train == 2))

    # test shuffle='group' option
    y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    for test_size, exp_test_size in zip([1, 2, 0.25, 0.5, 0.75],
                                        [3, 6, 3, 6, 9]):
        train, test = train_test_split(y, test_size=test_size,
                                       shuffle='group',
                                       labels=y,
                                       random_state=0)
        assert_equal(len(test), exp_test_size)
        assert_equal(np.unique(test).size, exp_test_size//3)
        assert_equal(len(test) + len(train), len(y))
        assert_equal(np.unique(train).size + np.unique(test).size, 4)

    # test shuffle='simple' option
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    for test_size, exp_test_size in zip([2, 4, 0.25, 0.5, 0.75],
                                        [2, 4, 2, 4, 6]):
        train, test = train_test_split(y, test_size=test_size,
                                       shuffle='simple',
                                       labels=y,
                                       random_state=42)
        assert_equal(len(test), exp_test_size)
        assert_equal(len(test) + len(train), len(y))

    # test unshuffled split
    y = np.arange(10)
    for test_size in [2, 0.2]:
        train, test = train_test_split(y, shuffle=None, test_size=test_size)
        assert_array_equal(test, [8, 9])
        assert_array_equal(train, [0, 1, 2, 3, 4, 5, 6, 7])


@ignore_warnings
def test_train_test_split_pandas():
    X = np.ones(10)
    # check train_test_split doesn't destroy pandas dataframe
    types = [MockDataFrame]
    try:
        from pandas import DataFrame
        types.append(DataFrame)
    except ImportError:
        pass
    for InputFeatureType in types:
        # X dataframe
        X_df = InputFeatureType(X)
        X_train, X_test = train_test_split(X_df)
        assert isinstance(X_train, InputFeatureType)
        assert isinstance(X_test, InputFeatureType)


def test_train_test_split_sparse():
    # check that train_test_split converts scipy sparse matrices
    # to csr, as stated in the documentation
    X = np.arange(100).reshape((10, 10))
    sparse_types = [csr_matrix, csc_matrix, coo_matrix]
    for InputFeatureType in sparse_types:
        X_s = InputFeatureType(X)
        X_train, X_test = train_test_split(X_s)
        assert isinstance(X_train, csr_matrix)
        assert isinstance(X_test, csr_matrix)


def test_train_test_split_mock_pandas():
    X = np.ones(10)
    # X mock dataframe
    X_df = MockDataFrame(X)
    X_train, X_test = train_test_split(X_df)
    assert isinstance(X_train, MockDataFrame)
    assert isinstance(X_test, MockDataFrame)
    X_train_arr, X_test_arr = train_test_split(X_df)


def test_train_test_split_list_input():
    # Check that when y is a list / list of string labels, it works.
    X = np.ones(7)
    y1 = ['1'] * 4 + ['0'] * 3
    y2 = np.hstack((np.ones(4), np.zeros(3)))
    y3 = y2.tolist()

    for shuffle in (None, 'simple', 'stratified', 'group'):
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X, y1, random_state=0,
            labels=y1 if shuffle in ('stratified', 'group') else None)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y2, random_state=0,
            labels=y2 if shuffle in ('stratified', 'group') else None)
        X_train3, X_test3, y_train3, y_test3 = train_test_split(
            X, y3, random_state=0,
            labels=y3 if shuffle in ('stratified', 'group') else None)

        np.testing.assert_equal(X_train1, X_train2)
        np.testing.assert_equal(y_train2, y_train3)
        np.testing.assert_equal(X_test1, X_test3)
        np.testing.assert_equal(y_test3, y_test2)


def test_ordered_kfold():
    y = list(range(12))
    got1 = []
    cv = OrderedKFold(3)
    for _, test_index in cv.split(y, y):
        got1.append(test_index.tolist())

    expect1 = [[0, 3, 6, 9],
               [1, 4, 7, 10],
               [2, 5, 8, 11]]

    assert got1 == expect1, got1

    got2 = []
    cv = OrderedKFold(3, shuffle=True, random_state=0)
    for _, test_index in cv.split(y, y):
        got2.append(test_index.tolist())

    expect2 = [[2, 5, 6, 11],
               [1, 3, 8, 9],
               [0, 4, 7, 10]]

    assert got2 == expect2, got2

    got3 = []
    cv = OrderedKFold(3, shuffle=True, random_state=10)
    for _, test_index in cv.split(y, y):
        got3.append(test_index.tolist())

    expect3 = [[0, 5, 8, 11],
               [2, 4, 7, 10],
               [1, 3, 6, 9]]

    assert got3 == expect3, got3


def test_repeated_ordered_kfold():
    y = list(range(12))
    got1 = []
    cv = RepeatedOrderedKFold(n_splits=4, n_repeats=2, random_state=42)
    for _, test_index in cv.split(y, y):
        got1.append(test_index.tolist())

    expect1 = [[1, 5, 11],
               [3, 7, 8],
               [0, 4, 9],
               [2, 6, 10],
               [1, 6, 8],
               [0, 5, 9],
               [3, 4, 10],
               [2, 7, 11]]

    assert got1 == expect1, got1

    got2 = []
    cv = RepeatedOrderedKFold(n_splits=4, n_repeats=2, random_state=999)
    for _, test_index in cv.split(y, y):
        got2.append(test_index.tolist())

    expect2 = [[2, 6, 8],
               [1, 7, 10],
               [3, 4, 9],
               [0, 5, 11],
               [3, 5, 8],
               [2, 6, 11],
               [1, 7, 10],
               [0, 4, 9]]

    assert got2 == expect2, got2


def test_fit_and_score():
    X = np.arange(100).reshape((10, 10))
    y = np.arange(10)
    estimator = RandomForestRegressor(random_state=42)

    scorer = SCORERS['r2']
    train, test = next(KFold(n_splits=5).split(X, y))

    parameters = {}
    fit_params = {}

    got1 = _fit_and_score(estimator, X, y, scorer, train, test,
                          verbose=0, parameters=parameters,
                          fit_params=fit_params)

    expect1 = [-16.0]
    assert expect1 == got1, got1

    got2 = _fit_and_score(estimator, X, y, scorer, train, test,
                          verbose=0, parameters=parameters,
                          fit_params=fit_params,
                          return_train_score=True)

    expect2 = [0.97, -16.0]
    assert expect2 == [round(x, 2) for x in got2], got2


def test_fit_and_score_keras_model():
    config = train_model.get_config()
    regressor = KerasGClassifier(config, optimizer='adam',
                                 metrics=[], batch_size=32,
                                 epochs=30, seed=42)

    scorer = SCORERS['accuracy']
    train, test = next(KFold(n_splits=5).split(X, y))
    assert np.array_equal(test, np.arange(154)), test

    new_params = {
        'layers_0_Dense__config__kernel_initializer__config__seed': 0,
        'layers_1_Dense__config__kernel_initializer__config__seed': 0
    }
    parameters = new_params
    fit_params = {'shuffle': False}

    got1 = _fit_and_score(regressor, X, y, scorer, train, test,
                          verbose=0, parameters=parameters,
                          fit_params=fit_params)

    assert round(got1[0], 2) == 0.69, got1


def test_fit_and_score_keras_model_callbacks():
    config = train_model.get_config()
    regressor = KerasGClassifier(config, optimizer='adam',
                                 metrics=[], batch_size=32,
                                 epochs=500, seed=42)

    scorer = SCORERS['accuracy']
    train, test = next(KFold(n_splits=5).split(X, y))

    new_params = {
        'layers_0_Dense__config__kernel_initializer__config__seed': 0,
        'layers_1_Dense__config__kernel_initializer__config__seed': 0
    }
    parameters = new_params
    callbacks = [EarlyStopping()]
    fit_params = {'shuffle': False, 'callbacks': callbacks}

    got1 = _fit_and_score(regressor, X, y, scorer, train, test,
                          verbose=0, parameters=parameters,
                          fit_params=fit_params)

    assert round(got1[0], 2) == 0.70, got1


def test_fit_and_score_keras_model_in_gridsearchcv():
    config = train_model.get_config()
    clf = KerasGClassifier(config, optimizer='adam',
                           metrics=[], batch_size=32,
                           epochs=10, seed=42)

    df = pd.read_csv('./tools/test-data/pima-indians-diabetes.csv', sep=',')
    X = df.iloc[:, 0:8].values.astype(float)
    y = df.iloc[:, 8].values

    scorer = SCORERS['balanced_accuracy']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    new_params = {
        'layers_0_Dense__config__kernel_initializer__config__seed': [0],
        'layers_1_Dense__config__kernel_initializer__config__seed': [0]
    }

    grid = GridSearchCV(clf, param_grid=new_params, scoring=scorer,
                        cv=cv, n_jobs=2, refit=False)

    grid.fit(X, y, verbose=0)

    got1 = grid.best_score_

    assert round(got1, 2) == 0.53, got1
