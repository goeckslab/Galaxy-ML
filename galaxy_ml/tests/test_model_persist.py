import json
import numpy as np
import os
import pandas as pd
import pickle
import tempfile
import time

import galaxy_ml

from nose.tools import nottest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier
from galaxy_ml.keras_galaxy_models import KerasGClassifier
from galaxy_ml import model_persist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


df = pd.read_csv('./tools/test-data/pima-indians-diabetes.csv', sep=',')
X = df.iloc[:, 0:8].values.astype(float)
y = df.iloc[:, 8].values

splitter = StratifiedShuffleSplit(n_splits=1, random_state=0)
train, test = next(splitter.split(X, y))

X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]

gbc = GradientBoostingClassifier(n_estimators=101, random_state=42)

gbc.fit(X_train, y_train)

module_folder = (os.path.dirname(galaxy_ml.__file__))
gbc_pickle = os.path.join(module_folder,
                          './tools/test-data/gbc_model01.zip')
_, tmp_gbc_pickle = tempfile.mkstemp(suffix='.zip')

gbc_json = os.path.join(module_folder,
                        './tools/test-data/gbc_model01.json')
gbc_h5 = os.path.join(module_folder,
                      'tools/test-data/gbc_model01.h5')
_, tmp_gbc_h5 = tempfile.mkstemp(suffix='.h5')

xgbc_json = os.path.join(module_folder,
                         'tools/test-data/xgbc_model01.json')
xgbc_h5 = os.path.join(module_folder,
                       'tools/test-data/xgbc_model01.h5')
_, tmp_xgbc_h5 = tempfile.mkstemp(suffix='.h5')

kgc_h5 = os.path.join(module_folder,
                      'tools/test-data/kgc_model01.h5')
_, tmp_kgc_h5 = tempfile.mkstemp(suffix='.h5')


def teardown():
    os.remove(tmp_gbc_pickle)
    os.remove(tmp_gbc_h5)
    os.remove(tmp_xgbc_h5)
    os.remove(tmp_kgc_h5)


def test_jpickle_dumpc():
    # GradientBoostingClassifier
    got = model_persist.dumpc(gbc)
    r_model = model_persist.loadc(got)

    assert np.array_equal(
        gbc.predict(X_test),
        r_model.predict(X_test)
    )

    got.pop('-cpython-')

    with open(gbc_json, 'w') as f:
        json.dump(got, f, indent=2)

    with open(gbc_json, 'r') as f:
        expect = json.load(f)
    expect.pop('-cpython-', None)

    assert got == expect, got


def test_gbc_dump_and_load():

    print("\nDumping GradientBoostingClassifier model using pickle...")
    start_time = time.time()
    with open(tmp_gbc_pickle, 'wb') as f:
        pickle.dump(gbc, f, protocol=0)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))
    print("File size: %s" % str(os.path.getsize(tmp_gbc_pickle)))
    diff = os.path.getsize(tmp_gbc_pickle) - os.path.getsize(gbc_pickle)
    assert abs(diff) < 50

    print("\nDumping object to dict...")
    start_time = time.time()
    model_dict = model_persist.dumpc(gbc)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))

    print("\nDumping dict data to JSON file...")
    start_time = time.time()
    with open(gbc_json, 'w') as f:
        json.dump(model_dict, f, sort_keys=True)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))
    print("File size: %s" % str(os.path.getsize(gbc_json)))

    print("\nLoading data from JSON file...")
    start_time = time.time()
    with open(gbc_json, 'r') as f:
        json.load(f)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))

    print("\nRe-build the model object...")
    start_time = time.time()
    re_model = model_persist.loadc(model_dict)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))
    print("%r" % re_model)

    print("\nDumping object to HDF5...")
    start_time = time.time()
    model_dict = model_persist.dump_model_to_h5(gbc, tmp_gbc_h5)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))
    print("File size: %s" % str(os.path.getsize(tmp_gbc_h5)))
    diff = os.path.getsize(tmp_gbc_h5) - os.path.getsize(gbc_h5)
    assert abs(diff) < 20, os.path.getsize(gbc_h5)

    print("\nLoading hdf5 model...")
    start_time = time.time()
    model = model_persist.load_model_from_h5(tmp_gbc_h5)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))

    assert np.array_equal(
        gbc.predict(X_test),
        model.predict(X_test)
    )


# CircleCI timeout with xgboost for no reason.
@nottest
def test_xgb_dump_and_load():
    xgbc = XGBClassifier(n_estimators=101, random_state=42, n_jobs=1)

    model_persist.dump_model_to_h5(xgbc, tmp_xgbc_h5)
    model_persist.load_model_from_h5(tmp_xgbc_h5)

    xgbc.fit(X_train, y_train)

    got = model_persist.dumpc(xgbc)
    r_model = model_persist.loadc(got)

    assert np.array_equal(
        xgbc.predict(X_test),
        r_model.predict(X_test)
    )

    print("\nDumping XGBC to dict...")
    start_time = time.time()
    model_dict = model_persist.dumpc(xgbc)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))

    print("\nDumping dict data to JSON file...")
    start_time = time.time()
    with open(xgbc_json, 'w') as f:
        json.dump(model_dict, f, sort_keys=True)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))
    print("File size: %s" % str(os.path.getsize(xgbc_json)))

    print("\nLoading data from JSON file...")
    start_time = time.time()
    with open(xgbc_json, 'r') as f:
        json.load(f)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))

    print("\nRe-build the model object...")
    start_time = time.time()
    re_model = model_persist.loadc(model_dict)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))
    print("%r" % re_model)

    print("\nDumping object to HDF5...")
    start_time = time.time()
    model_persist.dump_model_to_h5(xgbc, tmp_xgbc_h5)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))
    print("File size: %s" % str(os.path.getsize(tmp_xgbc_h5)))
    diff = os.path.getsize(tmp_xgbc_h5) - os.path.getsize(xgbc_h5)
    assert abs(diff) < 20, os.path.getsize(xgbc_h5)

    print("\nLoading hdf5 model...")
    start_time = time.time()
    model = model_persist.load_model_from_h5(tmp_xgbc_h5)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))

    assert np.array_equal(
        xgbc.predict(X_test),
        model.predict(X_test)
    )


# KerasGClassifier
def test_keras_dump_and_load():

    train_model = Sequential()
    train_model.add(Dense(12, input_dim=8, activation='relu'))
    train_model.add(Dense(1, activation='softmax'))
    config = train_model.get_config()

    kgc = KerasGClassifier(config, loss='binary_crossentropy',
                           metrics=['acc'], seed=42)

    kgc.fit(X_train, y_train)

    print("\nDumping KerasGClassifer to HDF5...")
    start_time = time.time()
    model_persist.dump_model_to_h5(kgc, tmp_kgc_h5)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))
    print("File size: %s" % str(os.path.getsize(tmp_kgc_h5)))
    diff = os.path.getsize(tmp_kgc_h5) - os.path.getsize(kgc_h5)
    assert abs(diff) < 40, os.path.getsize(kgc_h5)

    print("\nLoading hdf5 model...")
    start_time = time.time()
    model = model_persist.load_model_from_h5(tmp_kgc_h5)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))

    assert np.array_equal(
        kgc.predict(X_test),
        model.predict(X_test)
    )


if __name__ == '__main__':
    """ Generate couple of estimators for tests.
    """
    import xgboost
    from pathlib import Path
    from sklearn import linear_model, ensemble, pipeline

    test_folder = Path(__file__).parent.parent
    test_folder = test_folder.joinpath('tools', 'test-data')

    estimator = linear_model.LinearRegression()
    model_persist.dump_model_to_h5(
        estimator,
        test_folder.joinpath('LinearRegression01.h5'))

    estimator = ensemble.RandomForestRegressor(
        n_estimators=10, random_state=10)
    model_persist.dump_model_to_h5(
        estimator,
        test_folder.joinpath('RandomForestRegressor01.h5'))

    estimator = xgboost.XGBRegressor(
        learning_rate=0.1, n_estimators=100, random_state=0)
    model_persist.dump_model_to_h5(
        estimator,
        test_folder.joinpath('XGBRegressor01.h5'))

    estimator = ensemble.AdaBoostRegressor(
        learning_rate=1.0, n_estimators=50)
    pipe = pipeline.make_pipeline(estimator)
    model_persist.dump_model_to_h5(
        pipe,
        test_folder.joinpath('pipeline10'))
