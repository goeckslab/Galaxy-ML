import json
import os
import pickle
import re
import tempfile
import time

import galaxy_ml
from galaxy_ml import model_persist
from galaxy_ml.keras_galaxy_models import KerasGClassifier

from imblearn.over_sampling import SMOTEN
from imblearn.pipeline import make_pipeline

from nose.tools import nottest

import numpy as np

import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedShuffleSplit

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

from xgboost import XGBClassifier


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


# imbalanced-learn
def test_imblearn_dump_and_load():
    smt = SMOTEN(random_state=42)
    rfc = RandomForestClassifier(random_state=10)

    pipe = make_pipeline(smt, rfc)

    pipe.fit(X_train, y_train)

    _, tmp_imb_pipe_h5 = tempfile.mkstemp(suffix='.h5')
    imb_pipe_h5 = os.path.join(module_folder,
                               'tools/test-data/imb_pipeline.h5')

    print("\nDumping imblearn pipeline to HDF5...")
    start_time = time.time()
    model_persist.dump_model_to_h5(pipe, tmp_imb_pipe_h5)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))
    print("File size: %s" % str(os.path.getsize(tmp_imb_pipe_h5)))
    diff = os.path.getsize(tmp_imb_pipe_h5) - os.path.getsize(imb_pipe_h5)
    assert abs(diff) < 40, os.path.getsize(imb_pipe_h5)

    print("\nLoading hdf5 model...")
    start_time = time.time()
    model = model_persist.load_model_from_h5(tmp_imb_pipe_h5)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))

    assert np.array_equal(
        pipe.predict(X_test),
        model.predict(X_test)
    )
    os.remove(tmp_imb_pipe_h5)


def test_safe_load_model():
    model = './tools/test-data/RandomForestRegressor01.zip'
    with open(model, 'rb') as fh:
        safe_unpickler = model_persist._SafePickler(fh)

    assert RandomForestClassifier == \
        safe_unpickler.find_class(
            'sklearn.ensemble._forest',
            'RandomForestClassifier',
        )

    test_folder = './tools/test-data'
    for name in os.listdir(test_folder):
        if re.match(
            r'^(?!.*(json|\.h5|\.h5mlm)).*(pipeline|model|regressor)\d+.*$',
            name,
            flags=re.I,
        ):
            if name in ('gbr_model01_py3', 'rfr_model01'):
                continue
            model_path = os.path.join(test_folder, name)
            print(model_path)
            if model_path.endswith('.zip'):
                with open(model_path, 'rb') as fh:
                    model_persist.safe_load_model(fh)
            else:
                model_persist.load_model_from_h5(model_path)


def test_find_members():
    got = model_persist.find_members('galaxy_ml.metrics')
    expect = [
        'galaxy_ml.metrics._regression.spearman_correlation_score'
    ]
    assert got == expect, got

    got = model_persist.find_members('imblearn.over_sampling')
    expect = [
        "imblearn.over_sampling._adasyn.ADASYN",
        "imblearn.over_sampling._random_over_sampler.RandomOverSampler",
        "imblearn.over_sampling._smote.base.BaseSMOTE",
        "imblearn.over_sampling._smote.base.SMOTE",
        "imblearn.over_sampling._smote.base.SMOTEN",
        "imblearn.over_sampling._smote.base.SMOTENC",
        "imblearn.over_sampling._smote.cluster.KMeansSMOTE",
        "imblearn.over_sampling._smote.filter.BorderlineSMOTE",
        "imblearn.over_sampling._smote.filter.SVMSMOTE",
        "imblearn.over_sampling.base.BaseOverSampler"
    ]
    assert got == expect, got
