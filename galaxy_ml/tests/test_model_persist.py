import json
import os
import glob
import pickle
import tempfile
import time
import sys
import galaxy_ml
from galaxy_ml import model_persist

from nose.tools import nottest


test_model = './tools/test-data/gbr_model01_py3'
result_json = './tools/test-data/gbr_model01_py3.json'
module_folder = (os.path.dirname(galaxy_ml.__file__))
result_h5 = os.path.join(module_folder,
                         'tools/test-data/gbr_model01_py3.h5')


def teardown():
    files = glob.glob('./tests/*.hdf5', recursive=False)
    for fl in files:
        os.remove(fl)
    log_file = glob.glob('./tests/log.cvs', recursive=False)
    for fl in log_file:
        os.remove(fl)


def test_jpickle_dumpc():
    with open(test_model, 'rb') as f:
        model = pickle.load(f)

    got = model_persist.dumpc(model)
    got.pop('-cpython-')

    #with open(result_json, 'w') as f:
    #    json.dump(got, f, indent=2)

    with open(result_json, 'r') as f:
        expect = json.load(f)
    expect.pop('-cpython-')

    assert got == expect, got


def test_hdf5_model_dump_and_load():

    print("Loading pickled test model...")
    start_time = time.time()
    with open(test_model, 'rb') as f:
        model = pickle.load(f)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))

    tmp = tempfile.mktemp()

    try:
        print("\nDumping model using pickle...")
        start_time = time.time()
        with open(tmp, 'wb') as f:
            pickle.dump(model, f, protocol=0)
        end_time = time.time()
        print("(%s s)" % str(end_time - start_time))
        print("File size: %s" % str(os.path.getsize(tmp)))

        print("\nDumping object to dict...")
        start_time = time.time()
        model_dict = model_persist.dumpc(model)
        end_time = time.time()
        print("(%s s)" % str(end_time - start_time))
    finally:
        os.remove(tmp)

    tmp = tempfile.mktemp()

    try:
        print("\nDumping dict data to JSON file...")
        start_time = time.time()
        with open(tmp, 'w') as f:
            json.dump(model_dict, f, sort_keys=True)
        end_time = time.time()
        print("(%s s)" % str(end_time - start_time))
        print("File size: %s" % str(os.path.getsize(tmp)))

        print("\nLoading data from JSON file...")
        start_time = time.time()
        with open(tmp, 'r') as f:
            new_dict = json.load(f)
        end_time = time.time()
        print("(%s s)" % str(end_time - start_time))

        print("\nRe-build the model object...")
        start_time = time.time()
        re_model = model_persist.loadc(model_dict)
        end_time = time.time()
        print("(%s s)" % str(end_time - start_time))
        print("%r" % re_model)
    finally:
        os.remove(tmp)

    tmp = tempfile.mktemp()

    try:
        print("\nDumping model using pickle hdf5...")
        start_time = time.time()
        model_persist.dump_model_to_h5(model, tmp)
        end_time = time.time()
        print("(%s s)" % str(end_time - start_time))
        print("File size: %s" % str(os.path.getsize(tmp)))

        print("\nLoading hdf5 model...")
        start_time = time.time()
        model = model_persist.load_model_from_h5(tmp)
        end_time = time.time()
        print("(%s s)" % str(end_time - start_time))
    finally:
        os.remove(tmp)


def test_hdf5_model_keras():

    model_weights = './tools/test-data/train_test_eval_weights01.h5'
    model_config = './tools/test-data/train_test_eval_model01'

    with open(model_config, 'rb') as f:
        model = pickle.load(f)

    model.load_weights(model_weights)

    print(model)

    tmp = tempfile.mktemp()

    try:
        print("\nDumping model to hdf5...")
        start_time = time.time()
        model_persist.dump_model_to_h5(model, tmp)
        end_time = time.time()
        print("(%s s)" % str(end_time - start_time))
        print("File size: %s" % str(os.path.getsize(tmp)))

        print("\nLoading hdf5 model...")
        start_time = time.time()
        model = model_persist.load_model_from_h5(tmp)
        end_time = time.time()
        print("(%s s)" % str(end_time - start_time))
    finally:
        os.remove(tmp)


    tmp_skeleton = tempfile.mktemp()
    tmp_weights = tempfile.mktemp()

    try:
        print("\nComparing pickled file size before and after...")
        print(model)
        start_time = time.time()
        model.model_.save_weights(tmp_weights)
        del model.model_
        with open(tmp_skeleton, 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        end_time = time.time()
        print("(%s s)" % str(end_time - start_time))
        print("Model skeleton size: %s" % str(os.path.getsize(tmp_skeleton)))
        print("Model weights size: %s" % str(os.path.getsize(tmp_weights)))
    finally:
        os.remove(tmp_skeleton)
        os.remove(tmp_weights)