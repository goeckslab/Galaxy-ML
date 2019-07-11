import json
import pickle
import time
import sys
from galaxy_ml import model_persist
from sklearn import (
    cluster, decomposition, ensemble, feature_extraction,
    feature_selection, gaussian_process, kernel_approximation,
    kernel_ridge, linear_model, metrics, model_selection,
    naive_bayes, neighbors, pipeline, preprocessing, svm,
    tree, discriminant_analysis)


test_model = './tools/test-data/gbr_model01_py3'
result_json = './tools/test-data/gbr_model01_py3.json'


def test_jpikle_dumpc():
    with open(test_model, 'rb') as f:
        model = pickle.load(f)

    got = model_persist.dumpc(model)

    with open(result_json, 'r') as f:
        expect = json.load(f)

    assert got == expect, got


if __name__ == '__main__':
    if len(sys.argv) > 1:
        test_model = sys.argv[1]
    else:
        test_model = test_model

    print("Loading pickled test model...")
    start_time = time.time()
    with open(test_model, 'rb') as f:
        model = pickle.load(f)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))

    pickle0_file = test_model + '.pickle0'
    print("\nDumping model using pickle protocol-0...")
    start_time = time.time()
    with open(pickle0_file, 'wb') as f:
        pickle.dump(model, f)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))

    print("\nLoading model using pickle protocol-0...")
    start_time = time.time()
    with open(pickle0_file, 'rb') as f:
        pickle_model = pickle.load(f)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))

    print("\nDumping object to dict...")
    start_time = time.time()
    model_dict = model_persist.dumpc(model)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))

    # pprint.pprint(model_dict)

    json_file = test_model + '.json'
    print("\nDumping dict data to JSON file...")
    start_time = time.time()
    with open(json_file, 'w') as f:
        json.dump(model_dict, f, sort_keys=True)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))

    print("\nLoading data from JSON file...")
    start_time = time.time()
    with open(json_file, 'r') as f:
        new_dict = json.load(f)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))

    print("\nRe-build the model object...")
    start_time = time.time()
    re_model = model_persist.loadc(new_dict)
    end_time = time.time()
    print("(%s s)" % str(end_time - start_time))
    print("%r" % re_model)
