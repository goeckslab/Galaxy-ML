import argparse
import json
import pandas as pd
import pickle
import warnings
import xgboost
from mlxtend.regressor import StackingCVRegressor, StackingRegressor
from mlxtend.classifier import StackingCVClassifier, StackingClassifier
from sklearn import (cluster, compose, decomposition, ensemble,
                     feature_extraction, feature_selection,
                     gaussian_process, kernel_approximation, metrics,
                     model_selection, naive_bayes, neighbors,
                     pipeline, preprocessing, svm, linear_model,
                     tree, discriminant_analysis)
from sklearn.model_selection._split import check_cv
from feature_selectors import (DyRFE, DyRFECV,
                               MyPipeline, MyimbPipeline)
from iraps_classifier import (IRAPSCore, IRAPSClassifier,
                              BinarizeTargetClassifier,
                              BinarizeTargetRegressor)
from preprocessors import Z_RandomOverSampler
from utils import load_model, get_cv, get_search_params


warnings.filterwarnings('ignore')


def main(base_paths, meta_path, inputs_path, output_obj,
         outfile_params=None):
    """
    Parameter
    ---------
    base_paths : list of str(s)
        List of estimator file paths

    meta_path : str
        File path

    inputs_path : str
        File path for Galaxy parameters

    output_obj : str
        File path for ensemble estimator ouput

    outfile_params : str
        File path for params output
    """
    with open(inputs_path, 'r') as param_handler:
        params = json.load(param_handler)

    base_estimators = []
    for base_file in base_paths.split(','):
        if base_file:
            with open(base_file, 'rb') as handler:
                model = load_model(handler)
                base_estimators.append(model)

    with open(meta_path, 'rb') as f:
        meta_estimator = load_model(f)

    options = params['algo_selection']['options']

    cv_selector = options.pop('cv_selector', None)
    if cv_selector:
        splitter, groups = get_cv(cv_selector)
        options['cv'] = splitter

    # set cv `shuffle` to False
    if 'cv' in options:
        options['shuffle'] = False

    if params['algo_selection']['estimator_type'] == 'StackingCVClassifier':
        ensemble_estimator = StackingCVClassifier(
            classifiers=base_estimators,
            meta_classifier=meta_estimator,
            **options)

    elif params['algo_selection']['estimator_type'] == 'StackingClassifier':
        ensemble_estimator = StackingClassifier(
            classifiers=base_estimators,
            meta_classifier=meta_estimator,
            **options)

    elif params['algo_selection']['estimator_type'] == 'StackingCVRegressor':
        ensemble_estimator = StackingCVRegressor(
            regressors=base_estimators,
            meta_regressor=meta_estimator,
            **options)

    else:
        ensemble_estimator = StackingRegressor(
            regressors=base_estimators,
            meta_regressor=meta_estimator,
            **options)

    print(ensemble_estimator)
    with open(output_obj, 'wb') as out_handler:
        pickle.dump(ensemble_estimator, out_handler, pickle.HIGHEST_PROTOCOL)

    if params['get_params'] and outfile_params:
        results = get_search_params(ensemble_estimator)
        df = pd.DataFrame(results, columns=['', 'Parameter', 'Value'])
        df.to_csv(outfile_params, sep='\t', index=False)


if __name__ == '__main__':
    aparser = argparse.ArgumentParser()
    aparser.add_argument("-b", "--bases", dest="bases")
    aparser.add_argument("-m", "--meta", dest="meta")
    aparser.add_argument("-i", "--inputs", dest="inputs")
    aparser.add_argument("-o", "--outfile", dest="outfile")
    aparser.add_argument("-p", "--outfile_params", dest="outfile_params")
    args = aparser.parse_args()

    main(args.bases, args.meta, args.inputs, args.outfile,
         outfile_params=args.outfile_params)
