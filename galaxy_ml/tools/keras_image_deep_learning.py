import argparse
import json
import numpy as np
import pandas as pd
import warnings

from itertools import chain
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.utils import indexable, _safe_indexing
from galaxy_ml.model_validations import train_test_split
from galaxy_ml.keras_galaxy_models import (_predict_generator,
                                           KerasGBatchClassifier)
from galaxy_ml.preprocessors import ImageDataFrameBatchGenerator
from galaxy_ml.model_persist import load_model_from_h5, dump_model_to_h5
from galaxy_ml.utils import (SafeEval, clean_params, gen_compute_scores,
                             get_scoring, read_columns)


WORKING_DIR = __import__('os').getcwd()
IMAGES_DIR = __import__('os').path.join(WORKING_DIR, 'images')
NON_SEARCHABLE = ('n_jobs', 'pre_dispatch', 'memory', '_path', '_dir',
                  'nthread', 'callbacks')
ALLOWED_CALLBACKS = ('EarlyStopping', 'TerminateOnNaN', 'ReduceLROnPlateau',
                     'CSVLogger', 'None')


def _eval_swap_params(params_builder):
    swap_params = {}

    for p in params_builder['param_set']:
        swap_value = p['sp_value'].strip()
        if swap_value == '':
            continue

        param_name = p['sp_name']
        if param_name.lower().endswith(NON_SEARCHABLE):
            warnings.warn("Warning: `%s` is not eligible for search and was "
                          "omitted!" % param_name)
            continue

        if not swap_value.startswith(':'):
            safe_eval = SafeEval(load_scipy=True, load_numpy=True)
            ev = safe_eval(swap_value)
        else:
            # Have `:` before search list, asks for estimator evaluatio
            safe_eval_es = SafeEval(load_estimators=True)
            swap_value = swap_value[1:].strip()
            # TODO maybe add regular express check
            ev = safe_eval_es(swap_value)

        swap_params[param_name] = ev

    return swap_params


def train_test_split_none(*arrays, **kwargs):
    """extend train_test_split to take None arrays
    and support split by group names.
    """
    nones = []
    new_arrays = []
    for idx, arr in enumerate(arrays):
        if arr is None:
            nones.append(idx)
        else:
            new_arrays.append(arr)

    if kwargs['shuffle'] == 'None':
        kwargs['shuffle'] = None

    group_names = kwargs.pop('group_names', None)

    if group_names is not None and group_names.strip():
        group_names = [name.strip() for name in
                       group_names.split(',')]
        new_arrays = indexable(*new_arrays)
        groups = kwargs['labels']
        n_samples = new_arrays[0].shape[0]
        index_arr = np.arange(n_samples)
        test = index_arr[np.isin(groups, group_names)]
        train = index_arr[~np.isin(groups, group_names)]
        rval = list(chain.from_iterable(
            (_safe_indexing(a, train),
             _safe_indexing(a, test)) for a in new_arrays))
    else:
        rval = train_test_split(*new_arrays, **kwargs)

    for pos in nones:
        rval[pos * 2: 2] = [None, None]

    return rval


def _handle_image_generator_params(params, image_df):
    """reconstruct generator kwargs from tool inputs
    """
    safe_eval = SafeEval()

    options = {}

    headers = image_df.columns
    options['x_col'] = headers[params['x_col'][0] - 1]

    y_col = list(map(lambda x: x-1, params['y_col']))
    if len(y_col) == 1:
        options['y_col'] = headers[y_col[0]]
    else:
        options['y_col'] = list(headers[y_col])

    weight_col = params['weight_col'][0]
    if weight_col is None:
        options['weight_col'] = None
    else:
        options['weight_col'] = headers[weight_col - 1]

    other_options = params['options']
    for k, v in other_options.items():
        if k == 'target_size' or k.endswith('_range'):
            v = v.strip()
            if not v:
                other_options[k] = None
            else:
                other_options[k] = safe_eval(v)
        if k == 'classes':
            v = v.strip()
            if not v:
                other_options[k] = None
            else:
                other_options[k] = [x.strip() for x in v.split(',')]

    options.update(other_options)

    return options


def _evaluate_keras_and_sklearn_scores(estimator, data_generator, X,
                                       y=None, sk_scoring=None,
                                       steps=None, batch_size=32,
                                       return_predictions=False):
    """output scores for bother keras and sklearn metrics

    Parameters
    -----------
    estimator : object
        Fitted `galaxy_ml.keras_galaxy_models.KerasGBatchClassifier`.
    data_generator : object
        From `galaxy_ml.preprocessors.ImageDataFrameBatchGenerator`.
    X : 2-D array
        Contains indecies of images that need to be evaluated.
    y : None
        Target value.
    sk_scoring : dict
        Galaxy tool input parameters.
    steps : integer or None
        Evaluation/prediction steps before stop.
    batch_size : integer
        Number of samples in a batch
    return_predictions : bool, default is False
        Whether to return predictions and true labels.
    """
    scores = {}

    generator = data_generator.flow(X, y=y, batch_size=batch_size)
    # keras metrics evaluation
    # handle scorer, convert to scorer dict
    generator.reset()
    score_results = estimator.model_.evaluate_generator(generator,
                                                        steps=steps)
    metrics_names = estimator.model_.metrics_names
    if not isinstance(metrics_names, list):
        scores[metrics_names] = score_results
    else:
        scores = dict(zip(metrics_names, score_results))

    if sk_scoring['primary_scoring'] == 'default' and\
            not return_predictions:
        return scores

    generator.reset()
    predictions, y_true = _predict_generator(estimator.model_,
                                             generator,
                                             steps=steps)

    # for sklearn metrics
    if sk_scoring['primary_scoring'] != 'default':
        scorer = get_scoring(sk_scoring)
        if not isinstance(scorer, (dict, list)):
            scorer = [sk_scoring['primary_scoring']]
        scorer = _check_multimetric_scoring(estimator, scoring=scorer)
        sk_scores = gen_compute_scores(y_true, predictions, scorer)
        scores.update(sk_scores)

    if return_predictions:
        return scores, predictions, y_true
    else:
        return scores, None, None


def main(inputs, infile_estimator, infile_dataframe,
         outfile_result, outfile_object=None, outfile_y_true=None,
         outfile_y_preds=None, groups=None):
    """
    Parameter
    ---------
    inputs : str
        File path to galaxy tool parameter.

    infile_estimator : str
        File path to estimator.

    infile_dataframe : str
        File path to tabular dataset containing image information.

    outfile_result : str
        File path to save the results, either cv_results or test result.

    outfile_object : str, optional
        File path to save searchCV object.

    outfile_y_true : str, optional
        File path to target values for prediction.

    outfile_y_preds : str, optional
        File path to save predictions.

    groups : str
        File path to dataset containing groups labels.
    """
    warnings.simplefilter('ignore')

    with open(inputs, 'r') as param_handler:
        params = json.load(param_handler)

    #  load estimator
    estimator = load_model_from_h5(infile_estimator)

    estimator = clean_params(estimator)

    if not isinstance(estimator, KerasGBatchClassifier):
        raise ValueError(
            "Only `galaxy_ml.keras_galaxy_models.KerasGBatchClassifier` "
            "is supported!")

    # read DataFrame for images
    data_frame = pd.read_csv(infile_dataframe, sep='\t', header='infer')

    kwargs = _handle_image_generator_params(params['input_options'],
                                            data_frame)

    # build data generator
    image_generator = ImageDataFrameBatchGenerator(dataframe=data_frame,
                                                   directory=IMAGES_DIR,
                                                   **kwargs)
    estimator.set_params(data_batch_generator=image_generator)
    steps = estimator.prediction_steps
    batch_size = estimator.batch_size

    # Get X and y
    X = np.arange(data_frame.shape[0])[:, np.newaxis]

    if isinstance(kwargs['y_col'], list):
        y = None
    else:
        y = data_frame[kwargs['y_col']].ravel()

    # load groups
    if groups:
        groups_selector = (params['experiment_schemes']['test_split']
                                 ['split_algos']).pop('groups_selector')

        header = 'infer' if groups_selector['header_g'] else None
        column_option = \
            (groups_selector['column_selector_options_g']
                            ['selected_column_selector_option_g'])
        if column_option in ['by_index_number', 'all_but_by_index_number',
                             'by_header_name', 'all_but_by_header_name']:
            c = groups_selector['column_selector_options_g']['col_g']
        else:
            c = None

        if groups == infile_dataframe:
            groups = data_frame

        groups = read_columns(
                groups,
                c=c,
                c_option=column_option,
                sep='\t',
                header=header,
                parse_dates=True)
        groups = groups.ravel()

    exp_scheme = params['experiment_schemes']['selected_exp_scheme']

    # Model Predictions
    if exp_scheme == 'model_predict':
        steps = params['experiment_schemes']['pred_steps']

        generator = image_generator.flow(X, y=None, batch_size=batch_size)

        predictions = estimator.model_.predict_generator(generator,
                                                         steps=steps)
        try:
            pd.DataFrame(predictions).astype(np.float32).to_csv(
                outfile_result, sep='\t', index=False,
                float_format='%g', chunksize=10000)
        except Exception as e:
            print("Error in saving predictions: %s" % e)
        return 0

    # Model Evaluation
    if exp_scheme == 'model_eval':
        # compile model
        estimator.model_.compile(loss=estimator.loss,
                                 optimizer=estimator._optimizer,
                                 metrics=estimator.metrics)
        steps = params['experiment_schemes']['eval_steps']
        sk_scoring = params['experiment_schemes']['metrics']['scoring']

        scores, predictions, y_true = _evaluate_keras_and_sklearn_scores(
            estimator, image_generator, X, sk_scoring=sk_scoring,
            steps=steps, batch_size=batch_size,
            return_predictions=bool(outfile_y_true))

    # for other two modes, train/val and train/val/test
    else:
        # swap hyperparameter
        swapping = params['experiment_schemes']['hyperparams_swapping']
        swap_params = _eval_swap_params(swapping)
        estimator.set_params(**swap_params)

        # handle test (first) split
        test_split_options = (params['experiment_schemes']
                                    ['test_split']['split_algos'])

        if test_split_options['shuffle'] == 'group':
            test_split_options['labels'] = groups
        if test_split_options['shuffle'] == 'stratified':
            if y is not None:
                test_split_options['labels'] = y
            else:
                raise ValueError("Stratified shuffle split is not "
                                 "applicable on empty target values or "
                                 "multiple output targets!")

        X_train, X_test, y_train, y_test, groups_train, groups_test = \
            train_test_split_none(X, y, groups, **test_split_options)

        # handle validation (second) split
        if exp_scheme == 'train_val_test':
            val_split_options = (params['experiment_schemes']
                                       ['val_split']['split_algos'])

            if val_split_options['shuffle'] == 'group':
                val_split_options['labels'] = groups_train
            if val_split_options['shuffle'] == 'stratified':
                if y_train is not None:
                    val_split_options['labels'] = y_train
                else:
                    raise ValueError("Stratified shuffle split is not "
                                     "applicable on empty target values!")

            X_train, X_val, y_train, y_val, groups_train, groups_val = \
                train_test_split_none(X_train, y_train, groups_train,
                                      **val_split_options)

            # In image data generator, `y_val` must be None
            # labels will be retrived in generator.
            estimator.fit(X_train, y_train, validation_data=(X_val, ))

        else:
            estimator.fit(X_train, y_train, validation_data=(X_test, ))

        data_generator = estimator.data_generator_
        sk_scoring = params['experiment_schemes']['metrics']['scoring']
        steps = estimator.prediction_steps

        scores, predictions, y_true = _evaluate_keras_and_sklearn_scores(
            estimator, data_generator, X_test, sk_scoring=sk_scoring,
            steps=steps, batch_size=batch_size,
            return_predictions=bool(outfile_y_true))

    # handle output
    if outfile_y_true:
        try:
            pd.DataFrame(y_true).to_csv(outfile_y_true, sep='\t',
                                        index=False)
            pd.DataFrame(predictions).astype(np.float32).to_csv(
                outfile_y_preds, sep='\t', index=False,
                float_format='%g', chunksize=10000)
        except Exception as e:
            print("Error in saving predictions: %s" % e)

    # handle output
    for name, score in scores.items():
        scores[name] = [score]
    df = pd.DataFrame(scores)
    df = df[sorted(df.columns)]
    df.to_csv(path_or_buf=outfile_result, sep='\t', header=True,
              index=False)

    if outfile_object:
        dump_model_to_h5(estimator, outfile_object)


if __name__ == '__main__':
    aparser = argparse.ArgumentParser()
    aparser.add_argument("-i", "--inputs", dest="inputs", required=True)
    aparser.add_argument("-e", "--estimator", dest="infile_estimator")
    # aparser.add_argument("-X", "--infile_images", dest="infile_images")
    aparser.add_argument("-y", "--infile_dataframe", dest="infile_dataframe")
    aparser.add_argument("-O", "--outfile_result", dest="outfile_result")
    aparser.add_argument("-o", "--outfile_object", dest="outfile_object")
    aparser.add_argument("-l", "--outfile_y_true", dest="outfile_y_true")
    aparser.add_argument("-p", "--outfile_y_preds", dest="outfile_y_preds")
    aparser.add_argument("-g", "--groups", dest="groups")
    args = aparser.parse_args()

    main(args.inputs, args.infile_estimator,
         args.infile_dataframe, args.outfile_result,
         outfile_object=args.outfile_object,
         outfile_y_true=args.outfile_y_true,
         outfile_y_preds=args.outfile_y_preds,
         groups=args.groups)
