"""
class
-----
OrderedKFold
RepeatedOrderedKold


function
--------
train_test_split
"""

import numpy as np
import warnings

from itertools import chain
from math import ceil, floor
from sklearn.model_selection import (GroupShuffleSplit, ShuffleSplit,
                                     StratifiedShuffleSplit)
from sklearn.model_selection._split import _BaseKFold, _RepeatedSplits
from sklearn.utils import check_random_state, indexable, safe_indexing
from sklearn.utils.validation import _num_samples, check_array


__all__ = ('train_test_split', 'OrderedKFold', 'RepeatedOrderedKFold',
           '_fit_and_score')


def _validate_shuffle_split(n_samples, test_size, train_size,
                            default_test_size=None):
    """
    Validation helper to check if the test/test sizes are meaningful wrt to the
    size of the data (n_samples)
    """
    if test_size is None and train_size is None:
        test_size = default_test_size

    test_size_type = np.asarray(test_size).dtype.kind
    train_size_type = np.asarray(train_size).dtype.kind

    if (test_size_type == 'i' and (test_size >= n_samples or test_size <= 0)
       or test_size_type == 'f' and (test_size <= 0 or test_size >= 1)):
        raise ValueError('test_size={0} should be either positive and smaller'
                         ' than the number of samples {1} or a float in the '
                         '(0, 1) range'.format(test_size, n_samples))

    if (train_size_type == 'i' and (train_size >= n_samples or train_size <= 0)
       or train_size_type == 'f' and (train_size <= 0 or train_size >= 1)):
        raise ValueError('train_size={0} should be either positive and smaller'
                         ' than the number of samples {1} or a float in the '
                         '(0, 1) range'.format(train_size, n_samples))

    if train_size is not None and train_size_type not in ('i', 'f'):
        raise ValueError("Invalid value for train_size: {}".format(train_size))
    if test_size is not None and test_size_type not in ('i', 'f'):
        raise ValueError("Invalid value for test_size: {}".format(test_size))

    if (train_size_type == 'f' and test_size_type == 'f' and
            train_size + test_size > 1):
        raise ValueError(
            'The sum of test_size and train_size = {}, should be in the (0, 1)'
            ' range. Reduce test_size and/or train_size.'
            .format(train_size + test_size))

    if test_size_type == 'f':
        n_test = ceil(test_size * n_samples)
    elif test_size_type == 'i':
        n_test = float(test_size)

    if train_size_type == 'f':
        n_train = floor(train_size * n_samples)
    elif train_size_type == 'i':
        n_train = float(train_size)

    if train_size is None:
        n_train = n_samples - n_test
    elif test_size is None:
        n_test = n_samples - n_train

    if n_train + n_test > n_samples:
        raise ValueError('The sum of train_size and test_size = %d, '
                         'should be smaller than the number of '
                         'samples %d. Reduce test_size and/or '
                         'train_size.' % (n_train + n_test, n_samples))

    n_train, n_test = int(n_train), int(n_test)

    if n_train == 0:
        raise ValueError(
            'With n_samples={}, test_size={} and train_size={}, the '
            'resulting train set will be empty. Adjust any of the '
            'aforementioned parameters.'.format(n_samples, test_size,
                                                train_size)
        )

    return n_train, n_test


def train_test_split(*arrays, **options):
    """Extend sklearn.model_selection.train_test_slit to have group split.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    test_size : float, int or None, optional (default=None)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.
    train_size : float, int, or None, (default=None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    shuffle : None or str (default='simple')
        How to shuffle the data before splitting.
        None, no shuffle.
        For str, one of 'simple', 'stratified' and 'group', corresponding to
        `ShuffleSplit`, `StratifiedShuffleSplit` and `GroupShuffleSplit`,
        respectively.
    labels : array-like or None (default=None)
        Ignored if shuffle is None or 'simple'.
        When shuffle='stratified', this array is used as class labels.
        When shuffle='group', this array is used as groups.

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.

    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    test_size = options.pop('test_size', None)
    train_size = options.pop('train_size', None)
    random_state = options.pop('random_state', None)
    shuffle = options.pop('shuffle', 'simple')
    labels = options.pop('labels', None)

    if options:
        raise TypeError("Invalid parameters passed: %s" % str(options))

    arrays = indexable(*arrays)

    n_samples = _num_samples(arrays[0])
    if shuffle == 'group':
        if labels is None:
            raise ValueError("When shuffle='group', "
                             "labels should not be None!")
        labels = check_array(labels, ensure_2d=False, dtype=None)
        uniques = np.unique(labels)
        n_samples = uniques.size

    n_train, n_test = _validate_shuffle_split(n_samples, test_size, train_size,
                                              default_test_size=0.25)

    shuffle_options = dict(test_size=n_test,
                           train_size=n_train,
                           random_state=random_state)

    if shuffle is None:
        if labels is not None:
            warnings.warn("The `labels` is ignored for "
                          "shuffle being None!")

        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)

    elif shuffle == 'simple':
        if labels is not None:
            warnings.warn("The `labels` is not needed and therefore "
                          "ignored for ShuffleSplit, as shuffle='simple'!")

        cv = ShuffleSplit(**shuffle_options)
        train, test = next(cv.split(X=arrays[0], y=None))

    elif shuffle == 'stratified':
        cv = StratifiedShuffleSplit(**shuffle_options)
        train, test = next(cv.split(X=arrays[0], y=labels))

    elif shuffle == 'group':
        cv = GroupShuffleSplit(**shuffle_options)
        train, test = next(cv.split(X=arrays[0], y=None, groups=labels))

    else:
        raise ValueError("The argument `shuffle` only supports None, "
                         "'simple', 'stratified' and 'group', but got `%s`!"
                         % shuffle)

    return list(chain.from_iterable((safe_indexing(a, train),
                                    safe_indexing(a, test)) for a in arrays))


class OrderedKFold(_BaseKFold):
    """
    Split into K fold based on ordered target value

    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.
    shuffle : bool
    random_state : None or int
    """

    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        super(OrderedKFold, self).__init__(n_splits, shuffle, random_state)

    def _iter_test_indices(self, X, y, groups=None):
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        y = np.asarray(y)
        sorted_index = np.argsort(y)
        if self.shuffle:
            current = 0
            rng = check_random_state(self.random_state)
            for i in range(n_samples // int(n_splits)):
                start, stop = current, current + n_splits
                rng.shuffle(sorted_index[start:stop])
                current = stop
            rng.shuffle(sorted_index[current:])

        for i in range(n_splits):
            yield sorted_index[i:n_samples:n_splits]


class RepeatedOrderedKFold(_RepeatedSplits):
    """ Repeated OrderedKFold runs mutiple times with different randomization.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    n_repeats : int, default=5
        Number of times cross-validator to be repeated.
    random_state: int, RandomState instance or None. Optional
    """
    def __init__(self, n_splits=5, n_repeats=5, random_state=None):
        super(RepeatedOrderedKFold, self).__init__(
            OrderedKFold, n_repeats, random_state, n_splits=n_splits)


#####################################################################
#####################################################################

import numbers
import time
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection._validation import _index_param_value
from sklearn.model_selection._validation import _score
from sklearn.utils.metaestimators import _safe_split
# from sklearn.utils import _message_with_time
from traceback import format_exception_only


def _fit_and_score(estimator, X, y, scorer, train, test, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=False, return_estimator=False,
                   error_score=np.nan):
    """override the sklearn.model_selection._validation._fit_and_score

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape at least 2D
        The data to fit.
    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.
    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.
        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.
        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.
    train : array-like, shape (n_train_samples,)
        Indices of training samples.
    test : array-like, shape (n_test_samples,)
        Indices of test samples.
    verbose : integer
        The verbosity level.
    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised. This parameter
        does not affect the refit step, which will always raise the error.
    parameters : dict or None
        Parameters to be set on the estimator.
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.
    return_train_score : boolean, optional, default: False
        Compute and return score on training set.
    return_parameters : boolean, optional, default: False
        Return parameters that has been used for the estimator.
    return_n_test_samples : boolean, optional, default: False
        Whether to return the ``n_test_samples``
    return_times : boolean, optional, default: False
        Whether to return the fit/score times.
    return_estimator : boolean, optional, default: False
        Whether to return the fitted estimator.
    Returns
    -------
    train_scores : dict of scorer name -> float, optional
        Score on training set (for all the scorers),
        returned only if `return_train_score` is `True`.
    test_scores : dict of scorer name -> float, optional
        Score on testing set (for all the scorers).
    n_test_samples : int
        Number of test samples.
    fit_time : float
        Time spent for fitting in seconds.
    score_time : float
        Time spent for scoring in seconds.
    parameters : dict or None, optional
        The parameters that have been evaluated.
    estimator : estimator object
        The fitted estimator
    """
    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                          for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = {k: _index_param_value(X, v, train)
                  for k, v in fit_params.items()}

    train_scores = {}
    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    is_multimetric = not callable(scorer)
    n_scorers = len(scorer.keys()) if is_multimetric else 1

    ##########################################################
    # Changes on sklearn.model_selection._search
    for param in estimator.get_params().keys():
        # suppose work for both pipeline or single
        # keras model
        if param.endswith('validation_data'):
            fit_params.update(
                {param: (X_test, y_test)})
            break
    # Changes end
    ##########################################################
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            if is_multimetric:
                test_scores = dict(zip(scorer.keys(),
                                   [error_score, ] * n_scorers))
                if return_train_score:
                    train_scores = dict(zip(scorer.keys(),
                                        [error_score, ] * n_scorers))
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            warnings.warn("Estimator fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%s" %
                          (error_score, format_exception_only(type(e), e)[0]),
                          FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.time() - start_time

        #######################################################################
        #######################################################################
        if estimator.__class__.__name__ == 'KerasGBatchClassifier' \
                and hasattr(estimator.data_batch_generator, 'target_path'):
            test_scores = estimator.evaluate(X_test, y_test,
                                             scorer, is_multimetric)
        else:
            # _score will return dict if is_multimetric is True
            test_scores = _score(estimator, X_test, y_test,
                                 scorer, is_multimetric)
        #######################################################################
        #######################################################################
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(estimator, X_train, y_train, scorer,
                                  is_multimetric)
    if verbose > 2:
        if is_multimetric:
            for scorer_name in sorted(test_scores):
                msg += ", %s=" % scorer_name
                if return_train_score:
                    msg += "(train=%.3f," % train_scores[scorer_name]
                    msg += " test=%.3f)" % test_scores[scorer_name]
                else:
                    msg += "%.3f" % test_scores[scorer_name]
        else:
            msg += ", score="
            msg += ("%.3f" % test_scores if not return_train_score else
                    "(train=%.3f, test=%.3f)" % (train_scores, test_scores))

    if verbose > 1:
        total_time = score_time + fit_time
        print(_message_with_time('CV', msg, total_time))

    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    if return_estimator:
        ret.append(estimator)
    return ret


# `sklearn.utils._message_with_time` is available from v0.21.x.
def _message_with_time(source, message, time):
    """Create one line message for logging purposes
    Parameters
    ----------
    source : str
        String indicating the source or the reference of the message
    message : str
        Short message
    time : int
        Time in seconds
    """
    start_message = "[%s] " % source

    # adapted from joblib.logger.short_format_time without the Windows -.1s
    # adjustment
    if time > 60:
        time_str = "%4.1fmin" % (time / 60)
    else:
        time_str = " %5.1fs" % time
    end_message = " %s, total=%s" % (message, time_str)
    dots_len = (70 - len(start_message) - len(end_message))
    return "%s%s%s" % (start_message, dots_len * '.', end_message)
