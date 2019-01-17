
import numpy as np
import numbers
import os
import pandas as pd
import random
import time
import warnings

from abc import ABCMeta
from itertools import product
from scipy.stats import ttest_ind
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone, is_classifier
from sklearn.exceptions import FitFailedWarning
from sklearn.externals import joblib, six
from sklearn.feature_selection.base import SelectorMixin
from sklearn.feature_selection.univariate_selection import _BaseFilter, _clean_nans
from sklearn.metrics.scorer import _check_multimetric_scoring, check_scoring
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection._validation import _index_param_value, _score
from sklearn.model_selection._split import check_cv
from sklearn.utils import as_float_array, check_X_y, safe_sqr 
from sklearn.utils._joblib import Parallel, delayed, logger
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import indexable, check_is_fitted, _num_samples
from traceback import format_exception_only
from utils import SafeEval


memory = joblib.Memory('./memory_cache')
N_JOBS = int(os.environ.get('GALAXY_SLOTS', 1))


class MemoryFit(object):
    def fit(self, *args, **kwargs):
        fit = memory.cache(super(MemoryFit, self).fit)
        cached_self = fit(*args, **kwargs)
        vars(self).update(vars(cached_self))


class IRAPSCore(object):
    def __init__(self, n_iter=1000, responsive_thres=-1, resistant_thres=0):
        self.n_iter = n_iter
        self.responsive_thres = responsive_thres
        self.resistant_thres = resistant_thres

    def fit(self, X, y):
        """
        X: array-like (n_samples x n_features)
        y: 1-d array-like (n_samples)
        """
        #randomly selection of half samples
        SAMPLE_SIZE = 0.5
        n_samples = X.shape[0]
        pvalues = None
        fold_changes = None

        X, y = check_X_y(X, y, ['csr', 'csc'], multi_output=True)

        for seed in np.arange(self.n_iter):
            n_select = int(n_samples * SAMPLE_SIZE)
            index = random.Random(seed).sample(list(range(n_samples)), n_select)
            X_selected, y_selected = X[index], y[index]
            
            # Standardize (z-score) response.
            y_selected = (y_selected - y_selected.mean())/y_selected.std()
            
            responsive_index = np.arange(n_select)[y_selected <= self.responsive_thres]
            X_selected_responsive = X_selected[responsive_index]

            resistant_index = np.arange(n_select)[y_selected > self.resistant_thres]
            X_selected_resistant = X_selected[resistant_index]

            # p_values
            _, p = ttest_ind(X_selected_responsive, X_selected_resistant, axis=0, equal_var=False)
            if pvalues is None:
                pvalues = p
            else:
                pvalues = np.vstack((pvalues, p))

            # fold_change == mean change?
            # TODO implement other normalization method
            responsive_mean = X_selected_responsive.mean(axis=0)
            resistant_mean = X_selected_resistant.mean(axis=0)
            # mean_change = X_selected_responsive.mean(axis=0) - X_selected_resistant.mean(axis=0)
            mean_change = np.select([responsive_mean >= resistant_mean, responsive_mean < resistant_mean],
                                    [responsive_mean / resistant_mean, -resistant_mean / responsive_mean])
            if fold_changes is None:
                fold_changes = mean_change
            else:
                fold_changes = np.vstack((fold_changes, mean_change))

        self.fold_changes_ = np.asarray(fold_changes)
        self.pvalues_ = np.asarray(pvalues)

        return self


class CachedIRAPSCore(MemoryFit, IRAPSCore):
    pass


class IRAPSClassifier(six.with_metaclass(ABCMeta, _BaseFilter, BaseEstimator, ClassifierMixin)):
    """
    Grid search cross-validation using IRAPS method
    """
    def __init__(self, p_thres=1e-4, fc_thres=0.1,
                 occurance=0.8, n_iter=1000,
                 responsive_thres=-1, resistant_thres=0,
                 clf_thres=0.4):
        self.p_thres = p_thres
        self.fc_thres = fc_thres
        self.occurance = occurance
        self.n_iter = n_iter
        self.responsive_thres = responsive_thres
        self.resistant_thres = resistant_thres
        self.clf_thres = clf_thres

    def fit(self, X, y):
        iraps_handler = IRAPSCore(n_iter=self.n_iter,
                                responsive_thres=self.responsive_thres,
                                resistant_thres=self.resistant_thres)
        iraps_handler.fit(X, y)

        self.fold_changes_ = iraps_handler.fold_changes_
        self.pvalues_ = iraps_handler.pvalues_

        return self

    def _get_support_mask(self):
        """
        return mask of feature selection indices
        """
        check_is_fitted(self, 'pvalues_')

        p_thres = self.p_thres
        fc_thres = self.fc_thres
        occurance = self.occurance

        pvalues = as_float_array(self.pvalues_, copy=True)
        ## why np.nan is here?
        pvalues[np.isnan(pvalues)] = np.finfo(pvalues.dtype).max

        fold_changes = as_float_array(self.fold_changes_, copy=True)
        fold_changes[np.isnan(fold_changes)] = 0.0

        mask_0 = np.zeros(pvalues.shape, dtype=np.int32)
        # mark p_values less than the threashold
        mask_0[pvalues <= p_thres] = 1
        # mark fold_changes only when greater than the threashold
        mask_0[abs(fold_changes) < fc_thres] = 0

        # count the occurance and mask greater than the threshold
        counts = mask_0.sum(axis=0)
        occurance_thres = int(occurance * self.n_iter)
        mask = np.zeros(counts.shape, dtype=bool)
        mask[counts >= occurance_thres] = 1

        # generate signature
        fold_changes[mask_0 == 0] = 0.0
        signature = fold_changes[:, mask].sum(axis=0) / counts[mask]

        self.signature_ = np.asarray(signature)

        return mask

    def get_signature(self):
        if not hasattr(self, 'signature_'):
            self._get_support_mask()

        return self.signature_

    def predict(self, X):
        """
        compute the correlation coefficient with irpas signature and then convert to the bool classes
         True: >=  correlation coefficient threshold
         False: <  correlation coefficient threshold
        """
        predicted = np.zeros(X.shape[0], dtype=bool)
        signature = self.get_signature()
        if signature.size == 0:
            return predicted
        X_transformed = self.transform(X)
        corrcoef = np.array([np.corrcoef(signature, e)[0][1] for e in X_transformed])
        corrcoef[np.isnan(corrcoef)] = np.finfo(np.float32).min
        predicted[corrcoef >= self.clf_thres] = 1
        return predicted

    def decision_function(self, X):
        return self.predict(X)


def _iraps_fit_and_score(estimator, X, y, scorer, train, test, verbose,
                        parameters, fit_params, return_train_score=False,
                        return_parameters=False, return_n_test_samples=False,
                        return_times=False, return_estimator=False,
                        error_score='raise', responsive_thres=-1, resistant_thres=0):
    """Fit with IRAPS_Classifier and compute scores for a given dataset split.

    Parameters
    ----------
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
    error_score : 'raise' | numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised. This parameter
        does not affect the refit step, which will always raise the error.
        Default is np.nan.
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
    fit_params = dict([(k, _index_param_value(X, v, train))
                      for k, v in fit_params.items()])

    train_scores = {}
    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    is_multimetric = not callable(scorer)
    n_scorers = len(scorer.keys()) if is_multimetric else 1

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
        # _score will return dict if is_multimetric is True
        # This is the part that differs from sk-learn
        y_test = apply_label(y_train, y_test, responsive_thres=responsive_thres, resistant_thres=resistant_thres)
        test_scores = _score(estimator, X_test, y_test, scorer, is_multimetric)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            y_train = apply_label(y_train, y_train, responsive_thres=responsive_thres, resistant_thres=resistant_thres)
            train_scores = _score(estimator, X_train, y_train, scorer,
                                  is_multimetric)

    if verbose > 2:
        if is_multimetric:
            for scorer_name, score in test_scores.items():
                msg += ", %s=%s" % (scorer_name, score)
        else:
            msg += ", score=%s" % test_scores
    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = "%s, total=%s" % (msg, logger.short_format_time(total_time))
        print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

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


class IRAPSGirdSearchCV(GridSearchCV):
    
    def __init__(self, param_grid, scoring=None, n_jobs=None,
                 iid=False, refit=True, cv='warn', verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise',
                 return_train_score=False, responsive_thres=-1, resistant_thres=0):
        estimator = IRAPSClassifier()
        super(IRAPSGirdSearchCV, self).__init__(
            estimator, param_grid, scoring=scoring,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv,
            verbose=verbose, pre_dispatch=pre_dispatch, 
            error_score=error_score, return_train_score=return_train_score)
        self.responsive_thres = responsive_thres
        self.resistant_thres = resistant_thres
    
    def fit(self, X, y, groups=None, **fit_params):
        estimator = self.estimator
        responsive_thres = self.responsive_thres
        resistant_thres = self.resistant_thres
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        scorers, self.multimetric_ = _check_multimetric_scoring(
            self.estimator, scoring=self.scoring)

        if self.multimetric_:
            if self.refit is not False and (
                    not isinstance(self.refit, six.string_types) or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers):
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key "
                                 "to refit an estimator with the best "
                                 "parameter setting on the whole data and "
                                 "make the best_* attributes "
                                 "available for that metric. If this is not "
                                 "needed, refit should be set to False "
                                 "explicitly. %r was passed." % self.refit)
            else:
                refit_metric = self.refit
        else:
            refit_metric = 'score'

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                            pre_dispatch=self.pre_dispatch)
        
        fit_and_score_kwargs = dict(scorer=scorers,
                                    fit_params=fit_params,
                                    return_train_score=self.return_train_score,
                                    return_n_test_samples=True,
                                    return_times=True,
                                    return_parameters=False,
                                    error_score=self.error_score,
                                    verbose=self.verbose)
        results_container = [{}]
        with parallel:
            all_candidate_params = []
            all_out = []

            def evaluate_candidates(candidate_params):
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print("Fitting {0} folds for each of {1} candidates,"
                          " totalling {2} fits".format(
                              n_splits, n_candidates, n_candidates * n_splits))

                out = parallel(delayed(_iraps_fit_and_score)(clone(base_estimator),
                                                       X, y,
                                                       train=train, test=test,
                                                       parameters=parameters,
                                                       responsive_thres = responsive_thres,
                                                       resistant_thres = resistant_thres,
                                                       **fit_and_score_kwargs)
                               for parameters, (train, test)
                               in product(candidate_params,
                                          cv.split(X, y, groups)))

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                # XXX: When we drop Python 2 support, we can use nonlocal
                # instead of results_container
                results_container[0] = self._format_results(
                    all_candidate_params, scorers, n_splits, all_out)
                return results_container[0]

            self._run_search(evaluate_candidates)

        results = results_container[0]

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
            self.best_params_ = results["params"][self.best_index_]
            self.best_score_ = results["mean_test_%s" % refit_metric][
                self.best_index_]

        if self.refit:
            self.best_estimator_ = clone(base_estimator).set_params(
                **self.best_params_)
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers['score']

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self


def apply_label(y_train, y_test, responsive_thres=-1, resistant_thres=0):
    """
    Convert continuous values in y_test to z-score True/False/'-' based on mean and std of y_train.
    """
    y_test_zscores = (y_test - y_train.mean()) / y_train.std()

    # in order to make it numpy compatible, use 1 for True, 0 for False, -1 for unclassified
    #y_test_labels = np.full(y_test_zscores.shape, -1)
    #y_test_labels[y_test_zscores <= responsive_thres] = 1
    #y_test_labels[y_test_zscores > resistant_thres] = 0

    y_test_labels = np.zeros(y_test_zscores.shape, dtype=bool)
    y_test_labels[y_test_zscores <= responsive_thres] = 1

    return y_test_labels


def get_search_params(input_json):
    search_params = {}
    safe_eval = SafeEval(load_scipy=True, load_numpy=True)

    for p in params_builder['param_set']:
        search_p = p['search_param_selector']['search_p']
        if search_p.strip() == '':
            continue
        param_type = p['search_param_selector']['selected_param_type']

        lst = search_p.split(":")
        assert (len(lst) == 1), "Colon is not allowed in search parameter input."
        literal = lst[0].strip()
        ev = safe_eval(literal)
        search_params[param_type] = ev

    print(search_params)
    return search_params


if __name__ == "__main__":
    import json
    import os
    import pandas
    import pickle
    from sklearn.exceptions import FitFailedWarning
    import sys
    from utils import get_cv, get_scoring, read_columns

    warnings.simplefilter('ignore')

    input_json_path = sys.argv[1]
    with open(input_json_path, "r") as param_handler:
        params = json.load(param_handler)

    infile1 = sys.argv[2]
    infile2 = sys.argv[3]
    outfile_result = sys.argv[4]
    if len(sys.argv) > 5:
        outfile_estimator = sys.argv[5]
    else:
        outfile_estimator = None

    params_builder = params['search_schemes']['search_params_builder']

    input_type = params["input_options"]["selected_input"]
    if input_type=="tabular":
        header = 'infer' if params["input_options"]["header1"] else None
        column_option = params["input_options"]["column_selector_options_1"]["selected_column_selector_option"]
        if column_option in ["by_index_number", "all_but_by_index_number", "by_header_name", "all_but_by_header_name"]:
            c = params["input_options"]["column_selector_options_1"]["col1"]
        else:
            c = None
        X = read_columns(
                infile1,
                c = c,
                c_option = column_option,
                sep='\t',
                header=header,
                parse_dates=True
        )
    else:
        X = mmread(open(infile1, 'r'))

    header = 'infer' if params["input_options"]["header2"] else None
    column_option = params["input_options"]["column_selector_options_2"]["selected_column_selector_option2"]
    if column_option in ["by_index_number", "all_but_by_index_number", "by_header_name", "all_but_by_header_name"]:
        c = params["input_options"]["column_selector_options_2"]["col2"]
    else:
        c = None
    y = read_columns(
            infile2,
            c = c,
            c_option = column_option,
            sep='\t',
            header=header,
            parse_dates=True
    )
    y=y.ravel()

    options = params["search_schemes"]["options"]
    splitter, groups = get_cv(options.pop('cv_selector'))
    if groups is None:
        options['cv'] = splitter
    elif groups == "":
        options['cv'] = list( splitter.split(X, y, groups=None) )
    else:
        options['cv'] = list( splitter.split(X, y, groups=groups) )
    options['n_jobs'] = N_JOBS
    primary_scoring = options['scoring']['primary_scoring']
    options['scoring'] = get_scoring(options['scoring'])
    if options['error_score']:
        options['error_score'] = 'raise'
    else:
        options['error_score'] = np.NaN
    if options['refit'] and isinstance(options['scoring'], dict):
        options['refit'] = 'primary'
    if 'pre_dispatch' in options and options['pre_dispatch'] == '':
        options['pre_dispatch'] = None

    print(params_builder)
    search_params = get_search_params(params_builder)
    searcher = IRAPSGirdSearchCV(search_params, **options)

    if options['error_score'] == 'raise':
        searcher.fit(X, y)
    else:
        warnings.simplefilter('always', FitFailedWarning)
        with warnings.catch_warnings(record=True) as w:
            try:
                searcher.fit(X, y)
            except ValueError:
                pass
            for warning in w:
                print(repr(warning.message))

    cv_result = pandas.DataFrame(searcher.cv_results_)
    cv_result.rename(inplace=True, columns={"mean_test_primary": "mean_test_"+primary_scoring, "rank_test_primary": "rank_test_"+primary_scoring})
    cv_result.to_csv(path_or_buf=outfile_result, sep='\t', header=True, index=False)

    if outfile_estimator:
        with open(outfile_estimator, "wb") as output_handler:
            pickle.dump(searcher.best_estimator_, output_handler, pickle.HIGHEST_PROTOCOL)
