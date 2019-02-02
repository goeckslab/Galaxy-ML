import ast
import json
import numpy as np
import os
import pandas
import pickle
import re
import scipy
import sklearn
import sys
import warnings
import xgboost

from asteval import Interpreter, make_symbol_table
from sklearn import (cluster, compose, decomposition, ensemble, feature_extraction,
                    feature_selection, gaussian_process, kernel_approximation, metrics,
                    model_selection, naive_bayes, neighbors, pipeline, preprocessing,
                    svm, linear_model, tree, discriminant_analysis)

from sklearn.base import MetaEstimatorMixin, clone, is_classifier
from sklearn.feature_selection.rfe import _rfe_single_fit, RFE, RFECV
from sklearn.model_selection import check_cv
from sklearn.metrics.scorer import check_scoring
from sklearn.utils import check_X_y, safe_sqr
from sklearn.utils._joblib import Parallel, delayed, effective_n_jobs

try:
    import skrebate
except ModuleNotFoundError:
    pass

try:
    IRAPSClassifier
except NameError:
    try:
        from iraps_classifier import (IRAPSCore, IRAPSClassifier, OrderedKFold, BinarizeTargetClassifier,
                    BinarizeTargetRegressor, binarize_auc_scorer, binarize_average_precision_scorer)
    except ModuleNotFoundError:
        pass

try:
    sk_whitelist
except NameError:
    sk_whitelist = None

try:
    imbPipeline
except NameError:
    imbPipeline = None


N_JOBS = int(os.environ.get('GALAXY_SLOTS', 1))


class MyPipeline(pipeline.Pipeline):
    def fit(self, X, y=None, **fit_params):
        super(MyPipeline, self).fit(X, y, **fit_params)
        estimator = self.steps[-1][-1]
        if hasattr(estimator, 'coef_'):
            coefs = estimator.coef_
        else:
            coefs = getattr(estimator, 'feature_importances_', None)
        if coefs is None:
            raise RuntimeError('The estimator in the pipeline does not expose '
                                '"coef_" or "feature_importances_" '
                                'attributes')
        self.feature_importances_ = coefs
        return self


if imbPipeline:
    class MyimbPipeline(imbPipeline):
        def fit(self, X, y=None, **fit_params):
            super(MyimbPipeline, self).fit(X, y, **fit_params)
            estimator = self.steps[-1][-1]
            if hasattr(estimator, 'coef_'):
                coefs = estimator.coef_
            else:
                coefs = getattr(estimator, 'feature_importances_', None)
            if coefs is None:
                raise RuntimeError('The estimator in the pipeline does not expose '
                                    '"coef_" or "feature_importances_" '
                                    'attributes')
            self.feature_importances_ = coefs
            return self


def check_feature_importances(estimator):
    """
    For pipeline object which has no feature_importances_ property,
    this function returns the same comfigured pipeline object with
    attached the last estimator's feature_importances_.
    """
    if estimator.__class__.__module__ == 'sklearn.pipeline':
        pipeline_steps = estimator.get_params()['steps']
        estimator = MyPipeline(pipeline_steps)
    elif estimator.__class__.__module__ == 'imblearn.pipeline':
        pipeline_steps = estimator.get_params()['steps']
        estimator = MyimbPipeline(pipeline_steps)
    else:
        return estimator


class DyRFE(RFE):
    """
    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.
    n_features_to_select : int or None (default=None)
        The number of features to select. If `None`, half of the features
        are selected.
    step : int, float or list, optional (default=1)
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of features to remove at each iteration.
        If list, a series of steps of features to remove at each iteration.
        Iterations stops when steps finish
    verbose : int, (default=0)
        Controls verbosity of output.

    """
    def __init__(self, estimator, n_features_to_select=None, step=1,
                 verbose=0):
        super(DyRFE, self).__init__(estimator, n_features_to_select, step, verbose)


    def _fit(self, X, y, step_score=None):

        if type(self.step) is not list:
            return super(DyRFE, self)._fit(X, y, step_score)

        # dynamic step
        X, y = check_X_y(X, y, "csc")
        # Initialization
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        else:
            n_features_to_select = self.n_features_to_select

        step = []
        for s in self.step:
            if 0.0 < s < 1.0:
                step.append( int(max(1, s * n_features)) )
            else:
                step.append( int(s) )
            if s <= 0:
                raise ValueError("Step must be >0")

        support_ = np.ones(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int)

        if step_score:
            self.scores_ = []

        step_i = 0
        # Elimination
        while np.sum(support_) > n_features_to_select and step_i < len(step):

            # if last step is 1, will keep loop
            if step_i == len(step) - 1 and step[step_i] != 0:
                step.append(step[step_i])

            # Remaining features
            features = np.arange(n_features)[support_]

            # Rank the remaining features
            estimator = clone(self.estimator)
            if self.verbose > 0:
                print("Fitting estimator with %d features." % np.sum(support_))

            estimator.fit(X[:, features], y)

            # Get coefs
            if hasattr(estimator, 'coef_'):
                coefs = estimator.coef_
            else:
                coefs = getattr(estimator, 'feature_importances_', None)
            if coefs is None:
                raise RuntimeError('The classifier does not expose '
                                   '"coef_" or "feature_importances_" '
                                   'attributes')

            # Get ranks
            if coefs.ndim > 1:
                ranks = np.argsort(safe_sqr(coefs).sum(axis=0))
            else:
                ranks = np.argsort(safe_sqr(coefs))

            # for sparse case ranks is matrix
            ranks = np.ravel(ranks)

            # Eliminate the worse features
            threshold = min(step[step_i], np.sum(support_) - n_features_to_select)

            # Compute step score on the previous selection iteration
            # because 'estimator' must use features
            # that have not been eliminated yet
            if step_score:
                self.scores_.append(step_score(estimator, features))
            support_[features[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] = 1

            step_i += 1

        # Set final attributes
        features = np.arange(n_features)[support_]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, features], y)

        # Compute step score when only n_features_to_select features left
        if step_score:
            self.scores_.append(step_score(self.estimator_, features))
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self


class DyRFECV(RFECV, MetaEstimatorMixin):
    """
        Parameters
    ----------
    estimator : object
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.
    step : int or float, optional (default=1)
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of features to remove at each iteration.
        If list, a series of step to remove at each iteration. iteration stopes
        when finishing all steps
        Note that the last iteration may remove fewer than ``step`` features in
        order to reach ``min_features_to_select``.
    min_features_to_select : int, (default=1)
        The minimum number of features to be selected. This number of features
        will always be scored, even if the difference between the original
        feature count and ``min_features_to_select`` isn't divisible by
        ``step``.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. If the
        estimator is a classifier or if ``y`` is neither binary nor multiclass,
        :class:`sklearn.model_selection.KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        .. versionchanged:: 0.20
            ``cv`` default value of None will change from 3-fold to 5-fold
            in v0.22.
    scoring : string, callable or None, optional, (default=None)
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    verbose : int, (default=0)
        Controls verbosity of output.
    n_jobs : int or None, optional (default=None)
        Number of cores to run in parallel while fitting across folds.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    """
    def __init__(self, estimator, step=1, min_features_to_select=1, cv='warn',
                 scoring=None, verbose=0, n_jobs=None):
        super(DyRFECV, self).__init__(estimator, step=step,
                        min_features_to_select=min_features_to_select,
                        cv=cv, scoring=scoring, verbose=verbose,
                        n_jobs=n_jobs)

    def fit(self, X, y, groups=None):
        """Fit the RFE model and automatically tune the number of selected
           features.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the total number of features.
        y : array-like, shape = [n_samples]
            Target values (integers for classification, real numbers for
            regression).
        groups : array-like, shape = [n_samples], optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        """
        if type(self.step) is not list:
            return super(DyRFECV, self).fit(X, y, groups)

        X, y = check_X_y(X, y, "csr")

        # Initialization
        cv = check_cv(self.cv, y, is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        n_features = X.shape[1]

        step = []
        for s in self.step:
            if 0.0 < s < 1.0:
                step.append( int(max(1, s * n_features)) )
            else:
                step.append( int(s) )
            if s <= 0:
                raise ValueError("Step must be >0")

        # Build an RFE object, which will evaluate and score each possible
        # feature count, down to self.min_features_to_select
        rfe = DyRFE(estimator=self.estimator,
                  n_features_to_select=self.min_features_to_select,
                  step=self.step, verbose=self.verbose)

        # Determine the number of subsets of features by fitting across
        # the train folds and choosing the "features_to_select" parameter
        # that gives the least averaged error across all folds.

        # Note that joblib raises a non-picklable error for bound methods
        # even if n_jobs is set to 1 with the default multiprocessing
        # backend.
        # This branching is done so that to
        # make sure that user code that sets n_jobs to 1
        # and provides bound methods as scorers is not broken with the
        # addition of n_jobs parameter in version 0.18.

        if effective_n_jobs(self.n_jobs) == 1:
            parallel, func = list, _rfe_single_fit
        else:
            parallel = Parallel(n_jobs=self.n_jobs)
            func = delayed(_rfe_single_fit)

        scores = parallel(
            func(rfe, self.estimator, X, y, train, test, scorer)
            for train, test in cv.split(X, y, groups))

        scores = np.sum(scores, axis=0)
        scores_rev = scores[::-1]
        argmax_idx = len(scores) - np.argmax(scores_rev) - 1
        n_features_to_select = max(
            n_features - sum(step[:argmax_idx]),
            self.min_features_to_select)

        # Re-execute an elimination with best_k over the whole set
        rfe = DyRFE(estimator=self.estimator,
                  n_features_to_select=n_features_to_select, step=self.step,
                  verbose=self.verbose)

        rfe.fit(X, y)

        # Set final attributes
        self.support_ = rfe.support_
        self.n_features_ = rfe.n_features_
        self.ranking_ = rfe.ranking_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self.transform(X), y)

        # Fixing a normalization error, n is equal to get_n_splits(X, y) - 1
        # here, the scores are normalized by get_n_splits(X, y)
        self.grid_scores_ = scores[::-1] / cv.get_n_splits(X, y, groups)
        return self


class SafePickler(pickle.Unpickler):
    """
    Used to safely deserialize scikit-learn model objects serialized by cPickle.dump
    Usage:
        eg.: SafePickler.load(pickled_file_object)
    """
    def find_class(self, module, name):

        customer_classes = {
            'IRAPSCore': IRAPSCore,
            'IRAPSClassifier': IRAPSClassifier,
            'BinarizeTargetClassifier': BinarizeTargetClassifier,
            'BinarizeTargetRegressor': BinarizeTargetRegressor,
            'OrderedKFold': OrderedKFold
        }

        if module == '__main__':
            return customer_classes[name]
        # sk_whitelist could be read from tool
        global sk_whitelist
        if not sk_whitelist:
            whitelist_file = os.path.join(os.path.dirname(__file__), 'sk_whitelist.json')
            with open(whitelist_file, 'r') as f:
                sk_whitelist = json.load(f)

        bad_names = ('and', 'as', 'assert', 'break', 'class', 'continue',
                    'def', 'del', 'elif', 'else', 'except', 'exec',
                    'finally', 'for', 'from', 'global', 'if', 'import',
                    'in', 'is', 'lambda', 'not', 'or', 'pass', 'print',
                    'raise', 'return', 'try', 'system', 'while', 'with',
                    'True', 'False', 'None', 'eval', 'execfile', '__import__',
                    '__package__', '__subclasses__', '__bases__', '__globals__',
                    '__code__', '__closure__', '__func__', '__self__', '__module__',
                    '__dict__', '__class__', '__call__', '__get__',
                    '__getattribute__', '__subclasshook__', '__new__',
                    '__init__', 'func_globals', 'func_code', 'func_closure',
                    'im_class', 'im_func', 'im_self', 'gi_code', 'gi_frame',
                    '__asteval__', 'f_locals', '__mro__')
        good_names = ['copy_reg._reconstructor', '__builtin__.object']

        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            fullname = module + '.' + name
            if (fullname in good_names)\
                or  (   (   module.startswith('sklearn.')
                            or module.startswith('xgboost.')
                            or module.startswith('skrebate.')
                            or module.startswith('imblearn')
                            or module.startswith('numpy.')
                            or module == 'numpy'
                        )
                        and (name not in bad_names)
                    ):
                # TODO: replace with a whitelist checker
                if fullname not in sk_whitelist['SK_NAMES'] + sk_whitelist['SKR_NAMES'] + sk_whitelist['XGB_NAMES'] + sk_whitelist['NUMPY_NAMES'] + sk_whitelist['IMBLEARN_NAMES'] + good_names:
                    print("Warning: global %s is not in pickler whitelist yet and will loss support soon. Contact tool author or leave a message at github.com" % fullname)
                mod = sys.modules[module]
                return getattr(mod, name)

        raise pickle.UnpicklingError("global '%s' is forbidden" % fullname)


def load_model(file):
    return SafePickler(file).load()


def read_columns(f, c=None, c_option='by_index_number', return_df=False, **args):
    data = pandas.read_csv(f, **args)
    if c_option == 'by_index_number':
        cols = list(map(lambda x: x - 1, c))
        data = data.iloc[:, cols]
    if c_option == 'all_but_by_index_number':
        cols = list(map(lambda x: x - 1, c))
        data.drop(data.columns[cols], axis=1, inplace=True)
    if c_option == 'by_header_name':
        cols = [e.strip() for e in c.split(',')]
        data = data[cols]
    if c_option == 'all_but_by_header_name':
        cols = [e.strip() for e in c.split(',')]
        data.drop(cols, axis=1, inplace=True)
    y = data.values
    if return_df:
        return y, data
    else:
        return y


## generate an instance for one of sklearn.feature_selection classes
def feature_selector(inputs):
    selector = inputs['selected_algorithm']
    if selector != 'DyRFECV':
        selector = getattr(sklearn.feature_selection, selector)
    options = inputs['options']

    if inputs['selected_algorithm'] == 'SelectFromModel':
        if not options['threshold'] or options['threshold'] == 'None':
            options['threshold'] = None
        else:
            try:
                options['threshold'] = float(options['threshold'])
            except ValueError:
                pass
        if inputs['model_inputter']['input_mode'] == 'prefitted':
            model_file = inputs['model_inputter']['fitted_estimator']
            with open(model_file, 'rb') as model_handler:
                fitted_estimator = load_model(model_handler)
            new_selector = selector(fitted_estimator, prefit=True, **options)
        else:
            estimator_json = inputs['model_inputter']['estimator_selector']
            estimator = get_estimator(estimator_json)
            estimator = check_feature_importances(estimator)
            new_selector = selector(estimator, **options)

    elif inputs['selected_algorithm'] == 'RFE':
        step = options.get('step', None)
        if step and step >= 1.0:
            options['step'] = int(step)
        estimator = get_estimator(inputs["estimator_selector"])
        estimator = check_feature_importances(estimator)
        new_selector = selector(estimator, **options)

    elif inputs['selected_algorithm'] == 'RFECV':
        options['scoring'] = get_scoring(options['scoring'])
        options['n_jobs'] = N_JOBS
        splitter, groups = get_cv(options.pop('cv_selector'))
        if groups is None:
            options['cv'] = splitter
        else:
            options['cv'] = list( splitter.split(X, y, groups=groups) )
        step = options.get('step', None)
        if step and step >= 1.0:
            options['step'] = int(step)
        estimator = get_estimator(inputs['estimator_selector'])
        estimator = check_feature_importances(estimator)
        new_selector = selector(estimator, **options)

    elif inputs['selected_algorithm'] == 'DyRFECV':
        options['scoring'] = get_scoring(options['scoring'])
        options['n_jobs'] = N_JOBS
        splitter, groups = get_cv(options.pop('cv_selector'))
        if groups is None:
            options['cv'] = splitter
        else:
            options['cv'] = list( splitter.split(X, y, groups=groups) )
        step = options.get('step')
        if not step or step == 'None':
            step = None
        else:
            step = ast.literal_eval(step)
        options['step'] = step
        estimator = get_estimator(inputs["estimator_selector"])
        estimator = check_feature_importances(estimator)
        new_selector = DyRFECV(estimator, **options)

    elif inputs['selected_algorithm'] == 'VarianceThreshold':
        new_selector = selector(**options)

    else:
        score_func = inputs['score_func']
        score_func = getattr(sklearn.feature_selection, score_func)
        new_selector = selector(score_func, **options)

    return new_selector


def get_X_y(params, file1, file2):
    input_type = params['selected_tasks']['selected_algorithms']['input_options']['selected_input']
    if input_type == 'tabular':
        header = 'infer' if params['selected_tasks']['selected_algorithms']['input_options']['header1'] else None
        column_option = params['selected_tasks']['selected_algorithms']['input_options']['column_selector_options_1']['selected_column_selector_option']
        if column_option in ['by_index_number', 'all_but_by_index_number', 'by_header_name', 'all_but_by_header_name']:
            c = params['selected_tasks']['selected_algorithms']['input_options']['column_selector_options_1']['col1']
        else:
            c = None
        X = read_columns(
            file1,
            c=c,
            c_option=column_option,
            sep='\t',
            header=header,
            parse_dates=True).astype(float)
    else:
        X = mmread(file1)

    header = 'infer' if params['selected_tasks']['selected_algorithms']['input_options']['header2'] else None
    column_option = params['selected_tasks']['selected_algorithms']['input_options']['column_selector_options_2']['selected_column_selector_option2']
    if column_option in ['by_index_number', 'all_but_by_index_number', 'by_header_name', 'all_but_by_header_name']:
        c = params['selected_tasks']['selected_algorithms']['input_options']['column_selector_options_2']['col2']
    else:
        c = None
    y = read_columns(
        file2,
        c=c,
        c_option=column_option,
        sep='\t',
        header=header,
        parse_dates=True)
    y = y.ravel()

    return X, y


class SafeEval(Interpreter):

    def __init__(self, load_scipy=False, load_numpy=False, load_estimators=False):

        # File opening and other unneeded functions could be dropped
        unwanted = ['open', 'type', 'dir', 'id', 'str', 'repr']

        # Allowed symbol table. Add more if needed.
        new_syms = {
            'np_arange': getattr(np, 'arange'),
            'ensemble_ExtraTreesClassifier': getattr(ensemble, 'ExtraTreesClassifier')
        }

        syms = make_symbol_table(use_numpy=False, **new_syms)

        if load_scipy:
            scipy_distributions = scipy.stats.distributions.__dict__
            for k, v in scipy_distributions.items():
                if isinstance(v, (scipy.stats.rv_continuous, scipy.stats.rv_discrete)):
                    syms['scipy_stats_' + k] = v

        if load_numpy:
            from_numpy_random = ['beta', 'binomial', 'bytes', 'chisquare', 'choice', 'dirichlet', 'division',
                                'exponential', 'f', 'gamma', 'geometric', 'gumbel', 'hypergeometric',
                                'laplace', 'logistic', 'lognormal', 'logseries', 'mtrand', 'multinomial',
                                'multivariate_normal', 'negative_binomial', 'noncentral_chisquare', 'noncentral_f',
                                'normal', 'pareto', 'permutation', 'poisson', 'power', 'rand', 'randint',
                                'randn', 'random', 'random_integers', 'random_sample', 'ranf', 'rayleigh',
                                'sample', 'seed', 'set_state', 'shuffle', 'standard_cauchy', 'standard_exponential',
                                'standard_gamma', 'standard_normal', 'standard_t', 'triangular', 'uniform',
                                'vonmises', 'wald', 'weibull', 'zipf']
            for f in from_numpy_random:
                syms['np_random_' + f] = getattr(np.random, f)

        if load_estimators:
            estimator_table = {
                'sklearn_svm' : getattr(sklearn, 'svm'),
                'sklearn_tree' : getattr(sklearn, 'tree'),
                'sklearn_ensemble' : getattr(sklearn, 'ensemble'),
                'sklearn_neighbors' : getattr(sklearn, 'neighbors'),
                'sklearn_naive_bayes' : getattr(sklearn, 'naive_bayes'),
                'sklearn_linear_model' : getattr(sklearn, 'linear_model'),
                'sklearn_cluster' : getattr(sklearn, 'cluster'),
                'sklearn_decomposition' : getattr(sklearn, 'decomposition'),
                'sklearn_preprocessing' : getattr(sklearn, 'preprocessing'),
                'sklearn_feature_selection' : getattr(sklearn, 'feature_selection'),
                'sklearn_kernel_approximation' : getattr(sklearn, 'kernel_approximation'),
                'skrebate_ReliefF': getattr(skrebate, 'ReliefF'),
                'skrebate_SURF': getattr(skrebate, 'SURF'),
                'skrebate_SURFstar': getattr(skrebate, 'SURFstar'),
                'skrebate_MultiSURF': getattr(skrebate, 'MultiSURF'),
                'skrebate_MultiSURFstar': getattr(skrebate, 'MultiSURFstar'),
                'skrebate_TuRF': getattr(skrebate, 'TuRF'),
                'xgboost_XGBClassifier' : getattr(xgboost, 'XGBClassifier'),
                'xgboost_XGBRegressor' : getattr(xgboost, 'XGBRegressor')
            }
            syms.update(estimator_table)

        for key in unwanted:
            syms.pop(key, None)

        super(SafeEval, self).__init__(symtable=syms, use_numpy=False, minimal=False,
                                        no_if=True, no_for=True, no_while=True, no_try=True,
                                        no_functiondef=True, no_ifexp=True, no_listcomp=False,
                                        no_augassign=False, no_assert=True, no_delete=True,
                                        no_raise=True, no_print=True)



def get_estimator(estimator_json):

    estimator_module = estimator_json['selected_module']

    if estimator_module == 'customer_estimator':
        c_estimator = estimator_json['c_estimator']
        with open(c_estimator, 'rb') as model_handler:
            new_model = load_model(model_handler)
        return new_model

    if estimator_module == 'IRAPS':
        iraps_core = IRAPSCore()
        core_params = estimator_json['text_params'].strip()
        if core_params != '':
            try:
                params = safe_eval('dict(' + core_params + ')')
            except ValueError:
                sys.exit("Unsupported parameter input: `%s`" % core_params)
            iraps_core.set_params(**params)
        options = {}
        if estimator_json['p_thres'] is not None:
            options['p_thres'] = estimator_json['p_thres']
        if estimator_json['fc_thres'] is not None:
            options['fc_thres'] = estimator_json['fc_thres']
        if estimator_json['occurrence'] is not None:
            options['occurrence'] = estimator_json['occurrence']
        if estimator_json['discretize'] is not None:
            options['discretize'] = estimator_json['discretize']
        return IRAPSClassifier(iraps_core, **options)

    if estimator_module == "binarize_target":
        wrapped_estimator = estimator_json['wrapped_estimator']
        with open(wrapped_estimator, 'rb') as model_handler:
            wrapped_estimator = load_model(model_handler)
        options = {}
        if estimator_json['z_score'] is not None:
            options['z_score'] = estimator_json['z_score']
        if estimator_json['value'] is not None:
            options['value'] = estimator_json['value']
        options['less_is_positive'] = estimator_json['less_is_positive']
        if estimator_json['clf_or_regr'] == 'BinarizeTargetClassifier':
            return BinarizeTargetClassifier(wrapped_estimator, **options)
        else:
            return BinarizeTargetRegressor(wrapped_estimator, **options)

    estimator_cls = estimator_json['selected_estimator']

    if estimator_module == 'xgboost':
        cls = getattr(xgboost, estimator_cls)
    else:
        module = getattr(sklearn, estimator_module)
        cls = getattr(module, estimator_cls)

    estimator = cls()

    estimator_params = estimator_json['text_params'].strip()
    if estimator_params != '':
        try:
            params = safe_eval('dict(' + estimator_params + ')')
        except ValueError:
            sys.exit("Unsupported parameter input: `%s`" % estimator_params)
        estimator.set_params(**params)
    if 'n_jobs' in estimator.get_params():
        estimator.set_params(n_jobs=N_JOBS)

    return estimator


def get_cv(cv_json):
    """
    cv_json:
            e.g.:
            {
                'selected_cv': 'StratifiedKFold',
                'n_splits': 3,
                'shuffle': True,
                'random_state': 0
            }
    """
    cv = cv_json.pop('selected_cv')
    if cv == 'default':
        return cv_json['n_splits'], None

    groups = cv_json.pop('groups_selector', None)
    if groups is not None:
        infile_g = groups['infile_g']
        header = 'infer' if groups['header_g'] else None
        column_option = groups['column_selector_options_g']['selected_column_selector_option_g']
        if column_option in ['by_index_number', 'all_but_by_index_number', 'by_header_name', 'all_but_by_header_name']:
            c = groups['column_selector_options_g']['col_g']
        else:
            c = None
        groups = read_columns(
                infile_g,
                c = c,
                c_option = column_option,
                sep='\t',
                header=header,
                parse_dates=True)
        groups = groups.ravel()

    for k, v in cv_json.items():
        if v == '':
            cv_json[k] = None

    test_fold = cv_json.get('test_fold', None)
    if test_fold:
        if test_fold.startswith('__ob__'):
            test_fold = test_fold[6:]
        if test_fold.endswith('__cb__'):
            test_fold = test_fold[:-6]
        cv_json['test_fold'] = [int(x.strip()) for x in test_fold.split(',')]

    test_size = cv_json.get('test_size', None)
    if test_size and test_size > 1.0:
        cv_json['test_size'] = int(test_size)

    if cv == 'OrderedKFold':
        cv_class = OrderedKFold
    else:
        cv_class = getattr(model_selection, cv)
    splitter = cv_class(**cv_json)

    return splitter, groups


# needed when sklearn < v0.20
def balanced_accuracy_score(y_true, y_pred):
    C = metrics.confusion_matrix(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        warnings.warn('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    return score


def get_scoring(scoring_json):

    if scoring_json['primary_scoring'] == 'default':
        return None

    my_scorers = metrics.SCORERS
    my_scorers['binarize_auc_scorer'] = binarize_auc_scorer
    my_scorers['binarize_average_precision_scorer'] = binarize_average_precision_scorer
    if 'balanced_accuracy' not in my_scorers:
        my_scorers['balanced_accuracy'] = metrics.make_scorer(balanced_accuracy_score)

    if scoring_json['secondary_scoring'] != 'None'\
            and scoring_json['secondary_scoring'] != scoring_json['primary_scoring']:
        scoring = {}
        scoring['primary'] = my_scorers[scoring_json['primary_scoring']]
        for scorer in scoring_json['secondary_scoring'].split(','):
            if scorer != scoring_json['primary_scoring']:
                scoring[scorer] = my_scorers[scorer]
        return scoring

    return my_scorers[scoring_json['primary_scoring']]
