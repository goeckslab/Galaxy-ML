"""
Reference:

McGrail, D.J., Lin, C.C., Garnett, J. et al. Improved prediction of
PARP inhibitor response and identification of synergizing agents
through use of a novel gene expression signature generation algorithm.
npj Syst Biol Appl 3, 8 (2017) doi:10.1038/s41540-017-0011-6
"""

import warnings
from abc import ABCMeta

from joblib import Parallel, delayed

import numpy as np

from scipy.stats import ttest_ind

import six

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.feature_selection._univariate_selection import _BaseFilter
from sklearn.utils import as_float_array, check_X_y, check_random_state
from sklearn.utils.validation import check_is_fitted, check_memory


VERSION = '0.1.2'


class IRAPSCore(six.with_metaclass(ABCMeta, BaseEstimator)):
    """
    Base class of IRAPSClassifier
    From sklearn BaseEstimator:
        get_params()
        set_params()

    Parameters
    ----------
    n_iter : int
        sample count
    positive_thres : float
        z_score shreshold to discretize positive target values
    negative_thres : float
        z_score threshold to discretize negative target values
    verbose : int
        0 or geater, if not 0, print progress
    n_jobs : int, default=1
        The number of CPUs to use to do the computation.
    pre_dispatch : int, or string.
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    random_state : int or None
    parallel_backend : str, default is None.
        Used in `joblib.Parallel`. Common options are 'loky',
        'multiprocessing', and 'threading'. If None, `loky` is used.
    """

    def __init__(self, n_iter=1000, positive_thres=-1, negative_thres=0,
                 verbose=0, n_jobs=1, pre_dispatch='2*n_jobs',
                 random_state=None, parallel_backend=None):
        """
        IRAPS turns towwards general Anomaly Detection
        It comapares positive_thres with negative_thres,
        and decide which portion is the positive target.
        e.g.:
        (positive_thres=-1, negative_thres=0)
                 => positive = Z_score of target < -1
        (positive_thres=1, negative_thres=0)
                 => positive = Z_score of target > 1

        Note: The positive targets here is always the
            abnormal minority group.
        """
        self.n_iter = n_iter
        self.positive_thres = positive_thres
        self.negative_thres = negative_thres
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.random_state = random_state
        self.parallel_backend = parallel_backend

    def fit(self, X, y):
        """
        X: array-like (n_samples x n_features)
        y: 1-d array-like (n_samples)
        """
        SAMPLE_SIZE = [0.25, 0.75]

        X, y = check_X_y(X, y, ['csr', 'csc'], multi_output=False)
        n_samples = X.shape[0]
        min_samples = int(n_samples * SAMPLE_SIZE[0])
        max_samples = int(n_samples * SAMPLE_SIZE[1])
        span = max_samples - min_samples

        rng = check_random_state(self.random_state)

        def iter_index():
            for i in range(self.n_iter):
                n_select = min_samples + int(span * rng.uniform(0, 1))
                yield rng.choice(np.arange(n_samples), n_select, replace=False)

        def _stochastic_sampling(X, y, index, positive_thres=-1,
                                 negative_thres=0):
            # each iteration select a random number of random subset of
            # training samples. this is somewhat different from the original
            # IRAPS method, but effect is almost the same.
            X_selected, y_selected = X[index], y[index]

            # Spliting by z_scores.
            y_selected = (y_selected - y_selected.mean()) / y_selected.std()
            if positive_thres < negative_thres:
                X_selected_positive = X_selected[y_selected < positive_thres]
                X_selected_negative = X_selected[y_selected > negative_thres]
            else:
                X_selected_positive = X_selected[y_selected > positive_thres]
                X_selected_negative = X_selected[y_selected < negative_thres]

            # For every iteration, at least 5 responders are selected
            if X_selected_positive.shape[0] < 5:
                warnings.warn("Warning: fewer than 5 positives were selected!")
                return

            # p_values
            _, p = ttest_ind(X_selected_positive, X_selected_negative,
                             axis=0, equal_var=False)

            # fold_change == mean change?
            # TODO implement other normalization method
            positive_mean = X_selected_positive.mean(axis=0)
            negative_mean = X_selected_negative.mean(axis=0)
            mean_change = positive_mean - negative_mean
            # mean_change = np.select(
            #       [positive_mean > negative_mean,
            #           positive_mean < negative_mean],
            #       [positive_mean / negative_mean,
            #           -negative_mean / positive_mean])
            # mean_change could be adjusted by power of 2
            # mean_change = 2**mean_change \
            #       if mean_change>0 else -2**abs(mean_change)

            return p, mean_change, negative_mean

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                            pre_dispatch=self.pre_dispatch,
                            backend=self.parallel_backend)

        res = parallel(delayed(_stochastic_sampling)(
            X, y, index,
            positive_thres=self.positive_thres,
            negative_thres=self.negative_thres)
            for index in iter_index()
        )

        # remove bads
        res = [_ for _ in res if _]
        if len(res) < 50:
            raise ValueError("too few (%d) valid feature lists "
                             "were generated!" % len(res))
        pvalues = np.vstack([x[0] for x in res])
        fold_changes = np.vstack([x[1] for x in res])
        base_values = np.vstack([x[2] for x in res])

        self.pvalues_ = np.asarray(pvalues)
        self.fold_changes_ = np.asarray(fold_changes)
        self.base_values_ = np.asarray(base_values)

        return self


def _iraps_core_fit(iraps_core, X, y):
    return iraps_core.fit(X, y)


class IRAPSClassifier(six.with_metaclass(ABCMeta, _BaseFilter,
                                         BaseEstimator, RegressorMixin)):
    """
    Extend the bases of both sklearn feature_selector and classifier.
    From sklearn BaseEstimator:
        get_params()
        set_params()
    From sklearn _BaseFilter:
        get_support()
        fit_transform(X)
        transform(X)
    From sklearn RegressorMixin:
        score(X, y): R2
    New:
        predict(X)
        predict_label(X)
        get_signature()
    Properties:
        discretize_value

    Parameters
    ----------
    iraps_core : object
    p_thres : float, threshold for p_values
    fc_thres : float, threshold for fold change or mean difference
    occurrence : float, occurrence rate selected by set of p_thres and fc_thres
    discretize : float, threshold of z_score to discretize target value
    memory : None, str or joblib.Memory object
    min_signature_features : int, the mininum number of features in a signature
    """

    def __init__(self, iraps_core, p_thres=1e-4, fc_thres=0.1,
                 occurrence=0.8, discretize=-1, memory=None,
                 min_signature_features=1):
        self.iraps_core = iraps_core
        self.p_thres = p_thres
        self.fc_thres = fc_thres
        self.occurrence = occurrence
        self.discretize = discretize
        self.memory = memory
        self.min_signature_features = min_signature_features

    def fit(self, X, y):
        memory = check_memory(self.memory)
        cached_fit = memory.cache(_iraps_core_fit)
        iraps_core = clone(self.iraps_core)
        # allow pre-fitted iraps_core here
        if not hasattr(iraps_core, 'pvalues_'):
            iraps_core = cached_fit(iraps_core, X, y)
        self.iraps_core_ = iraps_core

        pvalues = as_float_array(iraps_core.pvalues_, copy=True)
        # why np.nan is here?
        pvalues[np.isnan(pvalues)] = np.finfo(pvalues.dtype).max

        fold_changes = as_float_array(iraps_core.fold_changes_, copy=True)
        fold_changes[np.isnan(fold_changes)] = 0.0

        base_values = as_float_array(iraps_core.base_values_, copy=True)

        p_thres = self.p_thres
        fc_thres = self.fc_thres
        occurrence = self.occurrence

        mask_0 = np.zeros(pvalues.shape, dtype=np.int32)
        # mark p_values less than the threashold
        mask_0[pvalues <= p_thres] = 1
        # mark fold_changes only when greater than the threashold
        mask_0[abs(fold_changes) < fc_thres] = 0

        # count the occurrence and mask greater than the threshold
        counts = mask_0.sum(axis=0)
        occurrence_thres = int(occurrence * iraps_core.n_iter)
        mask = np.zeros(counts.shape, dtype=bool)
        mask[counts >= occurrence_thres] = 1

        # generate signature
        fold_changes[mask_0 == 0] = 0.0
        signature = fold_changes[:, mask].sum(axis=0) / counts[mask]
        signature = np.vstack((signature, base_values[:, mask].mean(axis=0)))
        # It's not clearn whether min_size could impact prediction
        # performance
        if signature is None\
                or signature.shape[1] < self.min_signature_features:
            raise ValueError("The classifier got None signature or the number "
                             "of sinature feature is less than minimum!")

        self.signature_ = np.asarray(signature)
        self.mask_ = mask
        # TODO: support other discretize method: fixed value, upper
        # third quater, etc.
        self.discretize_value = y.mean() + y.std() * self.discretize
        if iraps_core.negative_thres > iraps_core.positive_thres:
            self.less_is_positive = True
        else:
            self.less_is_positive = False

        return self

    def _get_support_mask(self):
        """
        return mask of feature selection indices
        """
        check_is_fitted(self, 'mask_')

        return self.mask_

    def get_signature(self):
        """
        return signature
        """
        check_is_fitted(self, 'signature_')

        return self.signature_

    def decision_function(self, X):
        """
        compute the correlation coefficient with irpas signature
        """
        signature = self.get_signature()

        X = as_float_array(X)
        X_transformed = self.transform(X) - signature[1]
        corrcoef = np.array(
            [np.corrcoef(signature[0], e)[0][1] for e in X_transformed])
        corrcoef[np.isnan(corrcoef)] = np.finfo(np.float32).min

        return corrcoef

    def predict(self, X):
        return self.decision_function(X)

    def predict_label(self, X, clf_cutoff=0.4):
        return self.decision_function(X) >= clf_cutoff
