"""
class IRAPScore
class IRAPSClassifier
class IRAPSScorer
iraps_auc_scorer
"""

import numpy as np
import random

from abc import ABCMeta
from scipy.stats import ttest_ind
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.externals import joblib, six
from sklearn.feature_selection.univariate_selection import _BaseFilter
from sklearn.metrics import roc_auc_score
from sklearn.metrics.scorer import _BaseScorer
from sklearn.pipeline import Pipeline
from sklearn.utils import as_float_array, check_X_y
from sklearn.utils.validation import check_is_fitted


VERSION = '0.1.1'


class IRAPSCore(six.with_metaclass(ABCMeta, BaseEstimator)):
    """
    Base class of IRAPSClassifier
    From sklearn BaseEstimator:
        get_params()
        set_params()
    """
    def __init__(self, n_iter=1000, positive_thres=-1,
                negative_thres=0, verbose=0, random_state=None):
        """
        IRAPS turns towwards general Anomaly Detection
        It comapares positive_thres with negative_thres,
        and decide which portion is the positive target.
        e.g.:
        (positive_thres=-1, negative_thres=0) => positive = Z_score of target < -1
        (positive_thres=1, negative_thres=0) => positive = Z_score of target > 1

        Note: The positive targets here is always the abnormal minority group.
        """
        self.n_iter = n_iter
        self.positive_thres = positive_thres
        self.negative_thres = negative_thres
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y):
        """
        X: array-like (n_samples x n_features)
        y: 1-d array-like (n_samples)
        """
        X, y = check_X_y(X, y, ['csr', 'csc'], multi_output=True)
        #each iteration select a random number of random subset of training samples
        # this is somewhat different from the original IRAPS method, but effect is almost the same.
        SAMPLE_SIZE = [0.25, 0.75]
        n_samples = X.shape[0]
        pvalues = None
        fold_changes = None
        base_values = None

        i = 0
        while i < self.n_iter:
            ## TODO: support more random_state/seed
            if self.random_state is None:
                n_select = random.randint(int(n_samples*SAMPLE_SIZE[0]), int(n_samples*SAMPLE_SIZE[1]))
                index = random.sample(list(range(n_samples)), n_select)
            else:
                n_select = random.Random(i).randint(int(n_samples*SAMPLE_SIZE[0]), int(n_samples*SAMPLE_SIZE[1]))
                index = random.Random(i).sample(list(range(n_samples)), n_select)
            X_selected, y_selected = X[index], y[index]

            # Spliting by z_scores.
            y_selected = (y_selected - y_selected.mean())/y_selected.std()
            if self.positive_thres < self.negative_thres:
                X_selected_positive = X_selected[y_selected < self.positive_thres]
                X_selected_negative = X_selected[y_selected > self.negative_thres]
            else:
                X_selected_positive = X_selected[y_selected > self.positive_thres]
                X_selected_negative = X_selected[y_selected < self.negative_thres]

            # For every iteration, at least 5 responders are selected
            if X_selected_positive.shape[0] < 5:
                if self.random_state is not None:
                    raise Exception("Error: fewer than 5 positives were selected while random_state is not None!")
                continue

            if self.verbose:
                print("Working on iteration %d/%d, %s/%d samples were positive/selected."\
                        %(i+1, self.n_iter, X_selected_positive.shape[0], n_select))
            i += 1

            # p_values
            _, p = ttest_ind(X_selected_positive, X_selected_negative, axis=0, equal_var=False)
            if pvalues is None:
                pvalues = p
            else:
                pvalues = np.vstack((pvalues, p))

            # fold_change == mean change?
            # TODO implement other normalization method
            positive_mean = X_selected_positive.mean(axis=0)
            negative_mean = X_selected_negative.mean(axis=0)
            mean_change = positive_mean - negative_mean
            #mean_change = np.select([positive_mean > negative_mean, positive_mean < negative_mean],
            #                        [positive_mean / negative_mean, -negative_mean / positive_mean])
            # mean_change could be adjusted by power of 2
            # mean_change = 2**mean_change if mean_change>0 else -2**abs(mean_change)
            if fold_changes is None:
                fold_changes = mean_change
            else:
                fold_changes = np.vstack((fold_changes, mean_change))

            if base_values is None:
                base_values = negative_mean
            else:
                base_values = np.vstack((base_values, negative_mean))

        self.fold_changes_ = np.asarray(fold_changes)
        self.pvalues_ = np.asarray(pvalues)
        self.base_values_ = np.asarray(base_values)

        return self


"""
memory = joblib.Memory('./memory_cache')
class MemoryFit(object):
    def fit(self, *args, **kwargs):
        fit = memory.cache(super(MemoryFit, self).fit)
        cached_self = fit(*args, **kwargs)
        vars(self).update(vars(cached_self))
class CachedIRAPSCore(MemoryFit, IRAPSCore):
    pass
"""


class IRAPSClassifier(six.with_metaclass(ABCMeta, _BaseFilter, BaseEstimator, RegressorMixin)):
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
        predict_discretize(X)
        get_signature()
    Properties:
        discretize_value

    """
    def __init__(self, iraps_core, p_thres=1e-4, fc_thres=0.1, occurance=0.8, discretize='z_score'):
        self.iraps_core = iraps_core
        self.p_thres = p_thres
        self.fc_thres = fc_thres
        self.occurance = occurance
        self.discretize = discretize

    def fit(self, X, y):
        # allow pre-fitted iraps_core here
        if not hasattr(self.iraps_core, 'pvalues_'):
            self.iraps_core.fit(X, y)

        pvalues = as_float_array(self.iraps_core.pvalues_, copy=True)
        ## why np.nan is here?
        pvalues[np.isnan(pvalues)] = np.finfo(pvalues.dtype).max

        fold_changes = as_float_array(self.iraps_core.fold_changes_, copy=True)
        fold_changes[np.isnan(fold_changes)] = 0.0

        base_values = as_float_array(self.iraps_core.base_values_, copy=True)

        p_thres = self.p_thres
        fc_thres = self.fc_thres
        occurance = self.occurance

        mask_0 = np.zeros(pvalues.shape, dtype=np.int32)
        # mark p_values less than the threashold
        mask_0[pvalues <= p_thres] = 1
        # mark fold_changes only when greater than the threashold
        mask_0[abs(fold_changes) < fc_thres] = 0

        # count the occurance and mask greater than the threshold
        counts = mask_0.sum(axis=0)
        occurance_thres = int(occurance * self.iraps_core.n_iter)
        mask = np.zeros(counts.shape, dtype=bool)
        mask[counts >= occurance_thres] = 1

        # generate signature
        fold_changes[mask_0 == 0] = 0.0
        signature = fold_changes[:, mask].sum(axis=0) / counts[mask]
        signature = np.vstack((signature, base_values[:, mask].mean(axis=0)))

        self.signature_ = np.asarray(signature)
        self.mask_ = mask
        ## TODO: support other discretize method: fixed value, upper third quater, etc.
        self.discretize_value = y.mean() + y.std() * self.iraps_core.positive_thres

        return self


    def _get_support_mask(self):
        """
        return mask of feature selection indices
        """
        check_is_fitted(self, 'mask_')

        return self.mask_

    def get_signature(self, min_size=1):
        """
        return signature
        """
        #TODO: implement minimum size of signature
        # It's not clearn whether min_size could impact prediction performance
        check_is_fitted(self, 'signature_')

        if self.signature_.shape[1] >= min_size:
            return self.signature_
        else:
            return None

    def predict(self, X):
        """
        compute the correlation coefficient with irpas signature
        """
        signature = self.get_signature()
        if signature is None:
            print('The classifier got None signature or the number of sinature feature is less than minimum!')
            return

        X_transformed = self.transform(X) - signature[1]
        corrcoef = np.array([np.corrcoef(signature[0], e)[0][1] for e in X_transformed])
        corrcoef[np.isnan(corrcoef)] = np.finfo(np.float32).min

        return corrcoef

    def predict_discretize(self, X, clf_cutoff=0.4):
        return self.predict(X) >= clf_cutoff


class IRAPSScorer(_BaseScorer):
    """
    base class to make IRAPS specific scorer
    """
    def __call__(self, clf, X, y, sample_weight=None):
        # support pipeline object
        if isinstance(clf, Pipeline):
            clf = clf.steps[-1][-1]
        if clf.iraps_core.positive_thres < clf.iraps_core.negative_thres:
            y_true = y < clf.discretize_value
        else:
            y_true = y > clf.discretize_value
        y_pred = clf.predict(X)
        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred, **self._kwargs)

    def _factory_args(self):
        return ", needs_iraps=True"

iraps_auc_scorer = IRAPSScorer(roc_auc_score, 1, {})

