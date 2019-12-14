import numpy as np
from ..utils import get_main_estimator
from sklearn import metrics
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics.scorer import _BaseScorer


class _BinarizeTargetProbaScorer(_BaseScorer):
    """
    base class to make binarized target specific scorer
    """
    def __call__(self, clf, X, y, sample_weight=None):
        main_estimator = get_main_estimator(clf)
        discretize_value = main_estimator.discretize_value
        less_is_positive = main_estimator.less_is_positive

        if less_is_positive:
            y_trans = y < discretize_value
        else:
            y_trans = y > discretize_value

        y_type = type_of_target(y_trans)
        if y_type not in ("binary", "multilabel-indicator"):
            raise ValueError("{0} format is not supported".format(y_type))

        try:
            y_score = clf.decision_function(X)

            # For multi-output multi-class estimator
            if isinstance(y_score, list):
                y_score = np.vstack([p for p in y_score]).T

        except (NotImplementedError, AttributeError):
            y_score = clf.predict_proba(X)

            if y_type == "binary":
                if y_score.shape[1] == 2:
                    y_score = y_score[:, 1]
                else:
                    raise ValueError('got predict_proba of shape {},'
                                     ' but need classifier with two'
                                     ' classes for {} scoring'.format(
                                         y_score.shape,
                                         self._score_func.__name__))
            elif isinstance(y_score, list):
                y_score = np.vstack([p[:, -1] for p in y_score]).T

        if sample_weight is not None:
            return self._sign * self._score_func(y_trans, y_score,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_trans, y_score,
                                                 **self._kwargs)


# roc_auc
binarize_auc_scorer =\
        _BinarizeTargetProbaScorer(metrics.roc_auc_score, 1, {})

# average_precision_scorer
binarize_average_precision_scorer =\
        _BinarizeTargetProbaScorer(metrics.average_precision_score, 1, {})

# roc_auc_scorer
# iraps_auc_scorer = binarize_auc_scorer

# average_precision_scorer
# iraps_average_precision_scorer = binarize_average_precision_scorer

# roc_auc_scorer
# regression_auc_scorer = binarize_auc_scorer

# average_precision_scorer
# regression_average_precision_scorer = binarize_average_precision_scorer


class _BinarizeTargetPredictScorer(_BaseScorer):
    """
    base class to make binarized target specific scorer
    """
    def __call__(self, clf, X, y, sample_weight=None):
        main_estimator = get_main_estimator(clf)
        discretize_value = main_estimator.discretize_value
        less_is_positive = main_estimator.less_is_positive

        if less_is_positive:
            y_trans = y < discretize_value
        else:
            y_trans = y > discretize_value

        y_pred = clf.predict(X)
        if sample_weight is not None:
            return self._sign * self._score_func(y_trans, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_trans, y_pred,
                                                 **self._kwargs)


# accuracy_scorer
binarize_accuracy_scorer =\
        _BinarizeTargetPredictScorer(metrics.accuracy_score, 1, {})

# balanced_accuracy_scorer
binarize_balanced_accuracy_scorer =\
        _BinarizeTargetPredictScorer(
            metrics.balanced_accuracy_score, 1, {})

BINARIZE_SCORERS = dict(
    roc_auc=binarize_auc_scorer,
    average_precision=binarize_average_precision_scorer,
    accuracy=binarize_accuracy_scorer,
    balanced_accuracy=binarize_balanced_accuracy_scorer,
)


for name, metric in [('precision', metrics.precision_score),
                     ('recall', metrics.recall_score),
                     ('f1', metrics.f1_score)]:
    BINARIZE_SCORERS[name] = _BinarizeTargetPredictScorer(
        metric, 1, dict(average='binary'))
    for average in ['macro', 'micro', 'samples', 'weighted']:
        qualified_name = '{0}_{1}'.format(name, average)
        BINARIZE_SCORERS[qualified_name] = _BinarizeTargetPredictScorer(
            metric, 1, dict(pos_label=None, average=average))

# for regressor scorer
BINARIZE_SCORERS['explained_variance'] = \
    metrics.SCORERS['explained_variance']
BINARIZE_SCORERS['r2'] = metrics.SCORERS['r2']
BINARIZE_SCORERS['neg_median_absolute_error'] = \
    metrics.SCORERS['neg_median_absolute_error']
BINARIZE_SCORERS['neg_mean_absolute_error'] = \
    metrics.SCORERS['neg_mean_absolute_error']
BINARIZE_SCORERS['neg_mean_squared_error'] = \
    metrics.SCORERS['neg_mean_squared_error']
BINARIZE_SCORERS['neg_mean_squared_log_error'] = \
    metrics.SCORERS['neg_mean_squared_log_error']
