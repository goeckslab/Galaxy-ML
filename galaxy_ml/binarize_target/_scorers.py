import numpy as np

from sklearn import metrics
from sklearn.metrics._scorer import _BaseScorer
from sklearn.utils.multiclass import type_of_target

from ..utils import get_main_estimator


class _BinarizeTargetThresholdScorer(_BaseScorer):
    """
    Class to make binarized target specific scorer to evaluate decision
    function output.
    """
    def _score(self, method_caller, clf, X, y, sample_weight=None):
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
            y_pred = method_caller(clf, "decision_function", X)

            # For multi-output multi-class estimator
            if isinstance(y_pred, list):
                y_pred = np.vstack([p for p in y_pred]).T
            elif y_type == "binary" and "pos_label" in self._kwargs:
                self._check_pos_label(
                    self._kwargs["pos_label"], clf.classes_,
                )
                if self._kwargs["pos_label"] == clf.classes_[0]:
                    # The implicit positive class of the binary classifier
                    # does not match `pos_label`: we need to invert the
                    # predictions
                    y_pred *= -1

        except (NotImplementedError, AttributeError):
            y_pred = method_caller(clf, "predict_proba", X)

            if y_type == "binary":
                y_pred = self._select_proba_binary(y_pred, clf.classes_)
            elif isinstance(y_pred, list):
                y_pred = np.vstack([p[:, -1] for p in y_pred]).T

        if sample_weight is not None:
            return self._sign * self._score_func(
                y_trans, y_pred, sample_weight=sample_weight,
                **self._kwargs)
        else:
            return self._sign * self._score_func(
                y_trans, y_pred, **self._kwargs)

    def _factory_args(self):
        return ", needs_threshold=True"


# roc_auc
binarize_auc_scorer =\
    _BinarizeTargetThresholdScorer(metrics.roc_auc_score, 1, {})

# average_precision_scorer
binarize_average_precision_scorer =\
    _BinarizeTargetThresholdScorer(metrics.average_precision_score, 1, {})

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
    Class to make binarized target specific scorer to evaluate predicted
    target values.
    """
    def _score(self, method_caller, estimator, X, y, sample_weight=None):
        main_estimator = get_main_estimator(estimator)
        discretize_value = main_estimator.discretize_value
        less_is_positive = main_estimator.less_is_positive

        if less_is_positive:
            y_trans = y < discretize_value
        else:
            y_trans = y > discretize_value

        y_pred = method_caller(estimator, "predict", X)
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
    _BinarizeTargetPredictScorer(metrics.balanced_accuracy_score, 1, {})

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
