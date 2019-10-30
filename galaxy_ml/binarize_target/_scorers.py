from sklearn import metrics
from sklearn.metrics.scorer import _BaseScorer
from sklearn.pipeline import Pipeline
from sklearn.model_selection._search import BaseSearchCV


def _get_main_estimator(estimator):
    est_name = estimator.__class__.__name__
    # support pipeline object
    if isinstance(estimator, Pipeline):
        return _get_main_estimator(estimator.steps[-1][-1])
    # support GridSearchCV/RandomSearchCV
    elif isinstance(estimator, BaseSearchCV):
        return _get_main_estimator(estimator.best_estimator_)
    # support stacking ensemble estimators
    # TODO support nested pipeline/stacking estimators
    elif est_name in ['StackingCVClassifier', 'StackingClassifier']:
        return _get_main_estimator(estimator.meta_clf_)
    elif est_name in ['StackingCVRegressor', 'StackingRegressor']:
        return _get_main_estimator(estimator.meta_regr_)
    else:
        return estimator


class _BinarizeTargetProbaScorer(_BaseScorer):
    """
    base class to make binarized target specific scorer
    """
    def __call__(self, clf, X, y, sample_weight=None):
        main_estimator = _get_main_estimator(clf)
        discretize_value = main_estimator.discretize_value
        less_is_positive = main_estimator.less_is_positive

        if less_is_positive:
            y_trans = y < discretize_value
        else:
            y_trans = y > discretize_value

        y_score = clf.predict_score(X)
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
        main_estimator = _get_main_estimator(clf)
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
