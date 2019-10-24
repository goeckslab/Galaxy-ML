# test for irpas_classifier

import pandas as pd
import time
import warnings
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import balanced_accuracy_scorer
from sklearn.metrics.scorer import r2_scorer
from galaxy_ml.model_validations import OrderedKFold
from galaxy_ml.iraps_classifier import (
    IRAPSCore, IRAPSClassifier, BinarizeTargetClassifier,
    BinarizeTargetRegressor, binarize_auc_scorer,
    binarize_average_precision_scorer, BINARIZE_SCORERS)


warnings.simplefilter('ignore')

Olaparib_1017 = pd.read_csv('./tools/test-data/Olaparib_1017.tsv.gz',
                            sep='\t', index_col=0)
X, y = Olaparib_1017.iloc[:, 6:].values, Olaparib_1017.iloc[:, 5].values


def test_iraps_classifier_1():
    cv = OrderedKFold(10)
    scoring = dict(
        AUC=binarize_auc_scorer,
        AP=binarize_average_precision_scorer
    )
    iraps_core = IRAPSCore(
        n_iter=100, n_jobs=2, random_state=10, verbose=10,
        parallel_backend='threading')
    iraps = IRAPSClassifier(iraps_core, p_thres=0.01, occurrence=0.7)
    start_time = time.time()
    result_clf = cross_validate(
        iraps, X, y, cv=cv, scoring=scoring, verbose=10, n_jobs=2)
    stop_time = time.time()
    print("Time: %f " % (stop_time - start_time))
    ap_mean = result_clf['test_AP'].mean()
    assert round(ap_mean, 4) == 0.2879, ap_mean
    roc_mean = result_clf['test_AUC'].mean()
    assert round(roc_mean, 4) == 0.7038, roc_mean


def test_binarize_target_classifier():
    cv = OrderedKFold(10)
    scoring = dict(
        AUC=binarize_auc_scorer,
        AP=binarize_average_precision_scorer,
        BACC=BINARIZE_SCORERS['balanced_accuracy'],
        PRECISION=BINARIZE_SCORERS['precision'],
        F1_MACRO=BINARIZE_SCORERS['f1_macro']
    )
    clf = RandomForestClassifier(random_state=42)
    estimator = BinarizeTargetClassifier(clf)

    result_val = cross_validate(
        estimator, X, y, cv=cv, scoring=scoring,
        verbose=0, n_jobs=2)
    
    ap_mean = result_val['test_AP'].mean()
    assert round(ap_mean, 4) == 0.1710, ap_mean

    roc_mean = result_val['test_AUC'].mean()
    assert round(roc_mean, 4) == 0.5696, roc_mean

    bacc_mean = result_val['test_BACC'].mean()
    assert round(bacc_mean, 4) == 0.5089, bacc_mean

    precision_mean = result_val['test_PRECISION'].mean()
    assert round(precision_mean, 4) == 0.1, precision_mean

    f1_macro_mean = result_val['test_F1_MACRO'].mean()
    assert round(f1_macro_mean, 4) == 0.4922, f1_macro_mean


def test_binarize_target_regressor():
    cv = OrderedKFold(10)
    scoring = dict(
        AUC=binarize_auc_scorer,
        AP=binarize_average_precision_scorer,
        R2=r2_scorer
    )
    clf = RandomForestRegressor(random_state=42)
    estimator = BinarizeTargetRegressor(clf)

    result_val = cross_validate(
        estimator, X, y, cv=cv, scoring=scoring,
        verbose=0, n_jobs=2)
    
    ap_mean = result_val['test_AP'].mean()
    assert round(ap_mean, 4) == 0.1794, ap_mean
    roc_mean = result_val['test_AUC'].mean()
    assert round(roc_mean, 4) == 0.5764, roc_mean
    r2_mean = result_val['test_R2'].mean()
    assert round(r2_mean, 4) == -0.2046, r2_mean
