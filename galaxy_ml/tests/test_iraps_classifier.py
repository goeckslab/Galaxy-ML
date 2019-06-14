# test for irpas_classifier

import pandas as pd
import time
import warnings
from sklearn.model_selection import cross_validate
from galaxy_ml.model_validations import OrderedKFold
from galaxy_ml.iraps_classifier import (
    IRAPSCore, IRAPSClassifier, BinarizeTargetClassifier,
    BinarizeTargetRegressor, binarize_auc_scorer,
    binarize_average_precision_scorer)


warnings.simplefilter('ignore')

Olaparib_1017 = pd.read_csv('./test-data/Olaparib_1017.tsv.gz',
                            sep='\t', index_col=0)
X, y = Olaparib_1017.iloc[:, 6:].values, Olaparib_1017.iloc[:, 5].values


def test_iraps_classifier_1():
    cv = OrderedKFold(10)
    scoring = dict(
        AUC=binarize_auc_scorer,
        AP=binarize_average_precision_scorer
    )
    iraps_core = IRAPSCore(
        n_iter=100, n_jobs=3, random_state=10, verbose=10)
    iraps = IRAPSClassifier(iraps_core, p_thres=0.01, occurrence=0.7)
    start_time = time.time()
    result_clf = cross_validate(
        iraps, X, y, cv=cv, scoring=scoring, verbose=10)
    stop_time = time.time()
    print("Time: %f " % (stop_time - start_time))
    ap_mean = result_clf['test_AP'].mean()
    assert round(ap_mean, 4) == 0.2879, ap_mean
    roc_mean = result_clf['test_AUC'].mean()
    assert round(roc_mean, 4) == 0.7038, roc_mean
