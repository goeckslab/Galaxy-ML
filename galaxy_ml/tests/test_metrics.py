import pandas as pd
import warnings

from xgboost import XGBRegressor
from sklearn.metrics._scorer import r2_scorer
from sklearn.model_selection import cross_validate
from galaxy_ml.metrics import (spearman_correlation_score,
                               spearman_correlation_scorer)
from galaxy_ml.model_validations import OrderedKFold


warnings.simplefilter('ignore')

Olaparib_1017 = pd.read_csv('./tools/test-data/Olaparib_1017.tsv.gz',
                            sep='\t', index_col=0)
X, y = Olaparib_1017.iloc[:, 6:].values, Olaparib_1017.iloc[:, 5].values


def test_spearman_correlation_score():
    y_true = [1, 2, 3, 4, 5]
    y_pred = [5, 6, 7, 8, 7]

    spearm = spearman_correlation_score(y_true, y_pred)

    assert round(spearm, 2) == 0.82, spearm


def test_spearman_correlation_scorer():
    cv = OrderedKFold(5)
    scoring = dict(
        R2=r2_scorer,
        Spearman=spearman_correlation_scorer
    )

    estimator = XGBRegressor(random_state=42, booster='gblinear')

    result_val = cross_validate(
        estimator, X, y, cv=cv, scoring=scoring,
        verbose=0, n_jobs=2)

    r2_mean = result_val['test_R2'].mean()
    assert round(r2_mean, 4) == -0.0802, r2_mean

    spearman_mean = result_val['test_Spearman'].mean()
    assert round(spearman_mean, 4) == 0.1481, spearman_mean
