import sklearn
from nose.tools import raises
from galaxy_ml.utils import try_get_attr
from galaxy_ml.utils import SafeEval


def test_try_get_attr_1():
    try_get_attr('galaxy_ml.preprocessors', 'Z_RandomOverSampler')
    try_get_attr('galaxy_ml.iraps_classifier', 'IRAPSClassifier')


@raises(NameError)
def test_try_get_attr_2():
    try_get_attr('sklearn.utils', 'check_X_y')
    try_get_attr('galaxy_ml.preprocessors', 'check_X_y')
    try_get_attr('galaxy_ml.preprocessors', '_get_quantiles')


def test_safe_eval():
    safeeval = SafeEval(load_numpy=True, load_scipy=True)
    literals = ['[3, 5, 7, 9]',
                'list(range(50, 1001, 50))',
                'np_arange(0.01, 1, 0.1)',
                'np_random_choice(list(range(1, 51)) + [None], size=20)',
                'scipy_stats_randint(1, 11)']

    for li in literals:
        assert safeeval(li) is not None

    x_safeeval = SafeEval(load_estimators=True)

    literals = ['[sklearn_tree.DecisionTreeRegressor(), '
                'sklearn_tree.ExtraTreeRegressor()]',
                '[sklearn_feature_selection.SelectKBest(), '
                'sklearn_feature_selection.VarianceThreshold(), '
                'skrebate_ReliefF(), sklearn_preprocessing.RobustScaler()]']

    for li in literals:
        assert x_safeeval(li) is not None
