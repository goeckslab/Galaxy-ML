import os
import re

from nose.tools import raises
from sklearn import ensemble

from galaxy_ml.utils import safe_load_model
from galaxy_ml.utils import _SafePickler
from galaxy_ml.utils import try_get_attr
from galaxy_ml.utils import SafeEval
from galaxy_ml.utils import get_scoring
from galaxy_ml.utils import find_members


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


def test_get_scoring():
    inputs = {
        "primary_scoring": "binarize_average_precision_scorer",
        "secondary_scoring": "binarize_auc_scorer"
    }

    scoring = get_scoring(inputs)
    assert len(scoring) == 2

    inputs = {
        "primary_scoring": "r2",
        "secondary_scoring":
            "spearman_correlation,max_error,explained_variance"
    }

    scoring = get_scoring(inputs)
    assert len(scoring) == 4

    inputs = {
        "primary_scoring": "default",
    }

    scoring = get_scoring(inputs)
    assert scoring is None

    inputs = {
        "primary_scoring": "r2",
    }

    scoring = get_scoring(inputs)
    assert type(scoring).__name__ == '_PredictScorer', scoring


def test_safe_load_model():
    model = './tools/test-data/RandomForestRegressor01.zip'
    with open(model, 'rb') as fh:
        safe_unpickler = _SafePickler(fh)

    assert ensemble.RandomForestClassifier == \
        safe_unpickler.find_class('sklearn.ensemble._forest',
                                  'RandomForestClassifier')

    test_folder = './tools/test-data'
    for name in os.listdir(test_folder):
        if re.match('^(?!.*json).*(pipeline|model|regressor)\d+.*$',
                    name, flags=re.I):
            if name in ('gbr_model01_py3', 'rfr_model01'):
                continue
            model = os.path.join(test_folder, name)
            print(model)
            with open(model, 'rb') as fh:
                safe_load_model(fh)


def test_find_members():
    got = find_members('galaxy_ml.metrics')
    expect = [
        'galaxy_ml.metrics._regression.spearman_correlation_score'
    ]
    assert got == expect, got

    got = find_members('imblearn')
    expect = [
        "imblearn.LazyLoader",
        "imblearn.base.BaseSampler",
        "imblearn.base.FunctionSampler",
        "imblearn.base.SamplerMixin",
        "imblearn.base._identity",
        "imblearn.combine._smote_enn.SMOTEENN",
        "imblearn.combine._smote_tomek.SMOTETomek",
        "imblearn.datasets._imbalance.make_imbalance",
        "imblearn.datasets._zenodo.fetch_datasets",
        "imblearn.ensemble._bagging.BalancedBaggingClassifier",
        "imblearn.ensemble._easy_ensemble.EasyEnsembleClassifier",
        "imblearn.ensemble._forest.BalancedRandomForestClassifier",
        "imblearn.ensemble._forest._local_parallel_build_trees",
        "imblearn.ensemble._weight_boosting.RUSBoostClassifier",
        "imblearn.exceptions.raise_isinstance_error",
        "imblearn.keras._generator.BalancedBatchGenerator",
        "imblearn.keras._generator.balanced_batch_generator",
        "imblearn.keras._generator.import_keras",
        "imblearn.metrics._classification.classification_report_imbalanced",
        "imblearn.metrics._classification.geometric_mean_score",
        "imblearn.metrics._classification.macro_averaged_mean_absolute_error",
        "imblearn.metrics._classification.make_index_balanced_accuracy",
        "imblearn.metrics._classification.sensitivity_score",
        "imblearn.metrics._classification.sensitivity_specificity_support",
        "imblearn.metrics._classification.specificity_score",
        "imblearn.metrics.pairwise.ValueDifferenceMetric",
        "imblearn.over_sampling._adasyn.ADASYN",
        "imblearn.over_sampling._random_over_sampler.RandomOverSampler",
        "imblearn.over_sampling._smote.base.BaseSMOTE",
        "imblearn.over_sampling._smote.base.SMOTE",
        "imblearn.over_sampling._smote.base.SMOTEN",
        "imblearn.over_sampling._smote.base.SMOTENC",
        "imblearn.over_sampling._smote.cluster.KMeansSMOTE",
        "imblearn.over_sampling._smote.filter.BorderlineSMOTE",
        "imblearn.over_sampling._smote.filter.SVMSMOTE",
        "imblearn.over_sampling.base.BaseOverSampler",
        "imblearn.pipeline.Pipeline",
        "imblearn.pipeline._fit_resample_one",
        "imblearn.pipeline.make_pipeline",
        "imblearn.tensorflow._generator.balanced_batch_generator",
        "imblearn.under_sampling._prototype_generation._cluster_centroids.ClusterCentroids",
        "imblearn.under_sampling._prototype_selection._condensed_nearest_neighbour.CondensedNearestNeighbour",
        "imblearn.under_sampling._prototype_selection._edited_nearest_neighbours.AllKNN",
        "imblearn.under_sampling._prototype_selection._edited_nearest_neighbours.EditedNearestNeighbours",
        "imblearn.under_sampling._prototype_selection._edited_nearest_neighbours.RepeatedEditedNearestNeighbours",
        "imblearn.under_sampling._prototype_selection._instance_hardness_threshold.InstanceHardnessThreshold",
        "imblearn.under_sampling._prototype_selection._nearmiss.NearMiss",
        "imblearn.under_sampling._prototype_selection._neighbourhood_cleaning_rule.NeighbourhoodCleaningRule",
        "imblearn.under_sampling._prototype_selection._one_sided_selection.OneSidedSelection",
        "imblearn.under_sampling._prototype_selection._random_under_sampler.RandomUnderSampler",
        "imblearn.under_sampling._prototype_selection._tomek_links.TomekLinks",
        "imblearn.under_sampling.base.BaseCleaningSampler",
        "imblearn.under_sampling.base.BaseUnderSampler"
    ]
    assert got == expect, got
