import sys
import os
from nose.tools import raises
from galaxy_ml.utils import try_get_attr


def test_try_get_attr_1():
    try_get_attr('galaxy_ml.preprocessors', 'Z_RandomOverSampler')
    try_get_attr('galaxy_ml.iraps_classifier', 'IRAPSClassifier')


@raises(NameError)
def test_try_get_attr_2():
    try_get_attr('sklearn.utils', 'check_X_y')
    try_get_attr('galaxy_ml.preprocessors', 'check_X_y')
    try_get_attr('galaxy_ml.preprocessors', '_get_quantiles')
