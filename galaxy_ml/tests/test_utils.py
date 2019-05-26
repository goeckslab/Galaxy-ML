import sys
import os

from nose.tools import raises

galaxy_ml_folder = os.path.dirname(os.path.dirname(__file__))
sys.path.append(galaxy_ml_folder)
from utils import try_get_attr


def test_try_get_attr_1():
    try_get_attr('preprocessors', 'Z_RandomOverSampler')
    try_get_attr('iraps_classifier', 'IRAPSClassifier')


@raises(NameError)
def test_try_get_attr_2():
    try_get_attr('sklearn.utils', 'check_X_y')
    try_get_attr('preprocessors', 'check_X_y')
    try_get_attr('preprocessors', '_get_quantiles')

