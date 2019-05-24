import sys
import os
galaxy_ml_folder = os.path.dirname(os.path.dirname(__file__))
sys.path.append(galaxy_ml_folder)

from utils import try_get_attr
from nose.tools import raises


def test_try_get_attr_1():
	try_get_attr('preprocessors', 'Z_RandomOverSampler')
	try_get_attr('preprocessors', '_get_quantiles')
	try_get_attr('preprocessors', 'check_X_y', check_def=False)


@raises(NameError)
def test_try_get_attr_2():
	try_get_attr('sklearn.utils', 'check_X_y', check_def=False)
	try_get_attr('preprocessors', 'check_X_y', check_def=True)

