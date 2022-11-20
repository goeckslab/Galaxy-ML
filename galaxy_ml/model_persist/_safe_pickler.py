
import inspect
import json
import pickle
import pkgutil
import sys
from pathlib import Path

import _compat_pickle

import sklearn

from ..utils import try_get_attr


# handle pickle white list file
WL_FILE = str(Path(__file__).parent.joinpath(
    'pk_whitelist.json').absolute())


class _SafePickler(pickle.Unpickler, object):
    """
    Used to safely deserialize scikit-learn model objects
    Usage:
        eg.: _SafePickler.load(pickled_file_object)
    """
    def __init__(self, file):
        super(_SafePickler, self).__init__(file)
        # load global white list
        with open(WL_FILE, 'r') as f:
            pk_whitelist = json.load(f)

        self.whitelist = pk_whitelist['SK_NAMES'] + \
            pk_whitelist['SKR_NAMES'] + \
            pk_whitelist['XGB_NAMES'] + \
            pk_whitelist['NUMPY_NAMES'] + \
            pk_whitelist['IMBLEARN_NAMES'] + \
            pk_whitelist['MLXTEND_NAMES'] + \
            pk_whitelist['SKOPT_NAMES'] + \
            pk_whitelist['KERAS_NAMES'] + \
            pk_whitelist['GENERAL_NAMES']

        self.bad_names = (
            'and', 'as', 'assert', 'break', 'class', 'continue',
            'def', 'del', 'elif', 'else', 'except', 'exec',
            'finally', 'for', 'from', 'global', 'if', 'import',
            'in', 'is', 'lambda', 'not', 'or', 'pass', 'print',
            'raise', 'return', 'try', 'system', 'while', 'with',
            'True', 'False', 'None', 'eval', 'execfile', '__import__',
            '__package__', '__subclasses__', '__bases__', '__globals__',
            '__code__', '__closure__', '__func__', '__self__', '__module__',
            '__dict__', '__class__', '__call__', '__get__',
            '__getattribute__', '__subclasshook__', '__new__',
            '__init__', 'func_globals', 'func_code', 'func_closure',
            'im_class', 'im_func', 'im_self', 'gi_code', 'gi_frame',
            '__asteval__', 'f_locals', '__mro__')

        # custom module in Galaxy-ML
        self.custom_modules = [
            'keras_galaxy_models',
            'feature_selectors', 'preprocessors',
            'iraps_classifier', 'model_validations']

    # override
    def find_class(self, module, name):
        # balack list first
        if name in self.bad_names:
            raise pickle.UnpicklingError("Global '%s.%s' is forbidden"
                                         % (module, name))

        # custom module in Galaxy-ML
        # compatible with models from versions before 1.0.7.0
        if module in self.custom_modules:
            module = 'galaxy_ml.' + module

        # Load objects serialized in older versions
        # TODO make this deprecate
        if module.startswith('__main__.'):
            module = 'galaxy_ml.' + module[9:]

        if module.startswith('galaxy_ml.'):
            splits = module.split('.')
            if len(splits) > 2:
                module = splits[0] + '.' + splits[1]
            return try_get_attr(module, name)

        if (module, name) in _compat_pickle.NAME_MAPPING:
            module, name = _compat_pickle.NAME_MAPPING[(module, name)]
        elif module in _compat_pickle.IMPORT_MAPPING:
            module = _compat_pickle.IMPORT_MAPPING[module]

        fullname = module + '.' + name

        if fullname not in self.whitelist:
            # raise pickle.UnpicklingError
            raise pickle.UnpicklingError("Global '%s' is forbidden"
                                         % fullname)

        __import__(module, level=0)
        new_global = getattr(sys.modules[module], name)

        assert new_global.__module__ == module
        return new_global


def safe_load_model(file):
    """Load pickled object with `_SafePicker`
    """
    return _SafePickler(file).load()


def gen_pickle_whitelist():
    """ Generate dict and dump to `pk_whitelist.json`.
    """
    rval = {
        'SK_NAMES': [],
        'SKR_NAMES': [],
        'XGB_NAMES': [],
        'IMBLEARN_NAMES': [],
        'MLXTEND_NAMES': [],
        'SKOPT_NAMES': [],
        'NUMPY_NAMES': [],
        'KERAS_NAMES': [],
        'GENERAL_NAMES': []
    }

    sk_submodule_excludes = (
        'exceptions', 'externals', 'clone', 'get_config',
        'set_config', 'config_context', 'show_versions',
        'datasets')
    for submodule in (
        set(sklearn.__all__ + ['_loss']) - set(sk_submodule_excludes)
    ):
        rval['SK_NAMES'].extend(
            find_members('sklearn.' + submodule))

    rval['SKR_NAMES'].extend(find_members('skrebate'))

    for xgb_submodules in ('callback', 'compat', 'core',
                           'sklearn', 'training'):
        rval['XGB_NAMES'].extend(
            find_members('xgboost.' + xgb_submodules))

    rval['IMBLEARN_NAMES'].extend(find_members('imblearn'))

    for mlx_submodules in ('_base', 'classifier', 'regressor',
                           'frequent_patterns', 'cluster',
                           'feature_selection',
                           'feature_extraction',
                           'preprocessing'):
        rval['MLXTEND_NAMES'].extend(
            find_members('mlxtend.' + mlx_submodules))

    rval['SKOPT_NAMES'].extend(find_members('skopt.searchcv'))
    rval['NUMPY_NAMES'].extend([
        "numpy.core.multiarray._reconstruct",
        "numpy.core.multiarray.scalar",
        "numpy.dtype",
        "numpy.float64",
        "numpy.int64",
        "numpy.ma.core._mareconstruct",
        "numpy.ma.core.MaskedArray",
        "numpy.mean",
        "numpy.ndarray",
        "numpy.random.__RandomState_ctor",
        "numpy.random._pickle.__randomstate_ctor"])

    rval['KERAS_NAMES'].extend([
        "keras.engine.sequential.Sequential",
        "keras.engine.sequential.Functional",
        "keras.engine.sequential.Model"
    ])

    rval['GENERAL_NAMES'].extend([
        "_codecs.encode",
        "builtins.object",
        "builtins.bytearray",
        "collections.OrderedDict",
        "copyreg._reconstructor"
    ])

    with open(WL_FILE[:-5] + '_new.json', 'w') as fh:
        json.dump(rval, fh, indent=4)

    return rval


def find_members(module: str, enforce_import: bool = True):
    """ get class and function members, including those from submodules.
    """
    rval = []

    if module not in sys.modules and enforce_import:
        exec(f"import {module}")
    mod = sys.modules[module]

    members = inspect.getmembers(
        mod,
        lambda x: ((inspect.isclass(x) or inspect.isfunction(x))
                   and x.__module__ == module)
    )
    for mem in members:
        rval.append(module + '.' + mem[0])

    if hasattr(mod, '__path__'):
        for submodule in pkgutil.iter_modules(mod.__path__):
            if submodule.name.lower() in ('tests', 'utils'):
                continue
            rval.extend(find_members(module + '.' + submodule.name,
                                     enforce_import=enforce_import))

    return sorted(rval)
