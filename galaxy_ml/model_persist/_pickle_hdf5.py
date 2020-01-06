"""
Utility to persist arbitrary sklearn objects to HDF5

Author: Qiang Gu
Email: guqiang01@gmail.com
Date: 2020-2021

Classes:

    ModelToHDF5
    HDF5ToModel

"""

import sys
import six
import warnings
import types
import numpy
import h5py
import sklearn
import keras
from keras.engine.network import Network
from keras.models import load_model

# reserved keys
_PY_VERSION = '-cpython-'
_NP_VERSION = '-numpy-'
_REDUCE = '-reduce-'
_GLOBAL = '-global-'
_FUNC = '--func-'
_ARGS = '-args-'
_STATE = '-state-'
_KEYS = '-keys-'
_MEMO = '-memo-'
_NP_NDARRAY = '-np_ndarray-'
_DTYPE = '-dtype-'
_VALUES = '-values-'
_NP_DATATYPE = '-np_datatype-'
_DATATYPE = '-datatype-'
_VALUE = '-value-'
_TUPLE = '-tuple-'
_SET = '-set-'
_BYTES = '-bytes-'
_UNICODE = '-unicode-'
_NONE = '-None-'
_BOOL = '-boolean-'
_INT = '-int-'
_FLOAT = '-float-'
_STRING = '-string-'
_BYTEARRAY = '-bytearray-'
_COMPLEX = '-complex-'
_LIST = '-list-'
_DICT = '-dict-'
_MODULE_NAME = '-module_name-'
_OBJ_NAME = '-obj_name-'
_NP_NDARRAY_O = '-np_ndarray_o_type-'
_KERAS_MODEL = '-keras_model-'
_SK_VERSION = '-sklearn_version-'
_KERAS_VERSION = '-keras_version-'

PY_VERSION = sys.version.split(' ')[0]


class HPicklerError(Exception):
    pass


class ModelToHDF5:
    """
    Follow python `pickle`
    Save a scikit-learn/keras model to HDF5 file.
    """
    def __init__(self, verbose=0):
        self.memo = {}
        self.verbose = verbose

    def clear_memo(self):
        """Clears the `memo`
        """
        self.memo.clear()

    def memoize(self, obj):
        """
        Store an object id in the `memo`
        """
        assert id(obj) not in self.memo
        idx = len(self.memo)
        if self.verbose:
            print("Memoize: ", (idx, obj))
        self.memo[id(obj)] = idx, obj

    def dump(self, obj, file_path, mode='w'):
        """
        Main access of object save

        Parameters:
        obj : python object to save
        file_path : str or hdf5 or hdf5 group object
            Target file path in HDF5 format
        """
        try:
            if isinstance(file_path, str):
                file = h5py.File(file_path, mode=mode)

            if not isinstance(file, h5py.Group):
                raise ValueError("The type of file_path, %s, is "
                                 "not supported. Supported types "
                                 "are file path (str), h5py.File "
                                 "and h5py.Group object!"
                                 % (str(file_path)))

            file.attrs[_PY_VERSION] = str(PY_VERSION).encode('utf8')

            np_module = sys.modules.get('numpy')
            if np_module:
                file.attrs[_NP_VERSION] = \
                    str(np_module.__version__).encode('utf8')

            if isinstance(obj, sklearn.base.BaseEstimator):
                file.attrs[_SK_VERSION] = \
                    str(sklearn.__version__).encode('utf8')

            if isinstance(obj, Network):
                file.attrs[_KERAS_VERSION] = \
                    str(keras.__version__).encode('utf8')

            self.save(obj, file)
        finally:
            file.close()

    def save(self, obj, obj_group):

        # Check the `memo``
        x = self.memo.get(id(obj))
        if x:
            obj_group[_MEMO] = x[0]
            return

        # Check type in `dispath` table
        t = type(obj)
        f = self.dispatch.get(t)
        if f:
            f(self, obj, obj_group)
            return

        # Check for a class with a custom metaclass; treat as regular class
        try:
            issc = issubclass(t, type)
        except TypeError:
            issc = 0
        if issc:
            self.save_global(obj, obj_group)
            return

        self.save_reduce(obj, obj_group)

    def save_reduce(self, obj, obj_group):
        """
        Decompose an object using pickle reduce
        """
        if self.verbose:
            print(obj)
        reduce = getattr(obj, "__reduce__", None)
        if reduce:
            rv = reduce()
        else:
            raise HPicklerError(
                "Can't reduce %r object: %r" % (type(obj).__name__, obj))
        assert (type(rv) is tuple),\
            "%s must return a tuple, but got %s" % (reduce, type(rv))

        lenth = len(rv)
        assert (lenth in [2, 3]),\
            ("Reduce tuple is expected to return 2- 3 elements, "
             "but got %d elements" % lenth)

        reduce_group = obj_group.create_group(_REDUCE)

        save = self.save

        func = rv[0]
        assert callable(func), "func from reduce is not callable"
        func_group = reduce_group.create_group(_FUNC)
        save(func, func_group)

        args = rv[1]
        args_group = reduce_group.create_group(_ARGS)
        save(args, args_group)

        if lenth == 3:
            state = rv[2]
            state_group = reduce_group.create_group(_STATE)
            save(state, state_group)

        self.memoize(obj)

    dispatch = {}

    def save_none(self, obj, obj_group):
        obj_group[_NONE] = 'None'

    dispatch[type(None)] = save_none

    def save_bool(self, obj, obj_group):
        obj_group[_BOOL] = obj

    dispatch[bool] = save_bool

    def save_int(self, obj, obj_group):
        obj_group[_INT] = obj

    dispatch[int] = save_int

    def save_float(self, obj, obj_group):
        obj_group[_FLOAT] = obj

    dispatch[float] = save_float

    def save_complex(self, obj, obj_group):
        obj_group[_COMPLEX] = obj

    dispatch[complex] = save_complex

    def save_bytes(self, obj, obj_group):
        obj_group[_BYTES] = obj
        self.memoize(obj)

    dispatch[bytes] = save_bytes

    def save_string(self, obj, obj_group):
        obj_group[_STRING] = obj.encode('utf8')
        self.memoize(obj)

    dispatch[str] = save_string

    # deprecate in py3
    """def save_unicode(self, obj, obj_group):
        obj_group[_UNICODE] = obj.encode('utf8')
        self.memoize(obj)

    dispatch[unicode] = save_unicode"""

    def save_bytearray(self, obj, obj_group):
        obj_group[_BYTEARRAY] = obj
        self.memoize(obj)

    dispatch[bytearray] = save_bytearray

    def save_list(self, obj, obj_group):
        list_group = obj_group.create_group(_LIST)
        for i, e in enumerate(obj):
            group = list_group.create_group(str(i))
            self.save(e, group)

    dispatch[list] = save_list

    def save_tuple(self, obj, obj_group):
        tuple_group = obj_group.create_group(_TUPLE)
        self.save(list(obj), tuple_group)

    dispatch[tuple] = save_tuple

    def save_set(self, obj, obj_group):
        set_group = obj_group.create_group(_SET)
        self.save(list(obj), set_group)

    dispatch[set] = save_set

    def save_dict(self, obj, obj_group):
        dict_group = obj_group.create_group(_DICT)
        _keys = list(obj.keys())
        if len(_keys) == 0:
            return
        _keys.sort()
        dict_group[_KEYS] = [str(k).encode('utf-8') for k in _keys]
        for key in _keys:
            sub_group = dict_group.create_group(key)
            self.save(obj[key], sub_group)

    dispatch[dict] = save_dict

    def save_global(self, obj, obj_group):
        name = getattr(obj, '__name__', None)
        if name is None:
            raise HPicklerError("Can't get global name for object %r"
                                % obj)
        module_name = getattr(obj, '__module__', None)
        if module_name is None:
            raise HPicklerError("Can't get global module name for "
                                "object %r" % obj)

        global_group = obj_group.create_group(_GLOBAL)
        global_group[_MODULE_NAME] = module_name
        global_group[_OBJ_NAME] = name

        self.memoize(obj)

    dispatch[types.FunctionType] = save_global
    dispatch[types.BuiltinFunctionType] = save_global

    def save_np_ndarray(self, obj, obj_group):
        _dtype = obj.dtype
        if _dtype.descr == [('', '|O')]:
            array_group = obj_group.create_group(_NP_NDARRAY_O)
            dtype_group = array_group.create_group(_DTYPE)
            values_group = array_group.create_group(_VALUES)

            self.save(_dtype, dtype_group)
            self.save(obj.tolist(), values_group)
        else:
            obj_group[_NP_NDARRAY] = obj

    dispatch[numpy.ndarray] = save_np_ndarray

    def save_np_datatype(self, obj, obj_group):
        obj_group[_NP_DATATYPE] = obj

    dispatch[numpy.bool_] = save_np_datatype
    dispatch[numpy.int_] = save_np_datatype
    dispatch[numpy.intc] = save_np_datatype
    dispatch[numpy.intp] = save_np_datatype
    dispatch[numpy.int8] = save_np_datatype
    dispatch[numpy.int16] = save_np_datatype
    dispatch[numpy.int32] = save_np_datatype
    dispatch[numpy.int64] = save_np_datatype
    dispatch[numpy.uint8] = save_np_datatype
    dispatch[numpy.uint16] = save_np_datatype
    dispatch[numpy.uint32] = save_np_datatype
    dispatch[numpy.uint64] = save_np_datatype
    dispatch[numpy.float_] = save_np_datatype
    dispatch[numpy.float16] = save_np_datatype
    dispatch[numpy.float32] = save_np_datatype
    dispatch[numpy.float64] = save_np_datatype
    dispatch[numpy.complex_] = save_np_datatype
    dispatch[numpy.complex64] = save_np_datatype
    dispatch[numpy.complex128] = save_np_datatype

    def save_keras_model(self, obj, obj_group):
        keras_group = obj_group.create_group(_KERAS_MODEL)
        obj.save(keras_group)

    dispatch[Network] = save_keras_model


class HDF5ToModel:
    """
    Rebuild model from HDF5 generated by ModelToHDF5.save
    """

    def __init__(self, verbose=0):
        # Store newly-built object
        self.memo = {}
        self.verbose = verbose

    def memoize(self, obj):
        lenth = len(self.memo)
        self.memo[lenth] = obj
        if self.verbose:
            print("Memoize: ", (lenth, obj))

    def load(self, file_path):
        if isinstance(file_path, str):
            data = h5py.File(file_path, 'r')
        if not isinstance(data, h5py.Group):
            raise ValueError("The type of %s is not supported! "
                             "Supported types are file path (str), "
                             "h5py.File and h5py.Group object!"
                             % (str(file_path)))

        if data.attrs[_PY_VERSION].decode('utf8') != PY_VERSION:
            warnings.warn("Trying to load an object from python %s "
                          "when using python %s. This might lead to "
                          "breaking code or invalid results. Use at "
                          "your own risk."
                          % (data[_PY_VERSION].decode('utf8'),
                             PY_VERSION.decode('utf8')))
        return self.load_all(data[_REDUCE], data_name=_REDUCE)

    def load_all(self, data, data_name=None):
        """
        The main method to generate an object from dict data
        """
        if data_name is None:
            data_name = data.name.split('/')[-1]

        f = self.dispatch.get(data_name)
        if f:
            return f(self, data)

        raise HPicklerError("Unsupported data found: %s, data_name: %s"
                            % (str(data), str(data_name)))

    dispatch = {}

    def load_memo(self, data):
        return self.memo[data[()]]

    dispatch[_MEMO] = load_memo

    def load_none(self, data):
        return None

    dispatch[_NONE] = load_none

    def load_int(self, data):
        return int(data[()])

    dispatch[_INT] = load_int

    def load_bool(self, data):
        return bool(data[()])

    dispatch[_BOOL] = load_bool

    def load_float(self, data):
        return float(data[()])

    dispatch[_FLOAT] = load_float

    def load_string(self, data):
        obj = data[()].decode('utf8')
        self.memoize(obj)
        return obj

    dispatch[_STRING] = load_string

    """def load_unicode(self, data):
        obj = data[()].decode('utf8')
        self.memoize(obj)
        return obj

    dispatch[_UNICODE] = load_unicode"""

    def load_bytes(self, data):
        obj = data[()]
        self.memoize(obj)
        return obj

    dispatch[_BYTES] = load_bytes

    def load_complex(self, data):
        return complex(data[()])

    dispatch[_COMPLEX] = load_complex

    def load_bytearray(self, data):
        return bytearray(data[()])

    dispatch[_BYTEARRAY] = load_bytearray

    def load_list(self, data):
        n_elems = len(data.keys())

        rval = []
        for i in range(n_elems):
            elem_group = data[str(i)]
            assert len(elem_group.keys()) == 1, \
                "List element should contains one and only on item!"
            sub_key = list(elem_group.keys())[0]
            rval.append(self.load_all(elem_group[sub_key], data_name=sub_key))

        return rval

    dispatch[_LIST] = load_list

    def load_tuple(self, data):
        rval = self.load_all(data[_LIST], _LIST)
        rval = tuple(rval)
        return rval

    dispatch[_TUPLE] = load_tuple

    def load_set(self, data):
        rval = self.load_all(data[_LIST], _LIST)
        rval = set(rval)
        return rval

    dispatch[_SET] = load_set

    def load_dict(self, data):
        rval = {}
        keys = data.keys()
        if len(keys) == 0:
            return rval
        assert _KEYS in keys
        _keys = data[_KEYS][()]
        for k in _keys:
            k = k.decode('utf8')
            sub_key = list(data[k].keys())[0]
            rval[k] = self.load_all(data[k][sub_key], sub_key)

        return rval

    dispatch[_DICT] = load_dict

    def find_class(self, module, name):
        if module == 'copy_reg' and not six.PY2:
            module = 'copyreg'
        elif module == '__builtin__' and not six.PY2:
            module = 'builtins'
        __import__(module, level=0)
        mod = sys.modules[module]
        return getattr(mod, name)

    def load_global(self, data):
        module = data[_MODULE_NAME][()]
        name = data[_OBJ_NAME][()]
        func = self.find_class(module, name)
        self.memoize(func)
        return func

    dispatch[_GLOBAL] = load_global

    def load_reduce(self, data):
        """
        Build object
        """
        _func = data[_FUNC]
        assert len(_func.keys()) == 1
        sub_key = list(_func.keys())[0]
        func = self.load_all(_func[sub_key], data_name=sub_key)
        assert callable(func), "%r" % func

        _args = data[_ARGS]
        assert len(_args.keys()) == 1
        sub_key = list(_args.keys())[0]
        args = self.load_all(_args[sub_key], data_name=sub_key)

        try:
            obj = args[0].__new__(args[0], * args)
        except:
            obj = func(*args)

        _state = data.get(_STATE)
        if _state:
            assert len(_state.keys()) == 1
            sub_key = list(_state.keys())[0]
            state = self.load_all(_state[sub_key], data_name=sub_key)
            setstate = getattr(obj, "__setstate__", None)
            if setstate:
                setstate(state)
            else:
                assert (type(state) is dict)
                for k, v in state.items():
                    setattr(obj, k, v)

        if self.verbose:
            print(func)
        self.memoize(obj)
        return obj

    dispatch[_REDUCE] = load_reduce

    def load_np_ndarray_o_type(self, data):
        dtype_group = data[_DTYPE]
        assert len(dtype_group.keys()) == 1
        sub_key = list(dtype_group.keys())[0]
        _dtype = self.load_all(dtype_group[sub_key], data_name=sub_key)

        values_group = data[_VALUES]
        assert len(values_group.keys()) == 1
        sub_key = list(values_group.keys())[0]
        _values = self.load_all(values_group[sub_key], data_name=sub_key)

        obj = numpy.array(_values, dtype=_dtype)
        return obj

    dispatch[_NP_NDARRAY_O] = load_np_ndarray_o_type

    def load_np_ndarray(self, data):
        obj = data[()]
        return obj

    dispatch[_NP_NDARRAY] = load_np_ndarray

    def load_np_datatype(self, data):
        obj = data[()]
        return obj

    dispatch[_NP_DATATYPE] = load_np_datatype

    def load_keras_model(self, data):
        obj = load_model(data)
        return obj

    dispatch[_KERAS_MODEL] = load_keras_model


def dump_model_to_h5(obj, file_path, verbose=0):
    """
    Parameters
    ----------
    obj : python object
    file_path : str or hdf5.File or hdf5.Group object
    verbose : 0 or 1
    """
    return ModelToHDF5(verbose=verbose).dump(obj, file_path)


def load_model_from_h5(file_path, verbose=0):
    """
    file_path : str or hdf5.File or hdf5.Group object
    verbose : 0 or 1
    """
    return HDF5ToModel(verbose=verbose).load(file_path)
