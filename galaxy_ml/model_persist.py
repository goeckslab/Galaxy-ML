"""
Utility to persist sklearn objects to JSON

Author: Qiang Gu
Email: guqiang01@gmail.com
Date: 10/8/2018

Classes:

    ModelToDict
    DictToModel

Functions:

    dumpc(object) -> dictionary
    loadc(dictionary) -> object

"""

import sys
import six
import warnings
import types
import numpy

# reserved keys
_PY_VERSION = '-cpython-'
_OBJ = '-object-'
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

PY_VERSION = sys.version.split(' ')[0]


__all__ = ('JPicklerError', 'ModelToDict', 'DictToModel', 'dumpc', 'loadc')


class JPicklerError(Exception):
    pass


class ModelToDict:
    """
    Follow the track of python `pickle`
    Turn a scikit-learn model to a JSON-compatiable dictionary
    """
    def __init__(self):
        self.memo = {}

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
        self.memo[id(obj)] = idx, obj

    def dump(self, obj):
        """
        Main access of object save
        """
        retv = {_PY_VERSION: PY_VERSION}
        np_module = sys.modules.get('numpy')
        if np_module:
            retv[_NP_VERSION] = np_module.__version__
        retv[_OBJ] = self.save(obj)
        return retv

    def save(self, obj):

        # Check the `memo``
        x = self.memo.get(id(obj))
        if x:
            rval = {_MEMO: x[0]}
            return rval

        # Check type in `dispath` table
        t = type(obj)
        f = self.dispatch.get(t)
        if f:
            return f(self, obj)

        # Check for a class with a custom metaclass; treat as regular class
        try:
            issc = issubclass(t, type)
        except TypeError:
            issc = 0
        if issc:
            return self.save_global(obj)

        return self.save_reduce(obj)

    def save_reduce(self, obj):
        """
        Decompose an object using pickle reduce
        """
        reduce = getattr(obj, "__reduce__", None)
        if reduce:
            rv = reduce()
        else:
            raise JPicklerError(
                "Can't reduce %r object: %r" % (type(obj).__name__, obj))
        assert (type(rv) is tuple),\
            "%s must return a tuple, but got %s" % (reduce, type(rv))

        lenth = len(rv)
        assert (lenth in [2, 3]),\
            ("Reduce tuple is expected to return 2- 3 elements, "
             "but got %d elements" % lenth)

        save = self.save

        retv = {}

        func = rv[0]
        assert callable(func), "func from reduce is not callable"
        retv[_FUNC] = save(func)

        args = rv[1]
        assert (type(args) is tuple)
        retv[_ARGS] = {_TUPLE: save(list(args))}

        if lenth == 3:
            state = rv[2]
            retv[_STATE] = save(state)

        self.memoize(obj)
        return {_REDUCE: retv}

    dispatch = {}

    def save_primitive(self, obj):
        return obj

    dispatch[type(None)] = save_primitive
    dispatch[bool] = save_primitive
    dispatch[int] = save_primitive
    if six.PY2:
        dispatch[long] = save_primitive
    dispatch[float] = save_primitive
    dispatch[complex] = save_primitive

    def save_bytes(self, obj):
        print("save_bytes: %s" % type(obj))
        self.memoize(obj)
        return {_BYTES: obj.decode('utf-8')}

    dispatch[bytes] = save_bytes

    def save_string(self, obj):
        self.memoize(obj)
        return obj

    dispatch[str] = save_string

    def save_unicode(self, obj):
        self.memoize(obj)
        return {_UNICODE: obj}

    if six.PY2:
        dispatch[unicode] = save_unicode
    # dispatch[bytearray] = save_primitive

    def save_list(self, obj):
        return [self.save(e) for e in obj]

    dispatch[list] = save_list

    def save_tuple(self, obj):
        aslist = self.save(list(obj))
        return {_TUPLE: aslist}

    dispatch[tuple] = save_tuple

    def save_set(self, obj):
        aslist = self.save(list(obj))
        return {_SET: aslist}

    dispatch[set] = save_set

    def save_dict(self, obj):
        if len(obj) == 0:
            return {}
        newdict = {}
        _keys = list(obj.keys())
        _keys.sort()
        newdict[_KEYS] = _keys
        for k in _keys:
            newdict[k] = self.save(obj[k])
        # self.memoize(obj)
        return newdict

    dispatch[dict] = save_dict

    def save_global(self, obj):
        name = getattr(obj, '__name__', None)
        if name is None:
            raise JPicklerError("Can't get global name for object %r"
                                % obj)
        module_name = getattr(obj, '__module__', None)
        if module_name is None:
            raise JPicklerError("Can't get global module name for "
                                "object %r" % obj)

        newdict = {_GLOBAL: [module_name, name]}
        self.memoize(obj)
        return newdict

    dispatch[types.FunctionType] = save_global
    dispatch[types.BuiltinFunctionType] = save_global

    def save_np_ndarray(self, obj):
        newdict = {}
        newdict[_DTYPE] = self.save(obj.dtype)
        newdict[_VALUES] = self.save(obj.tolist())
        # self.memoize(obj)
        return {_NP_NDARRAY: newdict}

    dispatch[numpy.ndarray] = save_np_ndarray

    def save_np_datatype(self, obj):
        newdict = {}
        newdict[_DATATYPE] = self.save(type(obj))
        newdict[_VALUE] = self.save(obj.item())
        # self.memoize(obj)
        return {_NP_DATATYPE: newdict}

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


class DictToModel:
    """
    Rebuild a scikit-learn model from dict data generated by ModelToDict.save
    """

    def __init__(self):
        # Store newly-built object
        self.memo = {}

    def memoize(self, obj):
        lenth = len(self.memo)
        self.memo[lenth] = obj

    def load(self, data):
        if data[_PY_VERSION] != PY_VERSION:
            warnings.warn("Trying to load an object from python %s "
                          "when using python %s. This might lead to "
                          "breaking code or invalid results. Use at "
                          "your own risk." % (data[_PY_VERSION], PY_VERSION))
        return self.load_all(data[_OBJ])

    def load_all(self, data):
        """
        The main method to generate an object from dict data
        """
        dispatch = self.dispatch

        t = type(data)
        if t is dict:
            if _MEMO in data:
                return self.memo[data[_MEMO]]
            if _BYTES in data:
                return self.load_bytes(data[_BYTES])
            if _REDUCE in data:
                return self.load_reduce(data[_REDUCE])
            if _GLOBAL in data:
                return self.load_global(data[_GLOBAL])
            if _TUPLE in data:
                return self.load_tuple(data[_TUPLE])
            if _SET in data:
                return self.load_set(data[_SET])
            if _NP_NDARRAY in data:
                return self.load_np_ndarray(data[_NP_NDARRAY])
            if _NP_DATATYPE in data:
                return self.load_np_datatype(data[_NP_DATATYPE])
            if _UNICODE in data:
                return self.load_unicode2(data[_UNICODE])
            return self.load_dict(data)
        f = dispatch.get(t)
        if f:
            return f(self, data)
        else:
            raise JPicklerError("Unsupported data found: %s" % str(data))

    dispatch = {}

    def load_primitive(self, data):
        return data

    dispatch[type(None)] = load_primitive
    dispatch[bool] = load_primitive
    dispatch[int] = load_primitive
    if six.PY2:
        dispatch[long] = load_primitive
    dispatch[float] = load_primitive
    dispatch[complex] = load_primitive

    def load_string(self, data):
        self.memoize(data)
        return data

    dispatch[str] = load_string

    def load_unicode(self, data):
        """
        `json.load` loads string as unicode in python 2,
        while some classes don't support unicode, like numpy.dtype
        """
        data = str(data)
        self.memoize(data)
        return data

    if six.PY2:
        dispatch[unicode] = load_unicode

    def load_unicode2(self, data):
        self.memoize(data)
        return data

    def load_bytes(self, data):
        data = data.encode('utf-8')
        self.memoize(data)
        return data

    def load_list(self, data):
        return [self.load_all(e) for e in data]

    dispatch[list] = load_list

    def load_tuple(self, data):
        obj = self.load_all(data)
        # self.memoize(obj)
        return tuple(obj)

    def load_set(self, data):
        obj = self.load_all(data)
        # self.memoize(obj)
        return set(obj)

    def load_dict(self, data):
        if len(data) == 0:
            return {}
        newdict = {}
        _keys = data[_KEYS]
        for k in _keys:
            try:
                v = data[k]
            # JSON dumps non-string key to string
            except KeyError:
                v = data[str(k)]
            newdict[k] = self.load_all(v)
        # self.memoize( newdict )
        return newdict

    def find_class(self, module, name):
        if module == 'copy_reg' and not six.PY2:
            module = 'copyreg'
        elif module == '__builtin__' and not six.PY2:
            module = 'builtins'
        __import__(module, level=0)
        mod = sys.modules[module]
        return getattr(mod, name)

    def load_global(self, data):
        module = data[0]
        name = data[1]
        func = self.find_class(module, name)
        self.memoize(func)
        return func

    def load_reduce(self, data):
        """
        Build object
        """
        _func = data[_FUNC]
        func = self.load_all(_func)
        assert callable(func), "%r" % func

        _args = data[_ARGS][_TUPLE]
        args = tuple(self.load_all(_args))

        try:
            obj = args[0].__new__(args[0], * args)
        except:
            obj = func(*args)

        _state = data.get(_STATE)
        if _state:
            state = self.load_all(_state)
            setstate = getattr(obj, "__setstate__", None)
            if setstate:
                setstate(state)
            else:
                assert (type(state) is dict)
                for k, v in state.items():
                    setattr(obj, k, v)

        self.memoize(obj)
        return obj

    def load_np_ndarray(self, data):
        _dtype = self.load_all(data[_DTYPE])
        _values = self.load_all(data[_VALUES])
        obj = numpy.array(_values, dtype=_dtype)
        # self.memoize(obj)
        return obj

    def load_np_datatype(self, data):
        _datatype = self.load_all(data[_DATATYPE])
        _value = self.load_all(data[_VALUE])
        obj = _datatype(_value)
        # self.memoize(obj)
        return obj


def dumpc(obj):
    return ModelToDict().dump(obj)


def loadc(data):
    return DictToModel().load(data)
