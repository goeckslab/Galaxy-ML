"""
Galaxy wrapper for using Scikit-learn API with Keras models

Author: Qiang Gu
Email: guqiang01@gmail.com
2019 - 2020
"""

import collections
import keras
import numpy as np
import tensorflow as tf
from abc import ABCMeta
from keras import backend as K
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             TensorBoard, RemoteMonitor,
                             ModelCheckpoint, TerminateOnNaN,
                             CSVLogger, ReduceLROnPlateau)
from keras.models import Sequential, Model
from keras.optimizers import (SGD, RMSprop, Adagrad,
                              Adadelta, Adam, Adamax, Nadam)
from keras.utils import to_categorical
from keras.utils.generic_utils import has_arg, to_list
from sklearn.base import (BaseEstimator, ClassifierMixin,
                          RegressorMixin)
from sklearn.externals import six
from sklearn.utils import check_array, check_X_y
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from tensorflow import set_random_seed


__all__ = ('KerasEarlyStopping', 'KerasTensorBoard', 'KerasCSVLogger',
           'KerasLearningRateScheduler', 'KerasRemoteMonitor',
           'KerasModelCheckpoint', 'KerasTerminateOnNaN',
           'check_params', 'SearchParam', 'KerasLayers', 'BaseKerasModel',
           'KerasGClassifier', 'KerasGRegressor', 'KerasGBatchClassifier')


class BaseOptimizer(BaseEstimator):
    """
    Base wrapper for Keras Optimizers
    """
    def get_params(self, deep=True):
        out = super(BaseOptimizer, self).get_params(deep=deep)
        for k, v in six.iteritems(out):
            try:
                out[k] = K.eval(v)
            except AttributeError:
                pass
        return out

    def set_params(self, **params):
        raise NotImplementedError()


class KerasSGD(SGD, BaseOptimizer):
    pass


class KerasRMSprop(RMSprop, BaseOptimizer):
    pass


class KerasAdagrad(Adagrad, BaseOptimizer):
    pass


class KerasAdadelta(Adadelta, BaseOptimizer):
    pass


class KerasAdam(Adam, BaseOptimizer):
    pass


class KerasAdamax(Adamax, BaseOptimizer):
    pass


class KerasNadam(Nadam, BaseOptimizer):
    pass


class KerasEarlyStopping(EarlyStopping, BaseEstimator):
    pass


class KerasLearningRateScheduler(LearningRateScheduler, BaseEstimator):
    pass


class KerasTensorBoard(TensorBoard, BaseEstimator):
    pass


class KerasRemoteMonitor(RemoteMonitor, BaseEstimator):
    pass


class KerasModelCheckpoint(ModelCheckpoint, BaseEstimator):
    pass


class KerasTerminateOnNaN(TerminateOnNaN, BaseEstimator):
    pass


class KerasCSVLogger(CSVLogger, BaseEstimator):
    pass


def _get_params_from_dict(dic, name):
    """
    Genarate search parameters from `model.get_config()`

    Parameter:
    ----------
    dic: dict
    name: str, the name of dict.
    """
    out = {}

    for key, value in six.iteritems(dic):
        if isinstance(value, dict):
            out['%s__%s' % (name, key)] = value
            out.update(_get_params_from_dict(
                value, '%s__%s' % (name, key)))
        else:
            out['%s__%s' % (name, key)] = value

    return out


def _param_to_dict(s, v):
    """
    Turn search param to deep nested dictionary
    """
    rval = {}
    key, dlim, sub_key = s.partition('__')
    if not dlim:
        rval[key] = v
    else:
        rval[key] = _param_to_dict(sub_key, v)
    return rval


def _update_dict(d, u):
    """
    Update value for nested dictionary, but not adding new keys

    Parameters:
    d: dict, the source dictionary
    u: dict, contains value to update
    """
    for k, v in six.iteritems(u):
        if isinstance(v, collections.Mapping):
            d[k] = _update_dict(d[k], v)
        elif k not in d:
            raise KeyError
        else:
            d[k] = v
    return d


def check_params(params, fn):
    """
    Check whether params are valid for function(s)

    Parameter:
    ----------
    params : dict
    fn : function or functions iterables
    """
    if not isinstance(fn, (list, tuple)):
        fn = [fn]
    for p in list(six.iterkeys(params)):
        for f in fn:
            if has_arg(f, p):
                break
        else:
            raise ValueError(
                "{} is not a legal parameter".format(p))


class SearchParam(object):
    """
    Sortable Wrapper class for search parameters
    """
    def __init__(self, s_param, value):
        self.s_param = s_param
        self.value = value

    @property
    def depth(self):
        return len(self.s_param.split('__'))

    @property
    def sort_depth(self):
        if self.depth > 2:
            return 2
        else:
            return self.depth

    def to_dict(self):
        return _param_to_dict(self.s_param, self.value)


class KerasLayers(six.with_metaclass(ABCMeta, BaseEstimator)):
    """
    Parameters:
    -----------
    name: str
    layers: list of dict, the configuration of model
    """
    def __init__(self, name='sequential_1', layers=[]):
        self.name = name
        self.layers = layers

    @property
    def named_layers(self):
        rval = []
        for idx, lyr in enumerate(self.layers):
            named = 'layers_%s_%s' % (str(idx), lyr['class_name'])
            rval.append((named, lyr))

        return rval

    def get_params(self, deep=True):
        """Return parameter names for GridSearch"""
        out = super(KerasLayers, self).get_params(deep=False)

        if not deep:
            return out

        out.update(self.named_layers)
        for name, lyr in self.named_layers:
            out.update(_get_params_from_dict(lyr, name))

        return out

    def set_params(self, **params):

        for key in list(six.iterkeys(params)):
            if not key.startswith('layers'):
                raise ValueError("Only layer structure parameters are "
                                 "not searchable!")
        # 1. replace `layers`
        if 'layers' in params:
            setattr(self, 'layers', params.pop('layers'))

        # 2. replace individual layer
        layers = self.layers
        named_layers = self.named_layers
        names = []
        named_layers_dict = {}
        if named_layers:
            names, _ = zip(*named_layers)
            named_layers_dict = dict(named_layers)
        for name in list(six.iterkeys(params)):
            if '__' not in name:
                for i, layer_name in enumerate(names):
                    if layer_name == name:
                        new_val = params.pop(name)
                        if new_val is None:
                            del layers[i]
                        else:
                            layers[i] = new_val
                        break
                setattr(self, 'layers', layers)

        # 3. replace other layer parameter
        search_params = [SearchParam(k, v) for k, v in six.iteritems(params)]
        search_params = sorted(search_params, key=lambda x: x.depth)

        for param in search_params:
            update = param.to_dict()
            try:
                _update_dict(named_layers_dict, update)
            except KeyError:
                raise ValueError("Invalid parameter %s for estimator %s. "
                                 "Check the list of available parameters "
                                 "with `estimator.get_params().keys()`." %
                                 (param.s_param, self))

        return self


class BaseKerasModel(six.with_metaclass(ABCMeta, BaseEstimator)):
    """
    Base class for Galaxy Keras wrapper

    Parameters
    ----------
    config : dictionary
        from `model.get_config()`

    model_type : str
        'sequential' or 'functional'

    optimizer : str, default 'sgd'
        'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'

    loss : str, default 'binary_crossentropy'
        same as Keras `loss`

    metrics : list of strings, default []

    lr : None or float
        optimizer parameter, default change with `optimizer`

    momentum : None or float
        for optimizer `sgd` only, ignored otherwise

    nesterov : None or bool
        for optimizer `sgd` only, ignored otherwise

    decay : None or float
        optimizer parameter, default change with `optimizer`
    rho : None or float
        optimizer parameter, default change with `optimizer`

    epsilon : None or float
        optimizer parameter, default change with `optimizer`

    amsgrad : None or bool
        for optimizer `adam` only, ignored otherwise

    beta_1 : None or float
        optimizer parameter, default change with `optimizer`

    beta_2 : None or float
        optimizer parameter, default change with `optimizer`

    schedule_decay : None or float
        optimizer parameter, default change with `optimizer`

    epochs : int
        fit_param from Keras

    batch_size : int
        fit_param, from Keras

    callbacks : None or list of dict
        fit_param, each dict contains one type of callback configuration.
        e.g. {"callback_selection":
                {"callback_type": "EarlyStopping",
                 "monitor": "val_loss"
                 "baseline": None,
                 "min_delta": 0.0,
                 "patience": 10,
                 "mode": "auto",
                 "restore_best_weights": False}}

    validation_data : None or tuple of arrays, (X_test, y_test)
        fit_param

    seed : None or int, default 0
        backend random seed
    """
    def __init__(self, config, model_type='sequential',
                 optimizer='sgd', loss='binary_crossentropy',
                 metrics=[], lr=None, momentum=None, decay=None,
                 nesterov=None, rho=None, epsilon=None, amsgrad=None,
                 beta_1=None, beta_2=None, schedule_decay=None, epochs=1,
                 batch_size=None, seed=0, callbacks=None,
                 validation_data=None, **fit_params):
        self.config = config
        self.model_type = model_type
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.callbacks = callbacks
        self.validation_data = validation_data
        self.fit_params = fit_params
        # TODO support compile parameters

        check_params(fit_params, Model.fit)

        if self.optimizer == 'sgd':
            self.lr = 0.01 if lr is None else lr
            self.momentum = 0 if momentum is None else momentum
            self.decay = 0 if decay is None else decay
            self.nesterov = False if nesterov is None else nesterov

        elif self.optimizer == 'rmsprop':
            self.lr = 0.001 if lr is None else lr
            self.rho = 0.9 if rho is None else rho
            self.epsilon = None if epsilon is None else epsilon
            self.decay = 0 if decay is None else decay

        elif self.optimizer == 'adagrad':
            self.lr = 0.01 if lr is None else lr
            self.epsilon = None if epsilon is None else epsilon
            self.decay = 0 if decay is None else decay

        elif self.optimizer == 'adadelta':
            self.lr = 1.0 if lr is None else lr
            self.rho = 0.95 if rho is None else rho
            self.epsilon = None if epsilon is None else epsilon
            self.decay = 0 if decay is None else decay

        elif self.optimizer == 'adam':
            self.lr = 0.001 if lr is None else lr
            self.beta_1 = 0.9 if beta_1 is None else beta_1
            self.beta_2 = 0.999 if beta_2 is None else beta_2
            self.epsilon = None if epsilon is None else epsilon
            self.decay = 0 if decay is None else decay
            self.amsgrad = False if amsgrad is None else amsgrad

        elif self.optimizer == 'adamax':
            self.lr = 0.002 if lr is None else lr
            self.beta_1 = 0.9 if beta_1 is None else beta_1
            self.beta_2 = 0.999 if beta_2 is None else beta_2
            self.epsilon = None if epsilon is None else epsilon
            self.decay = 0 if decay is None else decay

        elif self.optimizer == 'nadam':
            self.lr = 0.002 if lr is None else lr
            self.beta_1 = 0.9 if beta_1 is None else beta_1
            self.beta_2 = 0.999 if beta_2 is None else beta_2
            self.epsilon = None if epsilon is None else epsilon
            self.schedule_decay = 0.004 if schedule_decay is None\
                else schedule_decay

        else:
            raise ValueError("Unsupported optimizer type: %s" % optimizer)

    @property
    def _optimizer(self):
        if self.optimizer == 'sgd':
            if not hasattr(self, 'momentum'):
                self.momentum = 0
            if not hasattr(self, 'decay'):
                self.decay = 0
            if not hasattr(self, 'nesterov'):
                self.nesterov = False
            return SGD(lr=self.lr, momentum=self.momentum,
                       decay=self.decay, nesterov=self.nesterov)

        elif self.optimizer == 'rmsprop':
            if not hasattr(self, 'rho'):
                self.rho = 0.9
            if not hasattr(self, 'epsilon'):
                self.epsilon = None
            if not hasattr(self, 'decay'):
                self.decay = 0
            return RMSprop(lr=self.lr, rho=self.rho,
                           epsilon=self.epsilon, decay=self.decay)

        elif self.optimizer == 'adagrad':
            if not hasattr(self, 'epsilon'):
                self.epsilon = None
            if not hasattr(self, 'decay'):
                self.decay = 0
            return Adagrad(lr=self.lr, epsilon=self.epsilon,
                           decay=self.decay)

        elif self.optimizer == 'adadelta':
            if not hasattr(self, 'rho'):
                self.rho = 0.95
            if not hasattr(self, 'epsilon'):
                self.epsilon = None
            if not hasattr(self, 'decay'):
                self.decay = 0
            return Adadelta(lr=self.lr, rho=self.rho,
                            epsilon=self.epsilon,
                            decay=self.decay)

        elif self.optimizer == 'adam':
            if not hasattr(self, 'beta_1'):
                self.beta_1 = 0.9
            if not hasattr(self, 'beta_2'):
                self.beta_2 = 0.999
            if not hasattr(self, 'decay'):
                self.decay = 0
            if not hasattr(self, 'epsilon'):
                self.epsilon = None
            if not hasattr(self, 'amsgrad'):
                self.amsgrad = False
            return Adam(lr=self.lr, beta_1=self.beta_1,
                        beta_2=self.beta_2, epsilon=self.epsilon,
                        decay=self.decay, amsgrad=self.amsgrad)

        elif self.optimizer == 'adamax':
            if not hasattr(self, 'beta_1'):
                self.beta_1 = 0.9
            if not hasattr(self, 'beta_2'):
                self.beta_2 = 0.999
            if not hasattr(self, 'decay'):
                self.decay = 0
            if not hasattr(self, 'epsilon'):
                self.epsilon = None
            return Adamax(lr=self.lr, beta_1=self.beta_1,
                          beta_2=self.beta_2, epsilon=self.epsilon,
                          decay=self.decay)

        elif self.optimizer == 'nadam':
            if not hasattr(self, 'beta_1'):
                self.beta_1 = 0.9
            if not hasattr(self, 'beta_2'):
                self.beta_2 = 0.999
            if not hasattr(self, 'epsilon'):
                self.epsilon = None
            if not hasattr(self, 'schedule_decay'):
                self.schedule_decay = 0.004
            return Nadam(lr=self.lr, beta_1=self.beta_1,
                         beta_2=self.beta_2, epsilon=self.epsilon,
                         schedule_decay=self.schedule_decay)

    @property
    def named_layers(self):
        rval = []
        for idx, lyr in enumerate(self.config['layers']):
            class_name = lyr['class_name']
            if class_name in ['Model', 'Sequential']:
                raise ValueError("Model layers are not supported yet!")
            named = 'layers_%s_%s' % (str(idx), class_name)
            rval.append((named, lyr))

        return rval

    @property
    def _callbacks(self):
        """ return list of callback objects from parameters.
        suppose correct input format.

        Notes
        -----
        For `filepath`, `log_dir`, `filename`,
        if None, `os.getcwd()` is used.
        """
        if not self.callbacks:
            return None

        callbacks = []
        for cb in self.callbacks:
            params = cb['callback_selection']
            callback_type = params.pop('callback_type')

            curr_dir = __import__('os').getcwd()

            if callback_type == 'None':
                continue
            elif callback_type == 'ModelCheckpoint':
                if not params.get('filepath', None):
                    params['filepath'] = \
                        __import__('os').path.join(curr_dir, 'weights.hdf5')
            elif callback_type == 'TensorBoard':
                if not params.get('log_dir', None):
                    params['log_dir'] = \
                        __import__('os').path.join(curr_dir, 'logs')
            elif callback_type == 'CSVLogger':
                if not params:
                    params['filename'] = \
                        __import__('os').path.join(curr_dir, 'log.csv')
                    params['separator'] = '\t'
                    params['append'] = True

            klass = getattr(keras.callbacks, callback_type)
            obj = klass(**params)
            callbacks.append(obj)

        if not callbacks:
            return None
        return callbacks

    def _fit(self, X, y, **kwargs):
        config = self.config

        if self.model_type not in ['sequential', 'functional']:
            raise ValueError("Unsupported model type %s" % self.model_type)

        if self.model_type == 'sequential':
            self.model_class_ = Sequential
        else:
            self.model_class_ = Model

        self.model_ = self.model_class_.from_config(
            config,
            custom_objects=dict(tf=tf))

        self.model_.compile(loss=self.loss, optimizer=self._optimizer,
                            metrics=self.metrics)

        if self.loss == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

        fit_params = self.fit_params
        callbacks = self._callbacks
        validation_data = self.validation_data
        fit_params.update(dict(epochs=self.epochs,
                               batch_size=self.batch_size,
                               callbacks=callbacks,
                               validation_data=validation_data))
        fit_params.update(kwargs)

        # set tensorflow random seed
        if self.seed is not None and K.backend() == 'tensorflow':
            set_random_seed(self.seed)

        self.model_.fit(X, y, **fit_params)

        return self

    def get_params(self, deep=True):
        """Return parameter names for GridSearch"""
        # call self._optimizer to activate hidden attributes
        self._optimizer
        out = super(BaseKerasModel, self).get_params(deep=deep)

        if not deep:
            return out

        out.update(self.named_layers)
        for name, lyr in self.named_layers:
            out.update(_get_params_from_dict(lyr, name))

        return out

    def set_params(self, **params):
        """
        """
        valid_params = self.get_params(deep=False)
        # 1. replace `config`
        if 'config' in params:
            setattr(self, 'config', params.pop('config'))

        # 2. replace individual layer or non-layer parameters
        named_layers = self.named_layers
        names = []
        named_layers_dict = {}
        if named_layers:
            names, _ = zip(*named_layers)
            named_layers_dict = dict(named_layers)
        for name in list(six.iterkeys(params)):
            if '__' not in name:
                for i, layer_name in enumerate(names):
                    # replace layer
                    if layer_name == name:
                        new_val = params.pop(name)
                        if new_val is None:
                            del self.config['layers'][i]
                        else:
                            self.config['layers'][i] = new_val
                        break
                else:
                    # replace non-layer parameter
                    if name not in valid_params:
                        raise ValueError(
                            "Invalid parameter %s for estimator %s. "
                            "Check the list of available parameters "
                            "with `estimator.get_params().keys()`."
                            % (name, self))
                    setattr(self, name, params.pop(name))

            elif not name.startswith('layers'):
                # suppose all other parameters are layers parameters,
                # raise error otherwise
                raise ValueError("Invalid parameter %s for estimator %s. "
                                 "Check the list of available parameters "
                                 "with `estimator.get_params().keys()`." %
                                 (name, self))

        # 3. replace layer parameter
        search_params = [SearchParam(k, v) for k, v in six.iteritems(params)]
        search_params = sorted(search_params, key=lambda x: x.depth)

        for param in search_params:
            update = param.to_dict()
            try:
                _update_dict(named_layers_dict, update)
            except KeyError:
                raise ValueError("Invalid parameter %s for estimator %s. "
                                 "Check the list of available parameters "
                                 "with `estimator.get_params().keys()`." %
                                 (param.s_param, self))

        return self

    def to_json(self):
        if hasattr(self, 'model_'):
            # fitted
            return self.model_.to_json()
        else:
            config = self.config

            if self.model_type not in ['sequential', 'functional']:
                raise ValueError("Unsupported model type %s" % self.model_type)

            if self.model_type == 'sequential':
                model_class_ = Sequential
            else:
                model_class_ = Model

            model_ = model_class_.from_config(
                config,
                custom_objects=dict(tf=tf))

            return model_.to_json()

    def save_weights(self, filepath, overwrite=True):
        """Dumps all layer weights to a HDF5 file.

        parameters
        ----------
        filepath : str
            path to the file to save the weights to.

        overwrite : bool, default is True
            Whether to silently overwrite any existing file at the
            target location, or provide the user with a manual prompt.
        """
        if not hasattr(self, 'model_'):
            raise ValueError("Keras model is not fitted. No weights to save!")

        self.model_.save_weights(filepath, overwrite=overwrite)

    def load_weights(self, filepath, by_name=False,
                     skip_mismatch=False, reshape=False):
        """Loads all layer weights from a HDF5 save file.

        parameters
        ----------
        filepath : str
            path to the weights file to load.

        by_name: Bool
            whether to load weights by name or by topological order.

        skip_mismatch : Boolean
            whether to skip loading of layers where there is a mismatch
            in the number of weights, or a mismatch in the shape of the
            weight (only valid when `by_name`=True).

        reshape : Reshape weights to fit the layer when the correct number
            of weight arrays is present but their shape does not match.
        """
        config = self.config

        if self.model_type not in ['sequential', 'functional']:
            raise ValueError("Unsupported model type %s" % self.model_type)

        if self.model_type == 'sequential':
            self.model_class_ = Sequential
        else:
            self.model_class_ = Model

        self.model_ = self.model_class_.from_config(
            config,
            custom_objects=dict(tf=tf))

        self.model_.load_weights(filepath, by_name=by_name,
                                 skip_mismatch=skip_mismatch,
                                 reshape=reshape)


class KerasGClassifier(BaseKerasModel, ClassifierMixin):
    """
    Scikit-learn classifier API for Keras
    """
    def fit(self, X, y, class_weight=None, **kwargs):
        """
        Parameters:
        -----------
        X : array-like, shape `(n_samples, feature_arrays)`
        """
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'], allow_nd=True)
        check_classification_targets(y)
        check_params(kwargs, Model.fit)

        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            self.classes_ = np.unique(y)
            y = np.searchsorted(self.classes_, y)
        else:
            raise ValueError('Invalid shape for y: ' + str(y.shape))
        self.n_classes_ = len(self.classes_)

        kwargs.update({'class_weight': class_weight})

        return super(KerasGClassifier, self)._fit(X, y, **kwargs)

    def predict_proba(self, X, **kwargs):
        check_is_fitted(self, 'model_')
        X = check_array(X, accept_sparse=['csc', 'csr'], allow_nd=True)
        check_params(kwargs, Model.predict)

        probs = self.model_.predict(X, **kwargs)
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        return probs

    def predict(self, X, **kwargs):
        check_is_fitted(self, 'model_')
        X = check_array(X, accept_sparse=['csc', 'csr'], allow_nd=True)
        check_params(kwargs, Model.predict)

        proba = self.model_.predict(X, **kwargs)
        if proba.shape[-1] > 1:
            classes = proba.argmax(axis=-1)
        else:
            classes = (proba > 0.5).astype('int32')
        return self.classes_[classes]

    def score(self, X, y, **kwargs):
        X = check_array(X, accept_sparse=['csc', 'csr'], allow_nd=True)
        y = np.searchsorted(self.classes_, y)
        check_params(kwargs, Model.evaluate)

        if self.loss == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

        outputs = self.model_.evaluate(X, y, **kwargs)
        outputs = to_list(outputs)
        for name, output in zip(self.model_.metrics_names, outputs):
            if name == 'acc':
                return output

        raise ValueError('The model is not configured to compute accuracy. '
                         'You should pass `metrics=["accuracy"]` to '
                         'the `model.compile()` method.')


class KerasGRegressor(BaseKerasModel, RegressorMixin):
    """
    Scikit-learn API wrapper for Keras regressor
    """
    def fit(self, X, y, **kwargs):
        X, y = check_X_y(X, y, accept_sparse=['csc', 'csr'], allow_nd=True)
        check_params(kwargs, Model.fit)

        return super(KerasGRegressor, self)._fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        check_is_fitted(self, 'model_')
        X = check_array(X, accept_sparse=['csc', 'csr'], allow_nd=True)
        check_params(kwargs, Model.predict)

        return np.squeeze(self.model_.predict(X, **kwargs), axis=-1)

    def score(self, X, y, **kwargs):
        check_is_fitted(self, 'model_')
        X = check_array(X, accept_sparse=['csc', 'csr'], allow_nd=True)
        check_params(kwargs, Model.evaluate)

        loss = self.model_.evaluate(X, y, **kwargs)
        if isinstance(loss, list):
            return -loss[0]
        return -loss


class KerasGBatchClassifier(KerasGClassifier):
    """
    keras classifier with batch generator

    Parameters
    ----------
    config : dictionary
        from `model.get_config()`

    train_batch_generator: instance of batch generator

    train_batch_generator: instance of batch generator (default=None)
        if None, same as train_batch_generator

    model_type : str
        'sequential' or 'functional'

    optimizer : str, default 'sgd'
        'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'

    loss : str, default 'binary_crossentropy'
        same as Keras `loss`

    metrics : list of strings, default []

    lr : None or float
        optimizer parameter, default change with `optimizer`

    momentum : None or float
        for optimizer `sgd` only, ignored otherwise

    nesterov : None or bool
        for optimizer `sgd` only, ignored otherwise

    decay : None or float
        optimizer parameter, default change with `optimizer`
    rho : None or float
        optimizer parameter, default change with `optimizer`

    epsilon : None or float
        optimizer parameter, default change with `optimizer`

    amsgrad : None or bool
        for optimizer `adam` only, ignored otherwise

    beta_1 : None or float
        optimizer parameter, default change with `optimizer`

    beta_2 : None or float
        optimizer parameter, default change with `optimizer`

    schedule_decay : None or float
        optimizer parameter, default change with `optimizer`

    epochs : int
        fit_param from Keras

    batch_size : int
        fit_param, from Keras

    callbacks : None or list of dict
        each dict contains one type of callback configuration.
        e.g. {"callback_selection":
                {"callback_type": "EarlyStopping",
                 "monitor": "val_loss"
                 "baseline": None,
                 "min_delta": 0.0,
                 "patience": 10,
                 "mode": "auto",
                 "restore_best_weights": False}}

    validation_data : None or tuple of arrays, (X_test, y_test)
        fit_param

    seed : None or int, default 0
        backend random seed
    """
    def __init__(self, config, train_batch_generator,
                 predict_batch_generator=None,
                 model_type='sequential', optimizer='sgd',
                 loss='binary_crossentropy', metrics=[], lr=None,
                 momentum=None, decay=None, nesterov=None, rho=None,
                 epsilon=None, amsgrad=None, beta_1=None,
                 beta_2=None, schedule_decay=None, epochs=1,
                 batch_size=None, seed=0, n_jobs=1,
                 callbacks=None, validation_data=None,
                 **fit_params):
        super(KerasGBatchClassifier, self).__init__(
            config, model_type='sequential', optimizer='sgd',
            loss='binary_crossentropy', metrics=[], lr=None,
            momentum=None, decay=None, nesterov=None, rho=None,
            epsilon=None, amsgrad=None, beta_1=None, beta_2=None,
            schedule_decay=None, epochs=1, batch_size=None,
            seed=0, callbacks=callbacks,
            validation_data=validation_data, **fit_params)
        self.train_batch_generator = train_batch_generator
        self.predict_batch_generator = predict_batch_generator
        self.n_jobs = n_jobs

    def fit(self, X, y, class_weight=None, sample_weight=None, **kwargs):
        """ fit the model
        """
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'], allow_nd=True)
        check_classification_targets(y)
        check_params(kwargs, Model.fit)

        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            self.classes_ = np.unique(y)
            y = np.searchsorted(self.classes_, y)
        else:
            raise ValueError('Invalid shape for y: ' + str(y.shape))
        self.n_classes_ = len(self.classes_)

        if class_weight is not None:
            kwargs['class_weight'] = class_weight

        config = self.config

        if self.model_type not in ['sequential', 'functional']:
            raise ValueError("Unsupported model type %s" % self.model_type)

        if self.model_type == 'sequential':
            self.model_class_ = Sequential
        else:
            self.model_class_ = Model

        self.model_ = self.model_class_.from_config(
            config,
            custom_objects=dict(tf=tf))

        self.model_.compile(loss=self.loss, optimizer=self._optimizer,
                            metrics=self.metrics)

        if self.loss == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

        fit_params = self.fit_params
        batch_size = self.batch_size
        epochs = self.epochs
        n_jobs = self.n_jobs
        validation_data = self.validation_data

        fit_params.update(dict(
            steps_per_epoch=X.shape[0]/batch_size,
            epochs=epochs,
            workers=n_jobs,
            use_multiprocessing=n_jobs > 1,
            validation_data=validation_data))

        # kwargs from function `fit ` override object initiation values.
        fit_params.update(kwargs)

        # set tensorflow random seed
        if self.seed is not None and K.backend() == 'tensorflow':
            set_random_seed(self.seed)

        self.model_.fit_generator(
            self.train_batch_generator.flow(X, y, batch_size=batch_size,
                                            sample_weight=sample_weight),
            **fit_params)

        return self

    def predict_proba(self, X, **kwargs):
        check_is_fitted(self, 'model_')
        X = check_array(X, accept_sparse=['csc', 'csr'], allow_nd=True)
        check_params(kwargs, Model.predict)

        if self.predict_batch_generator is None:
            predict_batch_generator = self.train_batch_generator
        else:
            predict_batch_generator = self.predict_batch_generator

        batch_size = self.batch_size
        n_jobs = self.n_jobs
        probs = self.model_.predict_generator(
            predict_batch_generator.flow(X, batch_size=batch_size),
            n_jobs=n_jobs,  use_multiprocessing=True if n_jobs > 1 else False,
            **kwargs)

        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        return probs

    def score(self, X, y, **kwargs):
        X = check_array(X, accept_sparse=['csc', 'csr'], allow_nd=True)
        y = np.searchsorted(self.classes_, y)
        check_params(kwargs, Model.evaluate)

        if self.loss == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

        if self.predict_batch_generator is None:
            predict_batch_generator = self.train_batch_generator
        else:
            predict_batch_generator = self.predict_batch_generator

        n_jobs = self.n_jobs
        batch_size = self.batch_size
        outputs = self.model_.evaluate_generator(
            predict_batch_generator.flow(X, y, batch_size=batch_size),
            n_jobs=n_jobs,  use_multiprocessing=True if n_jobs > 1 else False,
            **kwargs)

        outputs = to_list(outputs)
        for name, output in zip(self.model_.metrics_names, outputs):
            if name == 'acc':
                return output

        raise ValueError('The model is not configured to compute accuracy. '
                         'You should pass `metrics=["accuracy"]` to '
                         'the `model.compile()` method.')
