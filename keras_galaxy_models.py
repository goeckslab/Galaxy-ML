"""
Galaxy wrapper for using Scikit-learn API with Keras models
"""

import collections
import numpy as np
from keras import backend as K
from keras.models import Sequential, Model
from keras.optimizers import (SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam)
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, clone
from sklearn.externals import six
from sklearn.utils import check_array, check_X_y
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted


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
            out['%s__%s'% (name, key)] = value
            out.update(_get_params_from_dict(value, '%s__%s'% (name, key)))
        else:
            out['%s__%s'% (name, key)] = value

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

    def to_dict(self):
        return _param_to_dict(self.s_param, self.value)


class KerasLayers(BaseEstimator):
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
                raise ValueError("Only layer structure parameters are not searchable!")
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


class BaseKerasModel(BaseEstimator):
    """
    Base class for Galaxy Keras wrapper

    Parameters
    ----------
    layers: KerasLayers object
    optimizer: object
    model: str, 'sequential' or 'functional'
    loss: str, same as Keras `loss`
    epochs: int, from Keras
    batch_size: int, from Keras
    """
    def __init__(self, layers, model_type='sequential', optimizer='sgd', loss='binary_crossentropy',
                 epochs=100, batch_size=None, metrics=[], lr=None, momentum=None,
                 decay=None, nesterov=None, rho=None, epsilon=None, amsgrad=None,
                 beta_1=None, beta_2=None, schedule_decay=None):
        self.layers = layers
        self.optimizer = optimizer
        self.model_type = model_type
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.metrics = metrics
        self.optimizer = optimizer

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
            self.schedule_decay = 0.004 if schedule_decay is None else schedule_decay

        else:
            raise ValueError("Unsupported optimizer type: %s" % optimizer)

    @property
    def _optimizer(self):
        if self.optimizer == 'sgd':
            return SGD(lr=self.lr, momentum=self.momentum,
                       decay=self.decay, nesterov=self.nesterov)

        elif self.optimizer == 'rmsprop':
            return RMSprop(lr=self.lr, rho=self.rho, epsilon=self.epsilon,
                           decay=self.decay)

        elif self.optimizer == 'adagrad':
            return Adagrad(lr=self.lr, epsilon=self.epsilon,
                           decay=self.decay)

        elif self.optimizer == 'adadelta':
            return Adadelta(lr=self.lr, rho=self.rho, epsilon=self.epsilon,
                            decay=self.decay)

        elif self.optimizer == 'adam':
            return Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2,
                        epsilon=self.epsilon, decay=self.decay, amsgrad=self.amsgrad)

        elif self.optimizer == 'adamax':
            return Adamax(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2,
                          epsilon=self.epsilon, decay=self.decay)

        elif self.optimizer == 'nadam':
            return Nadam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2,
                         epsilon=self.epsilon, schedule_decay=self.schedule_decay)

    def _fit(self, X, y, **kwargs):
        config = dict(
            layers = self.layers.layers,
            name = self.layers.name)

        if self.model_type not in ['sequential', 'functional']:
            raise ValueError("Unsupported model type %s" % self.model_type)

        self.model_class_ = Sequential if self.model_type == 'sequential' else Model
        self.model_ = self.model_class_.from_config(config)

        self.model_.compile(loss=self.loss, optimizer=self._optimizer, metrics=self.metrics)

        if self.loss == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

        fit_params = dict(
            epochs = self.epochs,
            batch_size = self.batch_size
        )
        fit_params.update(kwargs)
        
        self.model_.fit(X, y, **fit_params)

        return self


class KerasGClassifier(BaseKerasModel):
    """
    Scikit-learn classifier API for Keras
    """
    def fit(self, X, y, class_weight=None, **kwargs):
        """
        Parameters:
        -----------
        X : array-like, shape `(n_samples, n_features)`
        """
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        check_classification_targets(y)
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

        return super(KerasGClassifier, self)._fit(X, y, **kwargs)

    
    def predict_proba(self, X, **kwargs):
        check_is_fitted(self, 'model_')
        X = check_array(X, accept_sparse=['csc', 'csr'])

        probs = self.model_.predict(X, **kwargs)
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        return probs

    def predict(self, X, **kwargs):
        check_is_fitted(self, 'model_')
        X = check_array(X, accept_sparse=['csc', 'csr'])

        proba = self.model_.predict(X, **kwargs)
        if proba.shape[-1] > 1:
            classes = proba.argmax(axis=-1)
        else:
            classes = (proba > 0.5).astype('int32')
        return self.classes_[classes]

    def score(self, X, y, **kwargs):
        X = check_array(X, accept_sparse=['csc', 'csr'])
        y = np.searchsorted(self.classes_, y)

        if self.loss == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

        outputs = self.model_.evaluate(x, y, **kwargs)
        outputs = to_list(outputs)
        for name, output in zip(self.model_.metrics_names, outputs):
            if name == 'acc':
                return output
        raise ValueError('The model is not configured to compute accuracy. '
                         'You should pass `metrics=["accuracy"]` to '
                         'the `model.compile()` method.')


class KerasGRegressor(BaseKerasModel):
    def fit(self, X, y, **kwargs):
         X, y = check_X_y(X, y, accept_sparse=['csc', 'csr'])
         return super(KerasGRegressor, self)._fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        check_is_fitted(self, 'model_')

        X = check_array(X, accept_sparse=['csc', 'csr'])
        return np.squeeze(self.model_.predict(X, **kwargs), axis=-1)

    def score(self, X, y, **kwargs):
        check_is_fitted(self, 'model_')
        X = check_array(X, accept_sparse=['csc', 'csr'])
        loss = self.model_.evaluate(X, y, **kwargs)
        if isinstance(loss, list):
            return -loss[0]
        return -loss