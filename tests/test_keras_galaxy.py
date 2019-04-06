import numpy as np
import pandas as pd
import warnings
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras_galaxy_models import (_get_params_from_dict, _param_to_dict, _update_dict,
                                 check_params, SearchParam, KerasLayers,
                                 BaseKerasModel, KerasGClassifier, KerasGRegressor)
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from tensorflow import set_random_seed

warnings.simplefilter('ignore')

np.random.seed(1)
set_random_seed(8888)

model = Sequential()
model.add(Dense(64))
model.add(Activation('tanh'))
model.add(Activation('tanh'))
model.add(Dense(32))

df = pd.read_csv('./test-data/pima-indians-diabetes.csv', sep=',')
X = df.iloc[:, 0:8].values.astype(float)
y = df.iloc[:, 8].values

train_model = Sequential()
train_model.add(Dense(12, input_dim=8, activation='relu'))
train_model.add(Dense(1, activation='sigmoid'))


d = {'name': 'sequential_1',
    'layers': [{'class_name': 'Dense',
        'config': {'name': 'dense_1',
        'trainable': True,
        'units': 64,
        'activation': 'linear',
        'use_bias': True,
        'kernel_initializer': {'class_name': 'VarianceScaling',
            'config': {'scale': 1.0,
            'mode': 'fan_avg',
            'distribution': 'uniform',
            'seed': None}},
        'bias_initializer': {'class_name': 'Zeros', 'config': {}},
        'kernel_regularizer': None,
        'bias_regularizer': None,
        'activity_regularizer': None,
        'kernel_constraint': None,
        'bias_constraint': None}},
        {'class_name': 'Activation',
        'config': {'name': 'activation_1',
        'trainable': True,
        'activation': 'tanh'}},
        {'class_name': 'Activation',
        'config': {'name': 'activation_2',
        'trainable': True,
        'activation': 'tanh'}},
        {'class_name': 'Dense',
        'config': {'name': 'dense_2',
        'trainable': True,
        'units': 32,
        'activation': 'linear',
        'use_bias': True,
        'kernel_initializer': {'class_name': 'VarianceScaling',
            'config': {'scale': 1.0,
            'mode': 'fan_avg',
            'distribution': 'uniform',
            'seed': None}},
        'bias_initializer': {'class_name': 'Zeros', 'config': {}},
        'kernel_regularizer': None,
        'bias_regularizer': None,
        'activity_regularizer': None,
        'kernel_constraint': None,
        'bias_constraint': None}}]
}


def test_get_params_from_dict():
    got = list(_get_params_from_dict(d['layers'][0], 'layers_0_Dense').keys())
    expect = ['layers_0_Dense__class_name', 'layers_0_Dense__config', 
              'layers_0_Dense__config__name', 'layers_0_Dense__config__trainable',
              'layers_0_Dense__config__units', 'layers_0_Dense__config__activation',
              'layers_0_Dense__config__use_bias', 'layers_0_Dense__config__kernel_initializer',
              'layers_0_Dense__config__kernel_initializer__class_name',
              'layers_0_Dense__config__kernel_initializer__config',
              'layers_0_Dense__config__kernel_initializer__config__scale',
              'layers_0_Dense__config__kernel_initializer__config__mode',
              'layers_0_Dense__config__kernel_initializer__config__distribution',
              'layers_0_Dense__config__kernel_initializer__config__seed',
              'layers_0_Dense__config__bias_initializer',
              'layers_0_Dense__config__bias_initializer__class_name',
              'layers_0_Dense__config__bias_initializer__config',
              'layers_0_Dense__config__kernel_regularizer',
              'layers_0_Dense__config__bias_regularizer',
              'layers_0_Dense__config__activity_regularizer',
              'layers_0_Dense__config__kernel_constraint',
              'layers_0_Dense__config__bias_constraint']
    assert got == expect, got


def test_param_to_dict():
    param = {'layers_0_Dense__config__kernel_initializer__config__distribution': 'uniform'}
    key = list(param.keys())[0]
    got = _param_to_dict(key, param[key])

    expect = {'layers_0_Dense': {'config': {'kernel_initializer': {'config': {'distribution': 'uniform'}}}}}
    assert got == expect, got


def test_update_dict():
    config = model.get_config()
    layers=config['layers']
    d = layers[0]
    u = {'config': {'kernel_initializer': {'config': {'distribution': 'random_uniform'}}}}
    got = _update_dict(d, u)
    
    expect = {'class_name': 'Dense', 
              'config': {'name': 'dense_1',
                         'trainable': True,
                         'units': 64,
                         'activation': 'linear',
                         'use_bias': True,
                         'kernel_initializer': {'class_name': 'VarianceScaling',
                                                'config': {'scale': 1.0,
                                                           'mode': 'fan_avg',
                                                           'distribution': 'random_uniform',
                                                           'seed': None}},
                         'bias_initializer': {'class_name': 'Zeros', 
                                              'config': {}},
                         'kernel_regularizer': None,
                         'bias_regularizer': None,
                         'activity_regularizer': None,
                         'kernel_constraint': None,
                         'bias_constraint': None}}
    assert got == expect, got


def test_get_params_keras_layers():
    config = model.get_config()
    layers = KerasLayers(name=config['name'], layers=config['layers'])
    got = list(layers.get_params().keys())
    expect = ['layers', 'name', 'layers_0_Dense', 'layers_1_Activation',
              'layers_2_Activation', 'layers_3_Dense', 'layers_0_Dense__class_name',
              'layers_0_Dense__config', 'layers_0_Dense__config__name',
              'layers_0_Dense__config__trainable', 'layers_0_Dense__config__units',
              'layers_0_Dense__config__activation', 'layers_0_Dense__config__use_bias',
              'layers_0_Dense__config__kernel_initializer',
              'layers_0_Dense__config__kernel_initializer__class_name',
              'layers_0_Dense__config__kernel_initializer__config',
              'layers_0_Dense__config__kernel_initializer__config__scale',
              'layers_0_Dense__config__kernel_initializer__config__mode',
              'layers_0_Dense__config__kernel_initializer__config__distribution',
              'layers_0_Dense__config__kernel_initializer__config__seed',
              'layers_0_Dense__config__bias_initializer',
              'layers_0_Dense__config__bias_initializer__class_name',
              'layers_0_Dense__config__bias_initializer__config',
              'layers_0_Dense__config__kernel_regularizer',
              'layers_0_Dense__config__bias_regularizer',
              'layers_0_Dense__config__activity_regularizer',
              'layers_0_Dense__config__kernel_constraint',
              'layers_0_Dense__config__bias_constraint',
              'layers_1_Activation__class_name',
              'layers_1_Activation__config', 'layers_1_Activation__config__name',
              'layers_1_Activation__config__trainable',
              'layers_1_Activation__config__activation',
              'layers_2_Activation__class_name', 'layers_2_Activation__config',
              'layers_2_Activation__config__name', 'layers_2_Activation__config__trainable',
              'layers_2_Activation__config__activation', 'layers_3_Dense__class_name',
              'layers_3_Dense__config', 'layers_3_Dense__config__name',
              'layers_3_Dense__config__trainable', 'layers_3_Dense__config__units',
              'layers_3_Dense__config__activation', 'layers_3_Dense__config__use_bias',
              'layers_3_Dense__config__kernel_initializer',
              'layers_3_Dense__config__kernel_initializer__class_name',
              'layers_3_Dense__config__kernel_initializer__config',
              'layers_3_Dense__config__kernel_initializer__config__scale',
              'layers_3_Dense__config__kernel_initializer__config__mode',
              'layers_3_Dense__config__kernel_initializer__config__distribution',
              'layers_3_Dense__config__kernel_initializer__config__seed',
              'layers_3_Dense__config__bias_initializer',
              'layers_3_Dense__config__bias_initializer__class_name',
              'layers_3_Dense__config__bias_initializer__config',
              'layers_3_Dense__config__kernel_regularizer',
              'layers_3_Dense__config__bias_regularizer',
              'layers_3_Dense__config__activity_regularizer',
              'layers_3_Dense__config__kernel_constraint',
              'layers_3_Dense__config__bias_constraint']
    
    assert got == expect, got


def test_set_params_keras_layers():
    config = model.get_config()
    layers = KerasLayers(name=config['name'], layers=config['layers'])
    params = {
        'layers_2_Activation': None,
        'layers_0_Dense__config__units': 96,
        'layers_3_Dense__config__kernel_initializer__config__scale': 2.0
    }
    layers.set_params(**params)
    new_layers = layers.layers

    got1 = len(new_layers)
    got2 = new_layers[0]['config']['units']
    got3 = new_layers[2]['config']['kernel_initializer']['config']['scale']

    assert got1 == 3, got1
    assert got2 == 96, got2
    assert got3 == 2.0, got3


def test_clone_keras_layers():
    config = model.get_config()
    layers = KerasLayers(name=config['name'], layers=config['layers'])
    layers_clone = clone(layers)

    new_params = {
        'layers_2_Activation': None,
    }

    layers_clone.set_params(**new_params)

    got1 = len(layers.layers)
    got2 = len(layers_clone.layers)

    assert got1 == 4, got1
    assert got2 == 3, got2


def test_get_params_base_keras_model():
    config = model.get_config()
    layers = KerasLayers(name=config['name'], layers=config['layers'])
    layers_clone = clone(layers)
    classifier = KerasGClassifier(layers_clone)

    params = classifier.get_params()
    got = {}
    for key, value in params.items():
        if not key.startswith('layers'):
            got[key] = value

    expect = {
        'amsgrad': None,
        'batch_size': None,
        'beta_1': None,
        'beta_2': None,
        'decay': 0,
        'epochs': 1,
        'epsilon': None,
        'loss': 'binary_crossentropy',
        'lr': 0.01,
        'metrics': [],
        'model_type': 'sequential',
        'momentum': 0,
        'nesterov': False,
        'optimizer': 'sgd',
        'rho': None,
        'schedule_decay': None
    }

    assert got == expect, got


def test_set_params_base_keras_model():
    config = model.get_config()
    layers = KerasLayers(name=config['name'], layers=config['layers'])
    layers_clone = clone(layers)
    classifier = KerasGClassifier(layers_clone)

    params = {
        'layers__layers_3_Dense__config__kernel_initializer__config__scale': 2.0,
        'layers__layers_2_Activation': None,
        'lr': 0.05,
    }

    classifier.set_params(**params)

    got1 = len(classifier.layers.layers)
    got2 = classifier.lr
    got3 = classifier.layers.layers[2]['config']['kernel_initializer']['config']['scale']

    assert got1 == 3, got1
    assert got2 == 0.05, got2
    assert got3 == 2.0, got3



def test_get_params_keras_g_classifier():
    config = train_model.get_config()
    layers = KerasLayers(name=config['name'], layers=config['layers'])
    layers_clone = clone(layers)
    classifier = KerasGClassifier(layers_clone, optimizer='adam', metrics=['accuracy'])

    got = list(classifier.get_params().keys())
    got = [x for x in got if not x.startswith('layers') or x.endswith('seed')]

    expect = ['amsgrad', 'batch_size', 'beta_1', 'beta_2', 'decay', 'epochs', 'epsilon',
              'layers__layers_0_Dense__config__kernel_initializer__config__seed',
              'layers__layers_1_Dense__config__kernel_initializer__config__seed',
              'loss', 'lr', 'metrics', 'model_type', 'momentum', 'nesterov', 'optimizer',
              'rho', 'schedule_decay']

    assert got == expect, got


def test_gridsearchcv_keras_g_classifier():
    config = train_model.get_config()
    layers = KerasLayers(name=config['name'], layers=config['layers'])
    layers_clone = clone(layers)
    classifier = KerasGClassifier(layers_clone, optimizer='adam', metrics=[])

    param_grid = dict(
        epochs = [60],
        batch_size = [20],
        lr = [0.03],
        layers__layers_1_Dense__config__kernel_initializer__config__seed = [42],
        layers__layers_0_Dense__config__kernel_initializer__config__seed = [999]
    )
    cv = StratifiedKFold(n_splits=3)

    grid = GridSearchCV(classifier, param_grid, cv=cv, scoring='accuracy', refit=True)
    grid_result = grid.fit(X, y)

    got1 = round(grid_result.best_score_, 2)
    got2 = grid_result.best_estimator_.lr
    got3 = grid_result.best_estimator_.epochs
    got4 = grid_result.best_estimator_.batch_size
    got5 = grid_result.best_estimator_.layers.layers[0]['config']['kernel_initializer']['config']['seed']
    got6 = grid_result.best_estimator_.layers.layers[1]['config']['kernel_initializer']['config']['seed']

    assert got1 == 0.65, got1
    assert got2 == 0.03, got2
    assert got3 == 60, got3
    assert got4 == 20, got4
    assert got5 == 999, got5
    assert got6 == 42, got6


def test_get_params_keras_g_regressor():
    config = train_model.get_config()
    layers = KerasLayers(name=config['name'], layers=config['layers'])
    layers_clone = clone(layers)
    regressor = KerasGRegressor(layers_clone, optimizer='sgd')

    got = list(regressor.get_params().keys())
    got = [x for x in got if not x.startswith('layers') or x.endswith('seed')]

    expect = ['amsgrad', 'batch_size', 'beta_1', 'beta_2', 'decay', 'epochs', 'epsilon',
              'layers__layers_0_Dense__config__kernel_initializer__config__seed',
              'layers__layers_1_Dense__config__kernel_initializer__config__seed',
              'loss', 'lr', 'metrics', 'model_type', 'momentum', 'nesterov', 'optimizer',
              'rho', 'schedule_decay']

    assert got == expect, got


def test_gridsearchcv_keras_g_regressor():
    config = train_model.get_config()
    layers = KerasLayers(name=config['name'], layers=config['layers'])
    layers_clone = clone(layers)
    regressor = KerasGRegressor(layers_clone, optimizer='adam', metrics=[])

    param_grid = dict(
        epochs = [60],
        batch_size = [20],
        lr = [0.03],
        layers__layers_1_Dense__config__kernel_initializer__config__seed = [42],
        layers__layers_0_Dense__config__kernel_initializer__config__seed = [999]
    )
    cv = KFold(n_splits=3)

    grid = GridSearchCV(regressor, param_grid, cv=cv, scoring='r2', refit=True)
    grid_result = grid.fit(X, y)

    got1 = round(grid_result.best_score_, 2)
    got2 = grid_result.best_estimator_.lr
    got3 = grid_result.best_estimator_.epochs
    got4 = grid_result.best_estimator_.batch_size
    got5 = grid_result.best_estimator_.layers.layers[0]['config']['kernel_initializer']['config']['seed']
    got6 = grid_result.best_estimator_.layers.layers[1]['config']['kernel_initializer']['config']['seed']

    assert got1 == -0.54, got1
    assert got2 == 0.03, got2
    assert got3 == 60, got3
    assert got4 == 20, got4
    assert got5 == 999, got5
    assert got6 == 42, got6


def test_check_params():
    fn = (Sequential.fit, Sequential.predict)
    params1 = dict(
        epochs = 100,
        validation_split = 0.2
    )
    params2 = dict(
        random_state = 9999
    )
    try:
        check_params(params1, fn)
        got1 = True
    except ValueError:
        got1 = False
        pass

    try:
        check_params(params2, Sequential.fit)
        got2 = True
    except ValueError:
        got2 = False
        pass

    assert got1 is True, got1
    assert got2 is False, got2
