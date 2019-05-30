import json
import keras
import numpy as np
import os
import pandas as pd
import tempfile
import warnings
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Activation
from sklearn.base import clone
from sklearn.metrics import SCORERS
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.model_selection import _search
from tensorflow import set_random_seed

from galaxy_ml.keras_galaxy_models import (
    _get_params_from_dict, _param_to_dict, _update_dict,
    check_params, SearchParam, KerasLayers,
    BaseKerasModel, KerasGClassifier, KerasGRegressor,
    KerasGBatchClassifier)
from galaxy_ml.preprocessors import ImageBatchGenerator
from galaxy_ml.model_validations import _fit_and_score


warnings.simplefilter('ignore')

np.random.seed(1)
set_random_seed(8888)

# test API model
model = Sequential()
model.add(Dense(64))
model.add(Activation('tanh'))
model.add(Activation('tanh'))
model.add(Dense(32))

# toy dataset
df = pd.read_csv('./test-data/pima-indians-diabetes.csv', sep=',')
X = df.iloc[:, 0:8].values.astype(float)
y = df.iloc[:, 8].values

# gridsearch model
train_model = Sequential()
train_model.add(Dense(12, input_dim=8, activation='relu'))
train_model.add(Dense(1, activation='sigmoid'))

# ResNet model
inputs = keras.Input(shape=(32, 32, 3), name='img')
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.Conv2D(64, 3, activation='relu')(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
block_2_output = layers.add([x, block_1_output])

x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_2_output)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation='relu')(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

resnet_model = keras.Model(inputs, outputs, name='toy_resnet')


d = {
    'name': 'sequential_1',
    'layers': [
        {
            'class_name': 'Dense',
            'config': {
                'name': 'dense_1',
                'trainable': True,
                'units': 64,
                'activation': 'linear',
                'use_bias': True,
                'kernel_initializer': {
                    'class_name': 'VarianceScaling',
                    'config': {
                        'scale': 1.0,
                        'mode': 'fan_avg',
                        'distribution': 'uniform',
                        'seed': None}},
                'bias_initializer': {
                    'class_name': 'Zeros',
                    'config': {}},
                'kernel_regularizer': None,
                'bias_regularizer': None,
                'activity_regularizer': None,
                'kernel_constraint': None,
                'bias_constraint': None}},
        {
            'class_name': 'Activation',
            'config': {
                'name': 'activation_1',
                'trainable': True,
                'activation': 'tanh'}},
        {
            'class_name': 'Activation',
            'config': {
                'name': 'activation_2',
                'trainable': True,
                'activation': 'tanh'}},
        {
            'class_name': 'Dense',
            'config': {
                'name': 'dense_2',
                'trainable': True,
                'units': 32,
                'activation': 'linear',
                'use_bias': True,
                'kernel_initializer': {
                    'class_name': 'VarianceScaling',
                    'config': {
                        'scale': 1.0,
                        'mode': 'fan_avg',
                        'distribution': 'uniform',
                        'seed': None}},
                'bias_initializer': {
                    'class_name': 'Zeros',
                    'config': {}},
                'kernel_regularizer': None,
                'bias_regularizer': None,
                'activity_regularizer': None,
                'kernel_constraint': None,
                'bias_constraint': None}}]}


def test_get_params_from_dict():
    got = list(_get_params_from_dict(d['layers'][0], 'layers_0_Dense').keys())
    expect = [
        'layers_0_Dense__class_name', 'layers_0_Dense__config',
        'layers_0_Dense__config__name',
        'layers_0_Dense__config__trainable',
        'layers_0_Dense__config__units',
        'layers_0_Dense__config__activation',
        'layers_0_Dense__config__use_bias',
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
        'layers_0_Dense__config__bias_constraint']
    assert got == expect, got


def test_param_to_dict():
    param = {
        'layers_0_Dense__config__kernel_initializer__config__distribution':
            'uniform'}
    key = list(param.keys())[0]
    got = _param_to_dict(key, param[key])

    expect = {'layers_0_Dense': {
                'config': {
                    'kernel_initializer': {
                        'config': {
                            'distribution': 'uniform'}}}}}
    assert got == expect, got


def test_update_dict():
    config = model.get_config()
    layers = config['layers']
    d = layers[0]
    u = {'config': {
            'kernel_initializer': {
                'config': {
                    'distribution': 'random_uniform'}}}}
    got = _update_dict(d, u)

    expect = {
        'class_name': 'Dense',
        'config': {
            'name': 'dense_1',
            'trainable': True,
            'units': 64,
            'activation': 'linear',
            'use_bias': True,
            'kernel_initializer': {
                'class_name': 'VarianceScaling',
                'config': {
                    'scale': 1.0,
                    'mode': 'fan_avg',
                    'distribution': 'random_uniform',
                    'seed': None}},
            'bias_initializer': {
                'class_name': 'Zeros',
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
    expect = [
        'layers', 'name', 'layers_0_Dense', 'layers_1_Activation',
        'layers_2_Activation', 'layers_3_Dense',
        'layers_0_Dense__class_name', 'layers_0_Dense__config',
        'layers_0_Dense__config__name',
        'layers_0_Dense__config__trainable',
        'layers_0_Dense__config__units',
        'layers_0_Dense__config__activation',
        'layers_0_Dense__config__use_bias',
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
        'layers_1_Activation__config',
        'layers_1_Activation__config__name',
        'layers_1_Activation__config__trainable',
        'layers_1_Activation__config__activation',
        'layers_2_Activation__class_name', 'layers_2_Activation__config',
        'layers_2_Activation__config__name',
        'layers_2_Activation__config__trainable',
        'layers_2_Activation__config__activation',
        'layers_3_Dense__class_name',
        'layers_3_Dense__config', 'layers_3_Dense__config__name',
        'layers_3_Dense__config__trainable',
        'layers_3_Dense__config__units',
        'layers_3_Dense__config__activation',
        'layers_3_Dense__config__use_bias',
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
    classifier = KerasGClassifier(config)

    params = classifier.get_params()
    got = {}
    for key, value in params.items():
        if not key.startswith('layers') and not key.startswith('config'):
            got[key] = value

    expect = {
        'amsgrad': None, 'batch_size': None,
        'beta_1': None, 'beta_2': None,
        'callbacks': None, 'decay': 0,
        'epochs': 1, 'epsilon': None,
        'loss': 'binary_crossentropy', 'lr': 0.01,
        'metrics': [], 'model_type': 'sequential',
        'momentum': 0, 'nesterov': False,
        'optimizer': 'sgd', 'rho': None,
        'schedule_decay': None, 'seed': 0,
        'validation_data': None}

    assert got == expect, got


def test_set_params_base_keras_model():
    config = model.get_config()
    classifier = KerasGClassifier(config)

    params = {
        'layers_3_Dense__config__kernel_initializer__config__scale': 2.0,
        'layers_2_Activation': None,
        'lr': 0.05,
    }

    classifier.set_params(**params)

    got1 = len(classifier.config['layers'])
    got2 = classifier.lr
    got3 = (classifier.config['layers'][2]['config']
            ['kernel_initializer']['config']['scale'])

    assert got1 == 3, got1
    assert got2 == 0.05, got2
    assert got3 == 2.0, got3


def test_get_params_keras_g_classifier():
    config = train_model.get_config()
    classifier = KerasGClassifier(config, optimizer='adam',
                                  metrics=['accuracy'])

    got = list(classifier.get_params().keys())
    got = [x for x in got if not x.startswith('layers') or x.endswith('seed')]

    expect = [
        'amsgrad', 'batch_size', 'beta_1', 'beta_2', 'callbacks',
        'config', 'decay', 'epochs', 'epsilon', 'loss', 'lr',
        'metrics', 'model_type', 'momentum', 'nesterov', 'optimizer',
        'rho', 'schedule_decay', 'seed', 'validation_data',
        'layers_0_Dense__config__kernel_initializer__config__seed',
        'layers_1_Dense__config__kernel_initializer__config__seed']

    assert got == expect, got


def test_gridsearchcv_keras_g_classifier():
    setattr(_search, '_fit_and_score', _fit_and_score)
    GridSearchCV = getattr(_search, 'GridSearchCV')

    config = train_model.get_config()
    classifier = KerasGClassifier(config, optimizer='adam',
                                  batch_size=32, metrics=[])

    param_grid = dict(
        epochs=[60],
        batch_size=[20],
        lr=[0.003],
        layers_1_Dense__config__kernel_initializer__config__seed=[999],
        layers_0_Dense__config__kernel_initializer__config__seed=[999]
    )
    cv = StratifiedKFold(n_splits=5)

    grid = GridSearchCV(classifier, param_grid, cv=cv,
                        scoring='accuracy', refit=True)
    grid_result = grid.fit(X, y)

    got1 = round(grid_result.best_score_, 2)
    got2 = grid_result.best_estimator_.lr
    got3 = grid_result.best_estimator_.epochs
    got4 = grid_result.best_estimator_.batch_size
    got5 = (grid_result.best_estimator_.config['layers'][0]['config']
            ['kernel_initializer']['config']['seed'])
    got6 = (grid_result.best_estimator_.config['layers'][1]['config']
            ['kernel_initializer']['config']['seed'])

    print(grid_result.best_score_)
    assert got1 == 0.68, got1
    assert got2 == 0.003, got2
    assert got3 == 60, got3
    assert got4 == 20, got4
    assert got5 == 999, got5
    assert got6 == 999, got6


def test_get_params_keras_g_regressor():
    config = train_model.get_config()
    regressor = KerasGRegressor(config, optimizer='sgd')

    got = list(regressor.get_params().keys())
    got = [x for x in got if not x.startswith('layers') or x.endswith('seed')]

    expect = [
        'amsgrad', 'batch_size', 'beta_1', 'beta_2', 'callbacks',
        'config', 'decay', 'epochs', 'epsilon', 'loss', 'lr',
        'metrics', 'model_type', 'momentum', 'nesterov', 'optimizer',
        'rho', 'schedule_decay', 'seed', 'validation_data',
        'layers_0_Dense__config__kernel_initializer__config__seed',
        'layers_1_Dense__config__kernel_initializer__config__seed']

    assert got == expect, got


def test_gridsearchcv_keras_g_regressor():
    config = train_model.get_config()
    regressor = KerasGRegressor(config, optimizer='adam', metrics=[])

    param_grid = dict(
        epochs=[60],
        batch_size=[20],
        lr=[0.002],
        layers_1_Dense__config__kernel_initializer__config__seed=[999],
        layers_0_Dense__config__kernel_initializer__config__seed=[999]
    )
    cv = KFold(n_splits=3)

    grid = GridSearchCV(regressor, param_grid, cv=cv, scoring='r2', refit=True)
    grid_result = grid.fit(X, y)

    got1 = round(grid_result.best_score_, 2)
    got2 = grid_result.best_estimator_.lr
    got3 = grid_result.best_estimator_.epochs
    got4 = grid_result.best_estimator_.batch_size
    got5 = (grid_result.best_estimator_.config['layers']
            [0]['config']['kernel_initializer']['config']['seed'])
    got6 = (grid_result.best_estimator_.config['layers'][1]['config']
            ['kernel_initializer']['config']['seed'])

    assert -0.04 <= got1 <= 0.04, got1
    assert got2 == 0.002, got2
    assert got3 == 60, got3
    assert got4 == 20, got4
    assert got5 == 999, got5
    assert got6 == 999, got6


def test_check_params():
    fn = (Sequential.fit, Sequential.predict)
    params1 = dict(
        epochs=100,
        validation_split=0.2
    )
    params2 = dict(
        random_state=9999
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


def test_funtional_model_get_params():
    config = resnet_model.get_config()
    classifier = KerasGClassifier(config, model_type='functional')

    params = classifier.get_params()
    got = {}
    for key, value in params.items():
        if key.startswith('layers_1_Conv2D__')\
            or (not key.endswith('config')
                and not key.startswith('layers')):
            got[key] = value

    expect = {
        'amsgrad': None,
        'batch_size': None,
        'beta_1': None,
        'beta_2': None,
        'callbacks': None,
        'decay': 0,
        'epochs': 1,
        'epsilon': None,
        'loss': 'binary_crossentropy',
        'lr': 0.01,
        'metrics': [],
        'model_type': 'functional',
        'momentum': 0,
        'nesterov': False,
        'optimizer': 'sgd',
        'rho': None,
        'schedule_decay': None,
        'seed': 0,
        'validation_data': None,
        'layers_1_Conv2D__name': 'conv2d_1',
        'layers_1_Conv2D__class_name': 'Conv2D',
        'layers_1_Conv2D__config': {
            'name': 'conv2d_1',
            'trainable': True,
            'filters': 32,
            'kernel_size': (3, 3),
            'strides': (1, 1),
            'padding': 'valid',
            'data_format': 'channels_last',
            'dilation_rate': (1, 1),
            'activation': 'relu',
            'use_bias': True,
            'kernel_initializer': {
                'class_name': 'VarianceScaling',
                'config': {'scale': 1.0,
                           'mode': 'fan_avg',
                           'distribution': 'uniform',
                           'seed': None}},
            'bias_initializer': {
                'class_name': 'Zeros',
                'config': {}},
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None,
            'kernel_constraint': None,
            'bias_constraint': None},
        'layers_1_Conv2D__config__name': 'conv2d_1',
        'layers_1_Conv2D__config__trainable': True,
        'layers_1_Conv2D__config__filters': 32,
        'layers_1_Conv2D__config__kernel_size': (3, 3),
        'layers_1_Conv2D__config__strides': (1, 1),
        'layers_1_Conv2D__config__padding': 'valid',
        'layers_1_Conv2D__config__data_format': 'channels_last',
        'layers_1_Conv2D__config__dilation_rate': (1, 1),
        'layers_1_Conv2D__config__activation': 'relu',
        'layers_1_Conv2D__config__use_bias': True,
        'layers_1_Conv2D__config__kernel_initializer': {
            'class_name': 'VarianceScaling',
            'config': {'scale': 1.0,
                       'mode': 'fan_avg',
                       'distribution': 'uniform',
                       'seed': None}},
        'layers_1_Conv2D__config__kernel_initializer__class_name':
            'VarianceScaling',
        'layers_1_Conv2D__config__kernel_initializer__config': {
            'scale': 1.0,
            'mode': 'fan_avg',
            'distribution': 'uniform',
            'seed': None},
        'layers_1_Conv2D__config__kernel_initializer__config__scale': 1.0,
        'layers_1_Conv2D__config__kernel_initializer__config__mode': 'fan_avg',
        'layers_1_Conv2D__config__kernel_initializer__config__distribution':
            'uniform',
        'layers_1_Conv2D__config__kernel_initializer__config__seed': None,
        'layers_1_Conv2D__config__bias_initializer': {
            'class_name': 'Zeros',
            'config': {}},
        'layers_1_Conv2D__config__bias_initializer__class_name': 'Zeros',
        'layers_1_Conv2D__config__bias_initializer__config': {},
        'layers_1_Conv2D__config__kernel_regularizer': None,
        'layers_1_Conv2D__config__bias_regularizer': None,
        'layers_1_Conv2D__config__activity_regularizer': None,
        'layers_1_Conv2D__config__kernel_constraint': None,
        'layers_1_Conv2D__config__bias_constraint': None,
        'layers_1_Conv2D__inbound_nodes': [[['img', 0, 0, {}]]]
    }

    assert got == expect, got


def test_set_params_functional_model():
    config = resnet_model.get_config()
    classifier = KerasGClassifier(config, model_type='functional')

    params = {
        'layers_1_Conv2D__config__kernel_initializer__config__seed': 9999,
        'layers_1_Conv2D__config__filters': 64,
        'lr': 0.03,
        'epochs': 200
    }

    classifier.set_params(**params)

    got1 = (classifier.config['layers'][1]['config']
            ['kernel_initializer']['config']['seed'])
    got2 = classifier.config['layers'][1]['config']['filters']
    got3 = classifier.lr
    got4 = classifier.epochs

    assert got1 == 9999, got1
    assert got2 == 64, got2
    assert got3 == 0.03, got3
    assert got4 == 200, got4


def test_to_json_keras_g_classifier():
    config = model.get_config()
    classifier = KerasGClassifier(config, model_type='sequential')

    got = classifier.to_json()

    with open('./test-data/to_json.txt', 'r') as f:
        expect = f.read()
    assert got == expect, got


def test_keras_model_to_json():
    with open('./test-data/keras02.json', 'r') as f:
        model_json = json.load(f)

    if model_json['class_name'] == 'Sequential':
        model_type = 'sequential'
    elif model_json['class_name'] == 'Model':
        model_type = 'functional'

    config = model_json.get('config')

    model = KerasGClassifier(config, model_type=model_type)

    got = model.to_json()  # json_string

    assert len(got) > 4850, len(got)
    assert got.startswith('{"class_name": "Model",'), got


def test_keras_model_load_and_save_weights():
    with open('./test-data/keras_model_drosophila01.json', 'r') as f:
        model_json = json.load(f)

    config = model_json.get('config')
    if model_json['class_name'] == 'Sequential':
        model_type = 'sequential'
    else:
        model_type = 'functional'
    model = KerasGRegressor(config, model_type=model_type)

    model.load_weights('./test-data/keras_model_drosophila_weights01.h5')

    tmp = tempfile.mktemp()
    model.save_weights(tmp)

    got = os.path.getsize(tmp)
    expect = os.path.getsize('./test-data/keras_model_drosophila_weights01.h5')

    assert abs(got - expect) < 40, got - expect


def test_image_batch_generator_get_params():
    batch_generator = ImageBatchGenerator()

    got = batch_generator.get_params()
    expect = {'brightness_range': None, 'channel_shift_range': 0.0,
              'cval': 0.0, 'data_format': 'channels_last',
              'dtype': 'float32', 'featurewise_center': False,
              'featurewise_std_normalization': False,
              'fill_mode': 'nearest', 'height_shift_range': 0.0,
              'horizontal_flip': False, 'preprocessing_function': None,
              'rescale': None, 'rotation_range': 0,
              'samplewise_center': False,
              'samplewise_std_normalization': False, 'shear_range': 0.0,
              'validation_split': None, 'vertical_flip': False,
              'width_shift_range': 0.0, 'zca_epsilon': 1e-06,
              'zca_whitening': False, 'zoom_range': [1.0, 1.0]}

    assert got == expect, got


def test_keras_batch_classifier_get_params():
    config = model.get_config()
    batch_generator = ImageBatchGenerator()
    array_convertor = None
    classifier = KerasGBatchClassifier(
        config, batch_generator, array_convertor, model_type='sequential')

    params = classifier.get_params()
    got = {key: value for key, value in params.items()
           if not key.startswith(('config', 'layers'))}

    got.pop('train_batch_generator', None)
    expect = {'amsgrad': None, 'batch_size': None, 'beta_1': None,
              'beta_2': None, 'callbacks': None, 'decay': 0,
              'epochs': 1, 'epsilon': None, 'loss': 'binary_crossentropy',
              'lr': 0.01, 'metrics': [], 'model_type': 'sequential',
              'momentum': 0, 'n_jobs': 1, 'nesterov': False,
              'optimizer': 'sgd', 'predict_batch_generator': None,
              'rho': None, 'schedule_decay': None, 'seed': 0,
              'to_array_converter': None,
              'train_batch_generator__brightness_range': None,
              'train_batch_generator__channel_shift_range': 0.0,
              'train_batch_generator__cval': 0.0,
              'train_batch_generator__data_format': 'channels_last',
              'train_batch_generator__dtype': 'float32',
              'train_batch_generator__featurewise_center': False,
              'train_batch_generator__featurewise_std_normalization': False,
              'train_batch_generator__fill_mode': 'nearest',
              'train_batch_generator__height_shift_range': 0.0,
              'train_batch_generator__horizontal_flip': False,
              'train_batch_generator__preprocessing_function': None,
              'train_batch_generator__rescale': None,
              'train_batch_generator__rotation_range': 0,
              'train_batch_generator__samplewise_center': False,
              'train_batch_generator__samplewise_std_normalization': False,
              'train_batch_generator__shear_range': 0.0,
              'train_batch_generator__validation_split': None,
              'train_batch_generator__vertical_flip': False,
              'train_batch_generator__width_shift_range': 0.0,
              'train_batch_generator__zca_epsilon': 1e-06,
              'train_batch_generator__zca_whitening': False,
              'train_batch_generator__zoom_range': [1.0, 1.0],
              'validation_data': None}

    for k, v in got.items():
        if k not in expect:
            print("%s: %s" % (k, v))

    assert got == expect, got


def test_keras_galaxy_model_callbacks():
    config = train_model.get_config()

    cbacks = [
        {'callback_selection':
            {'monitor': 'val_loss', 'min_delta': 0.001,
             'callback_type': 'ReduceLROnPlateau',
             'min_lr': 0.0, 'patience': 10, 'cooldown': 0,
             'mode': 'auto', 'factor': 0.2}},
        {'callback_selection':
            {'callback_type': 'TerminateOnNaN'}},
        {'callback_selection':
            {'callback_type': 'CSVLogger',
             'filename': './tests/log.cvs',
             'separator': '\t', 'append': True}},
        {'callback_selection':
            {'baseline': None, 'min_delta': 0.,
             'callback_type': 'EarlyStopping', 'patience': 10,
             'mode': 'auto', 'restore_best_weights': True,
             'monitor': 'val_loss'}},
        {'callback_selection':
            {'monitor': 'val_loss', 'save_best_only': True,
             'period': 1, 'save_weights_only': True,
             'filepath': './tests/weights.hdf5',
             'callback_type': 'ModelCheckpoint', 'mode': 'auto'}}]

    estimator = KerasGClassifier(config, optimizer='adam',
                                 metrics=[], batch_size=32,
                                 epochs=500,
                                 callbacks=cbacks)

    scorer = SCORERS['accuracy']
    train, test = next(KFold(n_splits=5).split(X, y))

    new_params = {
        'layers_0_Dense__config__kernel_initializer__config__seed': 42,
        'layers_1_Dense__config__kernel_initializer__config__seed': 42
    }
    parameters = new_params
    fit_params = {'shuffle': False}

    got1 = _fit_and_score(estimator, X, y, scorer, train, test,
                          verbose=0, parameters=parameters,
                          fit_params=fit_params)

    assert 0.69 <= round(got1[0], 2) <= 0.74, got1


def test_keras_galaxy_model_callbacks_girdisearch():

    setattr(_search, '_fit_and_score', _fit_and_score)
    GridSearchCV = getattr(_search, 'GridSearchCV')

    config = train_model.get_config()
    cbacks = [
        {'callback_selection':
            {'monitor': 'val_loss', 'min_delta': 0.001,
             'callback_type': 'ReduceLROnPlateau',
             'min_lr': 0.0, 'patience': 10, 'cooldown': 0,
             'mode': 'auto', 'factor': 0.2}},
        {'callback_selection':
            {'callback_type': 'TerminateOnNaN'}},
        {'callback_selection':
            {'callback_type': 'CSVLogger',
             'filename': './tests/log.cvs',
             'separator': '\t', 'append': True}},
        {'callback_selection':
            {'baseline': None, 'min_delta': 0.,
             'callback_type': 'EarlyStopping', 'patience': 10,
             'mode': 'auto', 'restore_best_weights': True,
             'monitor': 'val_loss'}}]
    estimator = KerasGClassifier(config, optimizer='adam',
                                 metrics=[], batch_size=32,
                                 epochs=500,
                                 callbacks=cbacks)

    scorer = SCORERS['balanced_accuracy']
    cv = KFold(n_splits=5)

    new_params = {
        'layers_0_Dense__config__kernel_initializer__config__seed': [42],
        'layers_1_Dense__config__kernel_initializer__config__seed': [42]
    }
    fit_params = {'shuffle': False}

    grid = GridSearchCV(estimator, param_grid=new_params, scoring=scorer,
                        cv=cv, n_jobs=2, refit=False, error_score=np.NaN)

    grid.fit(X, y, **fit_params)

    got1 = grid.best_score_

    assert 0.64 <= round(got1, 2) <= 0.70, got1
