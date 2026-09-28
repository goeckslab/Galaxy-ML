import glob
import copy
import json
import os
import tempfile
import warnings

from galaxy_ml.keras_galaxy_models import (
    KerasGBatchClassifier, KerasGClassifier, KerasGRegressor,
    KerasLayers, MetricCallback, _get_params_from_dict, _param_to_dict,
    _predict_generator, _update_dict, check_params, load_model,
)
from galaxy_ml.model_validations import _fit_and_score
from galaxy_ml.preprocessors import FastaDNABatchGenerator
from galaxy_ml.preprocessors import FastaProteinBatchGenerator

import h5py

from keras import layers
from keras.layers import (
    Activation, Conv1D, Conv2D, Dense, Flatten, MaxPooling2D, Reshape,
)
from keras.models import Model, Sequential
from keras.utils import to_categorical

import numpy as np

import pandas as pd

from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    GridSearchCV, KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit,
    _search,
)

import tensorflow as tf
import keras


warnings.simplefilter('ignore')

np.random.seed(1)
tf.random.set_seed(8888)

# test API model
model = Sequential()
model.add(Dense(64))
model.add(Activation('tanh'))
model.add(Activation('tanh'))
model.add(Dense(32))

# toy dataset
df = pd.read_csv('./tools/test-data/pima-indians-diabetes.csv', sep=',')
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
    'name': 'sequential',
    'layers': [
        {
            'class_name': 'Dense',
            'config': {
                'name': 'dense',
                'trainable': True,
                'dtype': 'float32',
                'units': 64,
                'activation': 'linear',
                'use_bias': True,
                'kernel_initializer': {
                    'class_name': 'GlorotUniform',
                    'config': {
                        'seed': None,
                    },
                },
                'bias_initializer': {
                    'class_name': 'Zeros',
                    'config': {}
                },
                'kernel_regularizer': None,
                'bias_regularizer': None,
                'activity_regularizer': None,
                'kernel_constraint': None,
                'bias_constraint': None
            }
        },
        {
            'class_name': 'Activation',
            'config': {
                'name': 'activation',
                'trainable': True,
                'dtype': 'float32',
                'activation': 'tanh'
            }
        },
        {
            'class_name': 'Activation',
            'config': {
                'name': 'activation_1',
                'trainable': True,
                'dtype': 'float32',
                'activation': 'tanh'
            }
        },
        {
            'class_name': 'Dense',
            'config': {
                'name': 'dense_1',
                'trainable': True,
                'dtype': 'float32',
                'units': 32,
                'activation': 'linear',
                'use_bias': True,
                'kernel_initializer': {
                    'class_name': 'GlorotUniform',
                    'config': {
                        'seed': None,
                    },
                },
                'bias_initializer': {
                    'class_name': 'Zeros',
                    'config': {}
                },
                'kernel_regularizer': None,
                'bias_regularizer': None,
                'activity_regularizer': None,
                'kernel_constraint': None,
                'bias_constraint': None
            }
        }
    ]
}


def teardown_module():
    files = glob.glob('./tests/*.weights.h5', recursive=False)
    for fl in files:
        os.remove(fl)
    log_file = glob.glob('./tests/log.cvs', recursive=False)
    for fl in log_file:
        os.remove(fl)


def test_get_params_from_dict():
    got = list(_get_params_from_dict(d['layers'][0], 'layers_0_Dense').keys())
    expect = [
        'layers_0_Dense__class_name',
        'layers_0_Dense__config',
        'layers_0_Dense__config__name',
        'layers_0_Dense__config__trainable',
        'layers_0_Dense__config__dtype',
        'layers_0_Dense__config__units',
        'layers_0_Dense__config__activation',
        'layers_0_Dense__config__use_bias',
        'layers_0_Dense__config__kernel_initializer',
        'layers_0_Dense__config__kernel_initializer__class_name',
        'layers_0_Dense__config__kernel_initializer__config',
        'layers_0_Dense__config__kernel_initializer__config__seed',
        'layers_0_Dense__config__bias_initializer',
        'layers_0_Dense__config__bias_initializer__class_name',
        'layers_0_Dense__config__bias_initializer__config',
        'layers_0_Dense__config__kernel_regularizer',
        'layers_0_Dense__config__bias_regularizer',
        'layers_0_Dense__config__activity_regularizer',
        'layers_0_Dense__config__kernel_constraint',
        'layers_0_Dense__config__bias_constraint'
    ]
    assert got == expect, got


def test_param_to_dict():
    param = {
        'layers_0_Dense__config__kernel_initializer__config__seed': None
    }
    key = list(param.keys())[0]
    got = _param_to_dict(key, param[key])

    expect = {
        'layers_0_Dense': {
            'config': {
                'kernel_initializer': {
                    'config': {
                        'seed': None,
                    },
                },
            },
        },
    }
    assert got == expect, got


def test_update_dict():
    config = model.get_config()
    layers = config['layers']
    d = layers[0]
    u = {
        'config': {
            'kernel_initializer': {
                'config': {
                    'seed': 42,
                },
            },
        },
    }
    expected = copy.deepcopy(d)
    expected['config']['kernel_initializer']['config']['seed'] = 42
    got = _update_dict(d, u)

    assert got == expected


def test_get_params_keras_layers():
    config = model.get_config()
    layers = KerasLayers(name=config['name'], layers=config['layers'])
    params = layers.get_params()
    assert params['layers_0_Dense__config__units'] == 64
    assert params['layers_3_Dense__config__units'] == 32
    assert params['layers_1_Activation__config__activation'] == 'tanh'
    assert params['layers_0_Dense__config__dtype__config__name'] == 'float32'
    assert clone(layers).get_params()['layers'] == config['layers']


def test_set_params_keras_layers():
    config = model.get_config()
    layers = KerasLayers(name=config['name'], layers=config['layers'])
    params = {
        'layers_2_Activation': None,
        'layers_0_Dense__config__units': 96,
        'layers_3_Dense__config__kernel_initializer__config__seed': 42
    }
    layers.set_params(**params)
    new_layers = layers.layers

    got1 = len(new_layers)
    got2 = new_layers[0]['config']['units']
    got3 = new_layers[2]['config']['kernel_initializer']['config']['seed']

    assert got1 == 3, got1
    assert got2 == 96, got2
    assert got3 == 42, got3


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
        'amsgrad': None, 'batch_size': 32, 'beta': None,
        'beta_1': None, 'beta_2': None, 'callbacks': None,
        'centered': None, 'epochs': 1, 'epsilon': None,
        'initial_accumulator_value': None,
        'l1_regularization_strength': None,
        'l2_regularization_strength': None,
        'l2_shrinkage_regularization_strength': None,
        'learning_rate': None, 'learning_rate_power': None,
        'loss': None, 'loss_weights': None, 'metrics': [],
        'model_type': 'sequential', 'momentum': None,
        'nesterov': None, 'optimizer': 'rmsprop', 'rho': None,
        'run_eagerly': None, 'seed': None,
        'steps_per_epoch': None, 'steps_per_execution': None,
        'validation_split': 0.1, 'validation_steps': None,
        'verbose': 1
    }

    assert got == expect, got


def test_set_params_base_keras_model():
    config = model.get_config()
    classifier = KerasGClassifier(config)

    params = {
        'layers_3_Dense__config__kernel_initializer__config__seed': 42,
        'layers_2_Activation': None,
        'learning_rate': 0.05,
    }

    classifier.set_params(**params)

    got1 = len(classifier.config['layers'])
    got2 = classifier.learning_rate
    got3 = (classifier.config['layers'][2]['config']
            ['kernel_initializer']['config']['seed'])

    assert got1 == 3, got1
    assert got2 == 0.05, got2
    assert got3 == 42, got3


def test_get_params_keras_g_classifier():
    config = train_model.get_config()
    classifier = KerasGClassifier(config, optimizer='adam',
                                  metrics=['accuracy'])

    got = list(classifier.get_params().keys())
    got = [x for x in got if not x.startswith('layers') or x.endswith('seed')]

    expect = [
        'amsgrad', 'batch_size', 'beta', 'beta_1', 'beta_2',
        'callbacks', 'centered', 'config', 'epochs', 'epsilon',
        'initial_accumulator_value', 'l1_regularization_strength',
        'l2_regularization_strength',
        'l2_shrinkage_regularization_strength', 'learning_rate',
        'learning_rate_power', 'loss', 'loss_weights', 'metrics',
        'model_type', 'momentum', 'nesterov', 'optimizer', 'rho',
        'run_eagerly', 'seed', 'steps_per_epoch',
        'steps_per_execution', 'validation_split',
        'validation_steps', 'verbose',
        'layers_1_Dense__config__kernel_initializer__config__seed',
        'layers_2_Dense__config__kernel_initializer__config__seed']

    assert got == expect, got


def test_gridsearchcv_keras_g_classifier():

    config = train_model.get_config()
    classifier = KerasGClassifier(config, optimizer='adam',
                                  loss='binary_crossentropy',
                                  batch_size=32, metrics=[],
                                  seed=42, verbose=0)

    param_grid = dict(
        epochs=[60],
        batch_size=[20],
        learning_rate=[0.003],
        layers_1_Dense__config__kernel_initializer__config__seed=[999],
        layers_2_Dense__config__kernel_initializer__config__seed=[999]
    )
    cv = StratifiedKFold(n_splits=5)

    grid = GridSearchCV(classifier, param_grid, cv=cv,
                        scoring='accuracy', refit=True,
                        error_score='raise')
    grid_result = grid.fit(X, y)

    got1 = round(grid_result.best_score_, 2)
    got2 = grid_result.best_estimator_.learning_rate
    got3 = grid_result.best_estimator_.epochs
    got4 = grid_result.best_estimator_.batch_size
    got5 = (grid_result.best_estimator_.config['layers'][1]['config']
            ['kernel_initializer']['config']['seed'])
    got6 = (grid_result.best_estimator_.config['layers'][2]['config']
            ['kernel_initializer']['config']['seed'])

    print(grid_result.best_score_)
    assert 0.68 <= got1 <= 0.74, got1
    assert got2 == 0.003, got2
    assert got3 == 60, got3
    assert got4 == 20, got4
    assert got5 == 999, got5
    assert got6 == 999, got6


def test_get_params_keras_g_regressor():
    config = train_model.get_config()
    regressor = KerasGRegressor(config, optimizer='sgd', loss='MSE')

    got = list(regressor.get_params().keys())
    got = [x for x in got if not x.startswith('layers') or x.endswith('seed')]

    expect = [
        'amsgrad', 'batch_size', 'beta', 'beta_1', 'beta_2',
        'callbacks', 'centered', 'config', 'epochs', 'epsilon',
        'initial_accumulator_value', 'l1_regularization_strength',
        'l2_regularization_strength',
        'l2_shrinkage_regularization_strength', 'learning_rate',
        'learning_rate_power', 'loss', 'loss_weights', 'metrics',
        'model_type', 'momentum', 'nesterov', 'optimizer', 'rho',
        'run_eagerly', 'seed', 'steps_per_epoch',
        'steps_per_execution', 'validation_split',
        'validation_steps', 'verbose',
        'layers_1_Dense__config__kernel_initializer__config__seed',
        'layers_2_Dense__config__kernel_initializer__config__seed']

    assert got == expect, got


def test_gridsearchcv_keras_g_regressor():
    train_model = Sequential()
    train_model.add(Dense(12, input_dim=8, activation='relu'))
    train_model.add(Dense(1))
    config = train_model.get_config()
    regressor = KerasGRegressor(config, optimizer='adam', metrics=[],
                                loss='mean_squared_error', seed=42,
                                verbose=0)

    param_grid = dict(
        epochs=[60],
        batch_size=[20],
        learning_rate=[0.002],
        layers_1_Dense__config__kernel_initializer__config__seed=[999],
        layers_2_Dense__config__kernel_initializer__config__seed=[999]
    )
    cv = KFold(n_splits=3)

    grid = GridSearchCV(regressor, param_grid, cv=cv, scoring='r2', refit=True)
    grid_result = grid.fit(X, y)

    print(grid_result.best_score_)
    got1 = round(grid_result.best_score_, 1)
    got2 = grid_result.best_estimator_.learning_rate
    got3 = grid_result.best_estimator_.epochs
    got4 = grid_result.best_estimator_.batch_size
    got5 = (grid_result.best_estimator_.config['layers']
            [1]['config']['kernel_initializer']['config']['seed'])
    got6 = (grid_result.best_estimator_.config['layers'][1]['config']
            ['kernel_initializer']['config']['seed'])

    assert -3. < got1 < -1., got1
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
    classifier = KerasGClassifier(config, model_type='functional',
                                  seed=0)

    params = classifier.get_params()
    assert params['layers_1_Conv2D__config__filters'] == 32
    assert params['layers_1_Conv2D__config__kernel_size'] == (3, 3)
    assert params['layers_1_Conv2D__config'] == config['layers'][1]['config']
    assert params['seed'] == 0
    assert clone(classifier).get_params()['config'] == config


def test_set_params_functional_model():
    config = resnet_model.get_config()
    classifier = KerasGClassifier(config, model_type='functional')

    params = {
        'layers_1_Conv2D__config__kernel_initializer__config__seed': 9999,
        'layers_1_Conv2D__config__filters': 64,
        'learning_rate': 0.03,
        'epochs': 200
    }

    classifier.set_params(**params)

    got1 = (classifier.config['layers'][1]['config']
            ['kernel_initializer']['config']['seed'])
    got2 = classifier.config['layers'][1]['config']['filters']
    got3 = classifier.learning_rate
    got4 = classifier.epochs

    assert got1 == 9999, got1
    assert got2 == 64, got2
    assert got3 == 0.03, got3
    assert got4 == 200, got4


def test_to_json_keras_g_classifier():
    config = model.get_config()
    classifier = KerasGClassifier(config, model_type='sequential')

    got = classifier.to_json()
    got = json.loads(got)
    assert got['class_name'] == 'Sequential'
    restored = keras.models.model_from_json(classifier.to_json())
    assert restored.get_config() == model.get_config()


def test_keras_model_to_json():
    with open('./tools/test-data/keras02.json', 'r') as f:
        model_json = json.load(f)

    if model_json['class_name'] == 'Sequential':
        model_type = 'sequential'
    elif model_json['class_name'] == 'Functional':
        model_type = 'functional'

    config = model_json.get('config')

    model = KerasGClassifier(config, model_type=model_type)

    got = model.to_json()  # json_string

    assert json.loads(got)['class_name'] == 'Functional'
    restored = keras.models.model_from_json(got)
    assert all(output.shape[-1] == 1 for output in restored.outputs)


def test_keras_model_load_and_save_weights():
    with open('./tools/test-data/keras_model_drosophila01.json', 'r') as f:
        model_json = json.load(f)

    config = model_json.get('config')
    if model_json['class_name'] == 'Sequential':
        model_type = 'sequential'
    else:
        model_type = 'functional'
    model = KerasGRegressor(config, model_type=model_type)

    model.load_weights('./tools/test-data/keras_model_drosophila_weights01.h5')

    _, tmp = tempfile.mkstemp()

    try:
        model.save_weights(tmp)

        restored = KerasGRegressor(config, model_type=model_type)
        restored.load_weights(tmp)
        for actual, expected in zip(restored.model_.get_weights(),
                                    model.model_.get_weights()):
            np.testing.assert_array_equal(actual, expected)
    finally:
        os.remove(tmp)


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
             'filepath': ('./tests/weights.{epoch:02d}-'
                          '{val_loss:.2f}.weights.h5'),
             'callback_type': 'ModelCheckpoint', 'mode': 'auto'}}]

    estimator = KerasGClassifier(config, optimizer='adam',
                                 loss='binary_crossentropy',
                                 metrics=[], batch_size=32,
                                 epochs=500, seed=42,
                                 callbacks=cbacks,
                                 verbose=0)

    scorer = get_scorer('accuracy')
    train, test = next(KFold(n_splits=5).split(X, y))

    new_params = {
        'layers_1_Dense__config__kernel_initializer__config__seed': 42,
        'layers_2_Dense__config__kernel_initializer__config__seed': 42
    }
    parameters = new_params
    fit_params = {'shuffle': False}

    got1 = _fit_and_score(estimator, X, y, scorer, train, test,
                          verbose=0, parameters=parameters,
                          fit_params=fit_params)

    print(got1['test_scores'])
    assert 0.68 <= round(got1['test_scores'], 2) <= 0.74, got1


def test_keras_galaxy_model_callbacks_girdisearch():

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
                                 loss='binary_crossentropy',
                                 metrics=[], batch_size=32,
                                 epochs=500, seed=42,
                                 callbacks=cbacks,
                                 verbose=0)

    scorer = get_scorer('balanced_accuracy')
    cv = KFold(n_splits=5)

    new_params = {
        'layers_1_Dense__config__kernel_initializer__config__seed': [42],
        'layers_2_Dense__config__kernel_initializer__config__seed': [42]
    }
    fit_params = {'shuffle': False}

    grid = GridSearchCV(estimator, param_grid=new_params, scoring=scorer,
                        cv=cv, n_jobs=2, refit=False, error_score='raise')

    grid.fit(X, y, **fit_params)

    got1 = grid.best_score_

    assert 0.60 <= round(got1, 2) <= 0.75, got1


def test_keras_fasta_batch_classifier():
    config = model.get_config()
    fasta_path = './tools/test-data/regulatory_mutations.fa'
    batch_generator = FastaDNABatchGenerator(fasta_path,
                                             seq_length=1000,
                                             seed=42)
    classifier = KerasGBatchClassifier(config, batch_generator,
                                       model_type='sequential',
                                       verbose=0)

    params = classifier.get_params()
    got = {
        key: value for key, value in params.items()
        if not key.startswith(('config', 'layers'))
        and not key.endswith('generator')
    }

    expect = {
        'amsgrad': None, 'batch_size': 32, 'beta': None,
        'beta_1': None, 'beta_2': None, 'callbacks': None,
        'centered': None, 'class_positive_factor': 1,
        'data_batch_generator__fasta_path':
            './tools/test-data/regulatory_mutations.fa',
        'data_batch_generator__seed': 42,
        'data_batch_generator__seq_length': 1000,
        'data_batch_generator__shuffle': True,
        'epochs': 1, 'epsilon': None,
        'initial_accumulator_value': None,
        'l1_regularization_strength': None,
        'l2_regularization_strength': None,
        'l2_shrinkage_regularization_strength': None,
        'learning_rate': None, 'learning_rate_power': None,
        'loss': 'binary_crossentropy', 'loss_weights': None,
        'metrics': [], 'model_type': 'sequential',
        'momentum': None, 'n_jobs': 1, 'nesterov': None,
        'optimizer': 'rmsprop', 'prediction_steps': None,
        'rho': None, 'run_eagerly': None, 'seed': None,
        'steps_per_epoch': None, 'steps_per_execution': None,
        'validation_split': 0., 'validation_steps': None,
        'verbose': 0
    }

    assert got == expect, got


def test_keras_fasta_protein_batch_classifier():

    inputs = keras.Input(shape=(500, 20), name='protein')
    x = layers.Conv1D(32, 3, activation='relu')(inputs)
    x = layers.Conv1D(64, 3, activation='relu')(x)
    block_1_output = layers.MaxPooling1D(3)(x)

    x = layers.Conv1D(64, 3, activation='relu', padding='same')(block_1_output)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    block_2_output = layers.add([x, block_1_output])

    x = layers.Conv1D(64, 3, activation='relu', padding='same')(block_2_output)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    block_3_output = layers.add([x, block_2_output])

    x = layers.Conv1D(64, 3, activation='relu')(block_3_output)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    config = model.get_config()
    fasta_path = "None"
    batch_generator = FastaProteinBatchGenerator(fasta_path,
                                                 seq_length=500,
                                                 seed=42)
    classifier = KerasGBatchClassifier(config, batch_generator,
                                       model_type='functional',
                                       batch_size=32,
                                       validation_split=0,
                                       epochs=3, seed=0,
                                       verbose=0)

    params = classifier.get_params()
    got = {}
    for key, value in params.items():
        if not key.startswith('layers') \
                and not key.startswith('config') \
                and not key.endswith('generator'):
            got[key] = value

    expect = {
        'amsgrad': None, 'batch_size': 32, 'beta': None,
        'beta_1': None, 'beta_2': None, 'callbacks': None,
        'centered': None, 'class_positive_factor': 1,
        'data_batch_generator__fasta_path': 'None',
        'data_batch_generator__seed': 42,
        'data_batch_generator__seq_length': 500,
        'data_batch_generator__shuffle': True, 'epochs': 3,
        'epsilon': None, 'initial_accumulator_value': None,
        'l1_regularization_strength': None,
        'l2_regularization_strength': None,
        'l2_shrinkage_regularization_strength': None,
        'learning_rate': None, 'learning_rate_power': None,
        'loss': 'binary_crossentropy', 'loss_weights': None,
        'metrics': [], 'model_type': 'functional', 'momentum': None,
        'n_jobs': 1, 'nesterov': None, 'optimizer': 'rmsprop',
        'prediction_steps': None, 'rho': None, 'run_eagerly': None,
        'seed': 0, 'steps_per_epoch': None,
        'steps_per_execution': None, 'validation_split': 0,
        'validation_steps': None, 'verbose': 0}
    assert got == expect, got

    cloned_clf = clone(classifier)
    new_params = {
        'data_batch_generator__fasta_path':
            './tools/test-data/uniprot_sprot_10000L.fasta'
    }
    cloned_clf.set_params(**new_params)

    # X = np.arange(560118)[:, np.newaxis]
    X1 = np.arange(1000)[:, np.newaxis]
    # y = np.random.randint(2, size=560118)
    y1 = np.random.randint(2, size=1000)
    cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)

    scoring = {
        'acc': get_scorer('accuracy'),
        'ba_acc': get_scorer('balanced_accuracy')
    }

    grid = GridSearchCV(cloned_clf, {}, cv=cv, scoring=scoring,
                        refit=False, error_score='raise')

    grid.fit(X1, y1)
    print(grid.cv_results_['mean_test_acc'])
    got = grid.cv_results_['mean_test_acc'].tolist()[0]
    assert 0.45 <= got <= 0.52, got


def test_keras_genomic_intervals_batch_classifier(
        genomic_generator, monkeypatch):
    model = Sequential([
        keras.Input(shape=(32, 4)), Conv1D(2, 3, activation='relu'),
        Flatten(), Dense(1, activation='sigmoid')])
    classifier = KerasGBatchClassifier(
        model.get_config(), clone(genomic_generator), optimizer='adam',
        batch_size=4, n_jobs=1, epochs=1, steps_per_epoch=1,
        validation_split=0.25, class_positive_factor=3,
        metrics=['accuracy'], seed=42, verbose=0)
    X = np.arange(12)[:, None]
    # Galaxy's scoring adapter obtains labels from the genomic generator.
    monkeypatch.setattr(_search, '_fit_and_score', _fit_and_score)
    grid = GridSearchCV(clone(classifier), {}, scoring='balanced_accuracy',
                        cv=ShuffleSplit(1, test_size=0.5, random_state=123),
                        refit=True, error_score='raise', n_jobs=1)
    try:
        grid.fit(X)
        score = grid.cv_results_['mean_test_score'][0]
        assert np.isfinite(score) and 0 <= score <= 1
        assert grid.best_estimator_.classes_.tolist() == [0, 1]
        probabilities = grid.best_estimator_.predict_proba(X)
        assert probabilities.shape == (12, 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    finally:
        if hasattr(grid, 'best_estimator_'):
            grid.best_estimator_.data_generator_.close()


def test_meric_callback():
    mcb = MetricCallback()
    params = mcb.get_params()
    expect = {'scorer': 'roc_auc'}

    assert params == expect, params

    validation_data = (X, y)
    setattr(mcb, 'validation_data', validation_data)
    x_val, y_val = mcb.validation_data

    assert np.array_equal(x_val, X)
    assert np.array_equal(y_val, y)


def test_predict_generator(genomic_generator, tmp_path):
    model = Sequential([
        keras.Input(shape=(32, 4)), Flatten(), Dense(1, activation='sigmoid')])
    clf = KerasGBatchClassifier(
        model.get_config(), clone(genomic_generator), optimizer='sgd',
        batch_size=4, n_jobs=1, epochs=1, steps_per_epoch=3,
        seed=42, verbose=0)
    X = np.arange(12)[:, None]
    try:
        clf.fit(X)
        # Two batches include a partial final batch and preserve label order.
        subset = X[:6]
        preds, labels = _predict_generator(
            clf.model_, genomic_generator.flow(subset, batch_size=4), steps=2)
        assert preds.shape == labels.shape == (6, 1)
        assert np.isfinite(preds).all()
        assert ((preds >= 0) & (preds <= 1)).all()
        np.testing.assert_array_equal(labels[:, 0], [1, 0, 1, 0, 1, 0])
        direct_X, direct_y = next(genomic_generator.flow(subset, batch_size=6))
        np.testing.assert_allclose(
            preds, clf.model_.predict_on_batch(direct_X), atol=1e-7)
        np.testing.assert_array_equal(labels, direct_y)

        path = tmp_path / 'batch_classifier.h5'
        clf.save_model(path)
        with h5py.File(path, 'r') as h:
            assert (h['class_name'][()].decode('utf8')
                    == 'KerasGBatchClassifier')
            params = json.loads(h['params'][()].decode('utf8'))
            assert params.get('data_batch_generator') is None
        restored = load_model(path)
        preds_2, labels_2 = _predict_generator(
            restored.model_, genomic_generator.flow(subset, batch_size=4))
        np.testing.assert_allclose(preds_2, preds, atol=1e-7)
        np.testing.assert_array_equal(labels_2, labels)
    finally:
        if hasattr(clf, 'data_generator_'):
            clf.data_generator_.close()


def test_multi_dimensional_output():
    # Exercise one-hot multiclass targets without downloading MNIST.
    X = np.random.RandomState(42).normal(size=(24, 16)).astype('float32')
    y = to_categorical(np.tile(np.arange(3), 8), num_classes=3)
    model = Sequential([
        keras.Input(shape=(16,)), Reshape((4, 4, 1)),
        Conv2D(2, kernel_size=3, activation='relu', padding='same'),
        MaxPooling2D((2, 2)), Flatten(), Dense(3, activation='softmax')])
    classifier = KerasGClassifier(
        model.get_config(), optimizer='adam', loss='categorical_crossentropy',
        metrics=['accuracy'], epochs=1, batch_size=6, seed=42, verbose=0,
        validation_split=0.25)
    classifier.fit(X[:18], y[:18])
    predicted = classifier.predict(X[18:])
    probabilities = classifier.predict_proba(X[18:])
    assert predicted.shape == (6,)
    assert probabilities.shape == (6, 3)
    np.testing.assert_array_equal(classifier.classes_, [0, 1, 2])
    np.testing.assert_array_equal(predicted, probabilities.argmax(axis=1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)


def test_model_save_and_load():
    df = pd.read_csv(
        './tools/test-data/pima-indians-diabetes.csv', sep=',')

    X = df.iloc[:, 0:8].values.astype(float)
    y = df.iloc[:, 8].values

    train_model = Sequential()
    train_model.add(Dense(12, input_dim=8, activation='relu'))
    train_model.add(Dense(1, activation='sigmoid'))

    config = train_model.get_config()

    clf = KerasGClassifier(config, loss='binary_crossentropy',
                           seed=42, batch_size=32)
    clf.fit(X, y, )

    _, tmp = tempfile.mkstemp()

    try:
        clf.save_model(tmp)

        with h5py.File(tmp, 'r') as h:
            assert set(h.keys()) == {'class_name', 'params',
                                     'weights', 'attributes'}, h.keys()
            assert h['class_name'][()].decode() == 'KerasGClassifier'
            params = json.loads(h['params'][()].decode('utf8'))
            assert params['loss'] == 'binary_crossentropy'
            assert params['seed'] == 42

            model = load_model(h)
        assert hasattr(model, 'model_')
    finally:
        os.remove(tmp)

    np.array_equal(clf.predict(X), model.predict(X))
