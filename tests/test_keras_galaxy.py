import warnings
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras_galaxy_models import (_get_params_from_dict, _param_to_dict, _update_dict,
                                SearchParam, KerasLayers, BaseKerasModel, KerasGClassifier,
                                KerasGRegressor)
from sklearn.base import clone

warnings.simplefilter('ignore')

model = Sequential()
model.add(Dense(64))
model.add(Activation('tanh'))
model.add(Activation('tanh'))
model.add(Dense(32))

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
                         'bias_constraint': None}
    }
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