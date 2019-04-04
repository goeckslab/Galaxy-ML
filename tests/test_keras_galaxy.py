import warnings
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras_galaxy_models import _get_params_from_dict

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
    d = model.get_config()
    got = list(_get_params_from_dict(d, 'd').keys())
    expect = ['d__name', 'd__layers', 'd__layers_0', 'd__layers_1', 'd__layers_2',
              'd__layers_3', 'd__layers_0__class_name', 'd__layers_0_Dense__config',
              'd__layers_0_Dense__config__name', 'd__layers_0_Dense__config__trainable',
              'd__layers_0_Dense__config__units', 'd__layers_0_Dense__config__activation',
              'd__layers_0_Dense__config__use_bias', 'd__layers_0_Dense__config__kernel_initializer',
              'd__layers_0_Dense__config__kernel_initializer__class_name',
              'd__layers_0_Dense__config__kernel_initializer_VarianceScaling__config',
              'd__layers_0_Dense__config__kernel_initializer_VarianceScaling__config__scale',
              'd__layers_0_Dense__config__kernel_initializer_VarianceScaling__config__mode',
              'd__layers_0_Dense__config__kernel_initializer_VarianceScaling__config__distribution',
              'd__layers_0_Dense__config__kernel_initializer_VarianceScaling__config__seed',
              'd__layers_0_Dense__config__bias_initializer', 'd__layers_0_Dense__config__bias_initializer__class_name',
              'd__layers_0_Dense__config__bias_initializer_Zeros__config',
              'd__layers_0_Dense__config__kernel_regularizer', 'd__layers_0_Dense__config__bias_regularizer',
              'd__layers_0_Dense__config__activity_regularizer', 'd__layers_0_Dense__config__kernel_constraint',
              'd__layers_0_Dense__config__bias_constraint', 'd__layers_1__class_name',
              'd__layers_1_Activation__config', 'd__layers_1_Activation__config__name',
              'd__layers_1_Activation__config__trainable', 'd__layers_1_Activation__config__activation',
              'd__layers_2__class_name', 'd__layers_2_Activation__config', 'd__layers_2_Activation__config__name',
              'd__layers_2_Activation__config__trainable', 'd__layers_2_Activation__config__activation',
              'd__layers_3__class_name', 'd__layers_3_Dense__config', 'd__layers_3_Dense__config__name',
              'd__layers_3_Dense__config__trainable', 'd__layers_3_Dense__config__units',
              'd__layers_3_Dense__config__activation', 'd__layers_3_Dense__config__use_bias',
              'd__layers_3_Dense__config__kernel_initializer',
              'd__layers_3_Dense__config__kernel_initializer__class_name',
              'd__layers_3_Dense__config__kernel_initializer_VarianceScaling__config',
              'd__layers_3_Dense__config__kernel_initializer_VarianceScaling__config__scale',
              'd__layers_3_Dense__config__kernel_initializer_VarianceScaling__config__mode',
              'd__layers_3_Dense__config__kernel_initializer_VarianceScaling__config__distribution',
              'd__layers_3_Dense__config__kernel_initializer_VarianceScaling__config__seed',
              'd__layers_3_Dense__config__bias_initializer',
              'd__layers_3_Dense__config__bias_initializer__class_name',
              'd__layers_3_Dense__config__bias_initializer_Zeros__config',
              'd__layers_3_Dense__config__kernel_regularizer', 'd__layers_3_Dense__config__bias_regularizer',
              'd__layers_3_Dense__config__activity_regularizer', 'd__layers_3_Dense__config__kernel_constraint',
              'd__layers_3_Dense__config__bias_constraint']
    assert got == expect, got


