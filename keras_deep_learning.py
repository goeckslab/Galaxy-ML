import keras

from ast import literal_eval
from keras.models import Sequential, Model
from keras_galaxy_models import KerasGClassifier, KerasGRegressor
from sklearn.externals import six


def _handle_shape(literal):
    """Eval integer or list/tuple of integers from string

    Parameters:
    -----------
    literal : str.
    """
    literal = literal.strip()
    if not literal:
        return None
    try:
        return literal_eval(literal)
    except NameError as e:
        print(e)
        return literal


def _handle_regularizer(literal):
    """Construct regularizer from string literal

    Parameters
    ----------
    literal : str. E.g. '(0.1, 0)'
    """
    literal = literal.strip()
    if not literal:
        return None

    l1, l2 = literal_eval(literal)

    if not l1 and not l2:
        return None

    if l1 is None:
        l1 = 0.
    if l2 is None:
        l2 = 0.

    return keras.regularizers.l1_l2(l1=l1, l2=l2)


def _handle_constraint(config):
    """Construct constraint from galaxy tool parameters.
    Suppose correct dictionary format

    Parameters
    ----------
    config : dict. E.g.
        "bias_constraint":
            {"constraint_options":
                {"max_value":1.0,
                "min_value":0.0,
                "axis":"[0, 1, 2]"
                },
            "constraint_type":
                "MinMaxNorm"
            }
    """
    constraint_type = config['constraint_type']
    if constraint_type == 'None':
        return None

    klass = getattr(keras.constraints, constraint_type)
    options = config.get('constraint_options', {})
    if 'axis' in options:
        options['axis'] = literal_eval(options['axis'])

    return klass(**options)


def _handle_lambda(literal):
    return None


def get_sequential_model(config):
    """Construct keras Sequential model from Galaxy tool parameters

    Parameters:
    -----------
    config : dictionary, galaxy tool parameters loaded by JSON
    """
    model = Sequential()
    input_shape = _handle_shape(config['input_shape'])
    layers = config['layers']
    for layer in layers:
        options = layer['layer_selection']
        layer_type = options.pop('layer_type')
        klass = getattr(keras.layers, layer_type)
        other_options = options.pop('layer_options', {})
        options.update(other_options)
        ## parameters needs special care
        for key, value in six.iteritems(options):
            if value == 'None':
                options[key] = None
                continue

            if type(value) in [int, float, bool]\
                or (type(value) is str and value.isalpha()):
                continue

            if key in ['input_shape', 'noise_shape', 'shape', 'batch_shape','target_shape',
                        'dims', 'kernel_size', 'strides', 'dilation_rate', 'output_padding'
                        'cropping', 'size', 'padding', 'pool_size', 'axis', 'shared_axes']:
                options[key] = _handle_shape(value)
            elif key.endswith('_regularizer'):
                options[key] = _handle_regularizer(value)
            elif key.endswith('_constraint'):
                options[key] = _handle_constraint(value)
            elif key == 'function': # No support for lambda/function eval
                options.pop(key)
            elif key == 'merging_layers':
                raise ValueError("Merge layers are not supported in Sequential model. Please "
                                 "Please consider using the functional model!")
                idxs = literal_eval(value)
                options[key] = [all_layers[i-1] for i in idxs]

        # add input_shape to the first layer only
        if not getattr(model, '_layers') and input_shape is not None:
            options['input_shape'] = input_shape

        model.add(klass(**options))

    return model


def get_functional_model(layers):
    """Construct keras functional model from Galaxy tool parameters

    Parameters:
    -----------
    layers : dictionary, galaxy tool parameters loaded by JSON
    """
    return layers


if __name__ == '__main__':
    import json
    import pickle
    import sys
    import warnings

    warnings.simplefilter('ignore')
    input_json_path = sys.argv[1]
    with open(input_json_path, 'r') as param_handler:
        inputs = json.load(param_handler)

    outfile = sys.argv[2]

    if inputs['learning_type'] == 'keras_classifier':
        klass = KerasGClassifier
    else:
        klass = KerasGRegressor

    model_type = inputs['layers_config']['model_selection']['model_type']
    layers_config = inputs['layers_config']['model_selection']

    if model_type == 'sequential':
        model = get_sequential_model(layers_config)
    else:
        model = get_functional_model(layers_config)

    config = model.get_config()
    options = {}
    options['model_type'] = model_type
    options['loss'] = inputs['compile_params']['loss']
    options['optimizer'] = inputs['compile_params']['optimizer_selection']['optimizer_type'].lower()
    options.update( inputs['compile_params']['optimizer_selection']['optimizer_options'] )
    options.update(inputs['fit_params'])

    estimator = klass(config, **options)

    with open(outfile, 'wb') as f:
        pickle.dump(estimator, f, pickle.HIGHEST_PROTOCOL)