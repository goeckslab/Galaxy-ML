import os
import sys

galaxy_ml_dir = os.path.dirname(os.path.dirname(__file__))

sys.path.append(galaxy_ml_dir)

from galaxy_ml import keras_galaxy_models
from galaxy_ml import model_persist
from galaxy_ml import model_validations
from galaxy_ml import preprocessors
from galaxy_ml import utils
from galaxy_ml import iraps_classifier
from galaxy_ml import feature_selectors


PAGES = [
    {
        'page': 'APIs/preprocessors.md',
        'classes': [
            preprocessors.Z_RandomOverSampler,
            preprocessors.TDMScaler,
            preprocessors.GenomeOneHotEncoder,
            preprocessors.ProteinOneHotEncoder,
            preprocessors.ImageBatchGenerator,
            preprocessors.FastaIterator,
            preprocessors.FastaToArrayIterator,
            preprocessors.FastaDNABatchGenerator,
            preprocessors.FastaRNABatchGenerator,
            preprocessors.FastaProteinBatchGenerator

        ]
    },
    {
        'page': 'APIs/feature-selectors.md',
        'classes': [
            feature_selectors.DyRFE,
            feature_selectors.DyRFECV
        ]
    },
    {
        'page': 'APIs/iraps-classifier.md',
        'classes': [
            iraps_classifier.IRAPSCore,
            iraps_classifier.IRAPSClassifier
        ]
    },
    {
        'page': 'APIs/binarize-target.md',
        'classes': [
            iraps_classifier.BinarizeTargetClassifier,
            iraps_classifier.BinarizeTargetRegressor,
            iraps_classifier.BinarizeTargetTransformer
        ]
    },
    {
        'page': 'APIs/keras-galaxy-models.md',
        'classes': [
            keras_galaxy_models.SearchParam,
            keras_galaxy_models.KerasLayers,
            keras_galaxy_models.BaseKerasModel,
            keras_galaxy_models.KerasGClassifier,
            keras_galaxy_models.KerasGRegressor,
            keras_galaxy_models.KerasGBatchClassifier
        ]
    },
    {
        'page': 'APIs/model-validations.md',
        'classes': [
            model_validations.OrderedKFold,
            model_validations.RepeatedOrderedKFold
        ]
    },
    {
        'page': 'APIs/model-persistent.md',
        'classes': [
            model_persist.JPicklerError,
            model_persist.ModelToDict,
            model_persist.DictToModel
        ]
    },
    {
        'page': 'APIs/utils.md',
        'classes': [
            utils._SafePickler,
            utils.SafeEval
        ]
    }
]
