import os
import sys

from galaxy_ml import binarize_target
from galaxy_ml import feature_selectors
from galaxy_ml import keras_galaxy_models
from galaxy_ml import model_persist
from galaxy_ml import model_validations
from galaxy_ml import preprocessors
from galaxy_ml import utils


galaxy_ml_dir = os.path.dirname(os.path.dirname(__file__))

sys.path.append(galaxy_ml_dir)


PAGES = [
    {
        'page': 'APIs/preprocessors.md',
        'classes': [
            preprocessors.Z_RandomOverSampler,
            preprocessors.TDMScaler,
            preprocessors.GenomeOneHotEncoder,
            preprocessors.ProteinOneHotEncoder,
            preprocessors.FastaIterator,
            preprocessors.FastaToArrayIterator,
            preprocessors.FastaDNABatchGenerator,
            preprocessors.FastaRNABatchGenerator,
            preprocessors.FastaProteinBatchGenerator,
            preprocessors.IntervalsToArrayIterator,
            preprocessors.GenomicIntervalBatchGenerator,
            preprocessors.GenomicVariantBatchGenerator,
            preprocessors.ImageDataFrameBatchGenerator
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
            binarize_target.IRAPSCore,
            binarize_target.IRAPSClassifier
        ]
    },
    {
        'page': 'APIs/binarize-target.md',
        'classes': [
            binarize_target.BinarizeTargetClassifier,
            binarize_target.BinarizeTargetRegressor,
            binarize_target.BinarizeTargetTransformer,
            binarize_target._BinarizeTargetThresholdScorer
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
            model_persist.ModelToDict,
            model_persist.DictToModel,
            model_persist.ModelToHDF5,
            model_persist.HDF5ToModel
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
