"""
This module has been moved. For backward compatibility, links are still
provided.
"""
from .binarize_target import (IRAPSCore, IRAPSClassifier,
                              BinarizeTargetClassifier,
                              BinarizeTargetRegressor,
                              BinarizeTargetTransformer,
                              binarize_auc_scorer,
                              binarize_average_precision_scorer)

from .binarize_target import _BinarizeTargetThresholdScorer \
                        as _BinarizeTargetProbaScorer


__all__ = ('IRAPSCore', 'IRAPSClassifier', 'binarize_auc_scorer',
           'binarize_average_precision_scorer', 'BinarizeTargetClassifier',
           'BinarizeTargetRegressor', 'BinarizeTargetTransformer',
           '_BinarizeTargetProbaScorer')
