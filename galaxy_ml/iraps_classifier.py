"""
This module has been moved. For backward compatibility, links are still
provided.
"""
from .binarize_target import (
    BinarizeTargetClassifier,
    BinarizeTargetRegressor,
    BinarizeTargetTransformer,
    IRAPSClassifier,
    IRAPSCore,
    binarize_auc_scorer,
    binarize_average_precision_scorer,
)


__all__ = ('IRAPSCore', 'IRAPSClassifier', 'binarize_auc_scorer',
           'binarize_average_precision_scorer', 'BinarizeTargetClassifier',
           'BinarizeTargetRegressor', 'BinarizeTargetTransformer')
