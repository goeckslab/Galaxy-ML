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


__all__ = ('IRAPSCore', 'IRAPSClassifier', 'binarize_auc_scorer',
           'binarize_average_precision_scorer', 'BinarizeTargetClassifier',
           'BinarizeTargetRegressor', 'BinarizeTargetTransformer')
