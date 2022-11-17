"""
This module contains classes and methods for target feature classes.
These are classes which define a way to access a "target feature" such
as a label or annotation on an input sequence.
"""
from .genomic_features import GenomicFeatures
from .target import Target


__all__ = ["Target", "GenomicFeatures"]
