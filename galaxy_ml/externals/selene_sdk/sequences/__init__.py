"""
This module provides the types for representing biological sequences.
"""

from . import _sequence
from .genome import Genome
from .proteome import Proteome
from .sequence import Sequence
from .sequence import encoding_to_sequence
from .sequence import get_reverse_encoding
from .sequence import sequence_to_encoding


__all__ = ["Sequence", "Genome", "Proteome", "sequence_to_encoding",
           "encoding_to_sequence", "get_reverse_encoding"]
