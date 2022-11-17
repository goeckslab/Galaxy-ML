from keras.utils import Sequence

from keras_preprocessing.image import Iterator

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import indexable


class FastaIterator(Iterator, BaseEstimator, Sequence):
    """Base class for fasta sequence iterators.

    Parameters
    ----------
    n : int
        Total number of samples
    batch_size : int
        Size of batch
    shuffle : bool
        Whether to shuffle data between epoch
    seed : int
        Random seed number for data shuffling
    """
    white_list_formats = {'fasta', 'fa'}

    def __init__(self, n, batch_size=32, shuffle=True, seed=0):
        super(FastaIterator, self).__init__(n, batch_size, shuffle, seed)


class FastaToArrayIterator(FastaIterator):
    """Iterator yielding Numpy array from fasta sequences

    Parameters
    ----------
    X : array
        Contains sequence indexes in the fasta file
    generator : fitted object
        instance of BatchGenerator, e.g.,  FastaDNABatchGenerator
        or FastaProteinBatchGenerator
    y : array
        Target labels or values
    batch_size : int, default=32
    shuffle : bool, default=True
        Whether to shuffle the data between epochs
    sample_weight : None or array
        Sample weight
    seed : int
        Random seed for data shuffling
    """
    def __init__(
        self, X, generator, y=None, batch_size=32,
        shuffle=True, sample_weight=None, seed=None,
    ):
        X, y, sample_weight = indexable(X, y, sample_weight)
        self.X = X
        self.generator = generator
        self.y = y
        self.sample_weight = sample_weight

        super(FastaToArrayIterator, self).__init__(
            X.shape[0], batch_size, shuffle, seed
        )

    def _get_batches_of_transformed_samples(self, index_array):
        generator = self.generator
        index_array = np.asarray(index_array)
        n_samples = index_array.shape[0]
        batch_x = np.zeros((n_samples,
                            generator.seq_length,
                            generator.n_bases), dtype='float32')
        for i in np.arange(n_samples):
            seq_idx = int(self.X[index_array[i], 0])
            batch_x[i] = generator.apply_transform(seq_idx)

        output = (batch_x,)

        if self.y is None:
            return output[0]

        output += (self.y[index_array],)
        if self.sample_weight is not None:
            output += (self.sample_weight[index_array],)

        return output
