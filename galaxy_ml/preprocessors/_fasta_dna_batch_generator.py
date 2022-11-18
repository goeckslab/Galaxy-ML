import numpy as np

import pyfaidx

from sklearn.base import BaseEstimator

from ._fasta_iterator import FastaToArrayIterator
from ..externals import selene_sdk


class FastaDNABatchGenerator(BaseEstimator):
    """Fasta squence batch data generator, online transformation of sequences
        to array.

    Parameters
    ----------
    fasta_path : str
        File path to fasta file.
    seq_length : int, default=1000
        Sequence length, number of bases.
    shuffle : bool, default=True
        Whether to shuffle the data between epochs
    seed : int
        Random seed for data shuffling
    """
    def __init__(self, fasta_path, seq_length=1000, shuffle=True, seed=None):
        self.fasta_path = fasta_path
        self.seq_length = seq_length
        self.shuffle = shuffle
        self.seed = seed
        self.n_bases = 4
        self.BASE_TO_INDEX = {
            'A': 0, 'C': 1, 'G': 2, 'T': 3,
            'a': 0, 'c': 1, 'g': 2, 't': 3,
        }
        self.UNK_BASE = 'N'

    def set_processing_attrs(self):
        self.fasta_file_ = pyfaidx.Fasta(self.fasta_path)
        return self

    def flow(self, X, y=None, batch_size=32, sample_weight=None,
             shuffle=None):
        if not hasattr(self, 'fasta_file_'):
            self.set_processing_attrs()

        if shuffle is None:
            shuffle = self.shuffle
        return FastaToArrayIterator(
            X, self, y=y, batch_size=batch_size,
            sample_weight=sample_weight,
            shuffle=shuffle,
            seed=self.seed)

    def apply_transform(self, idx):
        cur_sequence = str(self.fasta_file_[idx])
        if len(cur_sequence) > self.seq_length:
            cur_sequence = selene_sdk.predict._common._truncate_sequence(
                cur_sequence,
                self.seq_length)

        elif len(cur_sequence) < self.seq_length:
            cur_sequence = selene_sdk.predict._common._pad_sequence(
                cur_sequence,
                self.seq_length,
                self.UNK_BASE)

        cur_sequence_encodeing = selene_sdk.sequences._sequence.\
            _fast_sequence_to_encoding(
                cur_sequence,
                self.BASE_TO_INDEX,
                self.n_bases)

        return cur_sequence_encodeing

    def sample(self, X, y=None, sample_size=None):
        """Output the number of sample arrays and their target values,
        excluding bad samples. If reaching the end of sample index,
        go back to index 0, until the number of sample_size is met.
        For prediction, validation purpose.

        Parameters
        ----------
        X : 2-D array like
            Array of interval indices.
        y : array, default None
            Target values
        sample_size : int or None
            The number of output samples. If None,
            sample_size = X.shape[0]

        Returns
        -------
        (retrieved_sequences, targets)
        """
        if not sample_size:
            sample_size = X.shape[0]

        retrieved_X = np.zeros(
            (sample_size, self.seq_length, self.n_bases), dtype='float32')

        n_samples_drawn = 0
        i = 0
        while n_samples_drawn < sample_size:
            seq_idx = int(X[i, 0])
            seq_encoding = self.apply_transform(seq_idx)
            retrieved_X[i, :, :] = seq_encoding

            n_samples_drawn += 1
            i += 1

            # if reach the end of indices/epoch
            if i >= X.shape[0]:
                i = 0

        return retrieved_X, y

    def close(self):
        if hasattr(self, 'fasta_file_'):
            self.fasta_file_.close()
