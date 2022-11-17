import numpy as np

import pyfaidx

from sklearn.base import BaseEstimator, TransformerMixin

from ..externals import selene_sdk


class GenomeOneHotEncoder(BaseEstimator, TransformerMixin):
    """Convert Genomic sequences to one-hot encoded 2d array

    Paramaters
    ----------
    fasta_path : str, default None
        File path to the fasta file. There could two other ways to set up
        `fasta_path`. 1) through fit_params; 2) set_params(). If fasta_path is
        None, we suppose the sequences are contained in first column of X.
    padding : bool, default is False
        All sequences are expected to be in the same length, but sometimes not.
        If True, all sequences use the same length of first entry by either
        padding or truncating. If False, raise ValueError if different seuqnce
        lengths are found.
    seq_length : None or int
        Sequence length. If None, determined by the the first entry.
    """
    BASE_TO_INDEX = {
        'A': 0, 'C': 1, 'G': 2, 'T': 3,
        'a': 0, 'c': 1, 'g': 2, 't': 3,
    }

    UNK_BASE = 'N'

    def __init__(self, fasta_path=None, padding=True, seq_length=None):
        super(GenomeOneHotEncoder, self).__init__()
        self.fasta_path = fasta_path
        self.padding = padding
        self.seq_length = seq_length

    def fit(self, X, y=None, fasta_path=None):
        """
        Parameters
        ----------
        X : array, (n_samples, 1)
            Contains the index numbers of fasta sequnce in the fasta file.
        y : array or list
            Target values.
        fasta_path : str
            File path to the fasta file.

        Returns
        -------
        self
        """
        if fasta_path:
            self.fasta_path = fasta_path

        if self.fasta_path:
            fasta_file = pyfaidx.Fasta(self.fasta_path)
        else:
            fasta_file = None

        if not self.seq_length:
            # set up the sequence_length from the first entry
            sequence_length = len(fasta_file[int(X[0, 0])]) \
                if fasta_file else len(str(X[0, 0]))
        else:
            sequence_length = self.seq_length

        if not self.padding:
            for idx in np.arange(X.shape[0]):
                seq = fasta_file[int(X[idx, 0])] \
                    if fasta_file else str(X[idx, 0])
                if len(seq) != sequence_length:
                    raise ValueError("The first sequence record contains "
                                     "%d bases, while %s contrains %d bases"
                                     % (sequence_length,
                                        repr(seq),
                                        len(seq)))

        self.fasta_file = fasta_file
        self.seq_length_ = sequence_length
        return self

    def transform(self, X):
        """convert index in X into one-hot encoded 2d array

        Parameter
        ---------
        X : array, (n_samples, 1)
            Contains the index numbers of fasta sequnce in the fasta file.

        Returns
        -------
        Transformed X in 3d array, (n_sequences, sequence_length, 4)
        """
        # One base encodes for 4 byts
        sequences_endcoding = np.zeros((X.shape[0],
                                        self.seq_length_,
                                        4))
        for i in range(X.shape[0]):
            cur_sequence = str(self.fasta_file[int(X[i, 0])]) \
                if self.fasta_file else str(X[i, 0])

            if len(cur_sequence) > self.seq_length_:
                cur_sequence = selene_sdk.predict._common._truncate_sequence(
                    cur_sequence,
                    self.seq_length_)

            elif len(cur_sequence) < self.seq_length_:
                cur_sequence = selene_sdk.predict._common._pad_sequence(
                    cur_sequence,
                    self.seq_length_,
                    GenomeOneHotEncoder.UNK_BASE)

            cur_sequence_encodeing = selene_sdk.sequences._sequence.\
                _fast_sequence_to_encoding(
                    cur_sequence,
                    GenomeOneHotEncoder.BASE_TO_INDEX,
                    4)

            sequences_endcoding[i, :, :] = cur_sequence_encodeing

        return sequences_endcoding
