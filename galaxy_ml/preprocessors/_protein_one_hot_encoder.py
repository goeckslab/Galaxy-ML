import numpy as np

from ._genome_one_hot_encoder import GenomeOneHotEncoder
from ..externals import selene_sdk


class ProteinOneHotEncoder(GenomeOneHotEncoder):
    """Convert protein sequences to one-hot encoded 2d array

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
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6,
        'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13,
        'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
    }

    UNK_BASE = 'X'

    def transform(self, X):
        """convert index in X into one-hot encoded 2d array

        Parameter
        ---------
        X : array, (n_samples, 1)
            Contains the index numbers of fasta sequnce in the fasta file.

        Returns
        -------
        Transformed X in 3d array, (n_sequences, sequence_length, 20)
        """
        # One base encodes for 4 byts
        sequences_endcoding = np.zeros((X.shape[0],
                                        self.seq_length_,
                                        20))
        for i in range(X.shape[0]):
            cur_sequence = str(self.fasta_file[int(X[i, 0])]) \
                if self.fasta_file else str(X[i, 0])

            cur_sequence = str(cur_sequence)
            if len(cur_sequence) > self.seq_length_:
                cur_sequence = selene_sdk.predict._common._truncate_sequence(
                    cur_sequence,
                    self.seq_length_)

            elif len(cur_sequence) < self.seq_length_:
                cur_sequence = selene_sdk.predict._common._pad_sequence(
                    cur_sequence,
                    self.seq_length_,
                    ProteinOneHotEncoder.UNK_BASE)

            cur_sequence_encodeing = selene_sdk.sequences._sequence.\
                _fast_sequence_to_encoding(
                    cur_sequence,
                    ProteinOneHotEncoder.BASE_TO_INDEX,
                    20)

            sequences_endcoding[i, :, :] = cur_sequence_encodeing

        return sequences_endcoding
