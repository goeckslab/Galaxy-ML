from ._fasta_dna_batch_generator import FastaDNABatchGenerator


class FastaRNABatchGenerator(FastaDNABatchGenerator):
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
        super(FastaRNABatchGenerator, self).__init__(
            fasta_path=fasta_path, seq_length=seq_length,
            shuffle=shuffle, seed=seed
        )
        self.n_bases = 4
        self.BASE_TO_INDEX = {
            'A': 0, 'C': 1, 'G': 2, 'U': 3,
            'a': 0, 'c': 1, 'g': 2, 'u': 3,
        }
        self.UNK_BASE = 'N'
