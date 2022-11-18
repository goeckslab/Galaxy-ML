from ._fasta_dna_batch_generator import FastaDNABatchGenerator


class FastaProteinBatchGenerator(FastaDNABatchGenerator):
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
        super(FastaProteinBatchGenerator, self).__init__(
            fasta_path=fasta_path, seq_length=seq_length,
            shuffle=shuffle, seed=seed
        )
        self.n_bases = 20
        self.BASE_TO_INDEX = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6,
            'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13,
            'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
        }
        self.UNK_BASE = 'X'
