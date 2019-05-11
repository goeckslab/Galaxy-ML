import numpy as np
import warnings

from galaxy_ml.externals.selene_sdk.sequences._sequence import\
    _fast_sequence_to_encoding
from galaxy_ml.externals import pyfaidx
from galaxy_ml.preprocessors import GenomeOneHotEncoder


warnings.simplefilter('ignore')


sequence_path = './test-data/regulatory_mutations.fa'

BASE_TO_INDEX = {
        'A': 0, 'C': 1, 'G': 2, 'T': 3,
        'a': 0, 'c': 1, 'g': 2, 't': 3,
    }


def test_fast_sequence_to_encoding():
    fasta_file = pyfaidx.Fasta(sequence_path)

    sequences = np.vstack(
        [_fast_sequence_to_encoding(str(fast_record),
                                    BASE_TO_INDEX,
                                    4)[np.newaxis, :]
         for fast_record in fasta_file])

    expect = np.load('./test-data/sequence_encoding01.npy')

    assert np.array_equal(sequences, expect), sequences


def test_gnome_one_hot_encoder():
    coder = GenomeOneHotEncoder(padding=False)
    X = np.zeros((20, 1))
    X[:, 0] = np.arange(20)
    coder.fit(X, fasta_path=sequence_path)

    trans = coder.transform(X)

    expect = np.load('./test-data/sequence_encoding01.npy')

    assert np.array_equal(trans, expect), trans
