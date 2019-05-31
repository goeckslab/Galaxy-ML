import numpy as np
import warnings

from galaxy_ml.externals.selene_sdk.sequences._sequence import\
    _fast_sequence_to_encoding
from galaxy_ml.preprocessors import GenomeOneHotEncoder
from galaxy_ml.preprocessors import FastaIterator, FastaToArrayIterator
from galaxy_ml.preprocessors import FastaDNABatchGenerator
from galaxy_ml.preprocessors import FastaProteinBatchGenerator

try:
    import pyfaidx
except ImportError:
    rval = __import__('os').system("pip install pyfaidx==0.5.5.2")
    if rval != 0:
        raise ImportError("module pyfaidx is not installed. "
                          "Galaxy attemped to install but failed."
                          "Please Contact Admin for manual "
                          "installation.")


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


def test_fasta_iterator():
    iterator = FastaIterator(1000)

    params = iterator.get_params()

    expect = {
        'batch_size': 32, 'n': 1000,
        'seed': 0, 'shuffle': True
    }
    assert params == expect, params

    wl_formats = iterator.white_list_formats
    assert wl_formats == {'fasta'}, wl_formats


def test_fasta_to_array_iterator_params():
    fasta_file = None
    X = np.arange(2, 8)[:, np.newaxis]
    y = np.array([1, 0, 0, 1, 0, 1])
    toarray_iterator = FastaToArrayIterator(
        X, y, fasta_file, seed=42, n_bases=20)

    params = list(toarray_iterator.get_params().keys())

    expect1 = ['X', 'batch_size', 'fasta_file',
               'n_bases', 'sample_weight', 'seed',
               'seq_length', 'shuffle', 'y']

    assert params == expect1, params

    new_params = {
        'batch_size': 30,
        'fasta_file': pyfaidx.Fasta(sequence_path),
        'n_bases': 4
    }

    toarray_iterator.set_params(**new_params)

    got1 = toarray_iterator.fasta_file
    got2 = toarray_iterator.batch_size
    got3 = toarray_iterator.n_bases
    assert type(got1) == pyfaidx.Fasta, type(got1)
    assert got2 == 30, got2
    assert got3 == 4, got3


def test_fasta_to_array_iterator_transform():

    fasta_file = pyfaidx.Fasta(sequence_path)
    X = np.arange(2, 8)[:, np.newaxis]
    y = np.array([1, 0, 0, 1, 0, 1])
    toarray_iterator = FastaToArrayIterator(
        X, y, fasta_file, seed=42, n_bases=4)

    arr0 = toarray_iterator.apply_transform(0)
    expect2 = np.array([[0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 1., 0., 0.]])
    assert arr0.shape == (1000, 4), arr0.shape

    index_array = [1, 3, 4]
    batch_X, batch_y = toarray_iterator.\
        _get_batches_of_transformed_samples(index_array)

    assert batch_X.shape == (3, 1000, 4), batch_X.shape
    assert np.array_equal(batch_y,  np.array([0, 1, 0])), batch_y


def test_fasta_dna_batch_generator():
    fasta_path = sequence_path

    generator = FastaDNABatchGenerator(fasta_path, seq_length=1000,
                                       seed=42)
    params = generator.get_params()

    expect1 = {
        'fasta_path': './test-data/regulatory_mutations.fa',
        'seed': 42, 'seq_length': 1000, 'shuffle': True}

    assert params == expect1, params

    X = np.arange(2, 8)[:, np.newaxis]
    y = np.array([1, 0, 0, 1, 0, 1])
    batch_size = 3

    seq_iterator = generator.flow(X, y, batch_size=batch_size)
    batch_X, batch_y = next(seq_iterator)

    got1 = batch_X[0][3]
    got2 = batch_X[1][4]
    got3 = batch_X[2][6]

    assert batch_X.shape == (3, 1000, 4), batch_X.shape
    assert got1.tolist() == [0., 0., 0., 1.], got1
    assert got2.tolist() == [0., 0., 1., 0.], got2
    assert got3.tolist() == [0., 1., 0., 0.], got3
    assert np.array_equal(batch_y,  np.array([1, 0, 1])), batch_y


def test_fasta_protein_batch_generator():
    fasta_path = None

    generator = FastaProteinBatchGenerator(fasta_path, seq_length=600,
                                           seed=42)
    params = generator.get_params()

    expect1 = {
        'fasta_path': None,
        'seed': 42, 'seq_length': 600, 'shuffle': True}

    assert params == expect1, params
    assert generator.n_bases == 20, generator.n_bases
