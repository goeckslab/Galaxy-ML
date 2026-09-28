import warnings

from galaxy_ml.externals.selene_sdk.sequences._sequence import\
    _fast_sequence_to_encoding
from galaxy_ml.preprocessors import (
    FastaDNABatchGenerator, FastaIterator, FastaProteinBatchGenerator,
    FastaToArrayIterator, GenomeOneHotEncoder, GenomicIntervalBatchGenerator,
    GenomicVariantBatchGenerator, ProteinOneHotEncoder,
)

import numpy as np

from sklearn.base import clone

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


sequence_path = './tools/test-data/regulatory_mutations.fa'

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

    expect = np.load('./tools/test-data/sequence_encoding01.npy')

    assert np.array_equal(sequences, expect), sequences


def test_gnome_one_hot_encoder():
    coder = GenomeOneHotEncoder(padding=False)
    coder1 = clone(coder)
    X = np.zeros((20, 1))
    X[:, 0] = np.arange(20)
    coder.fit(X, fasta_path=sequence_path)

    trans = coder.transform(X)

    expect = np.load('./tools/test-data/sequence_encoding01.npy')

    assert np.array_equal(trans, expect), trans

    fasta_file = pyfaidx.Fasta(sequence_path)
    X1 = np.array([str(fasta_file[i]) for i in range(20)])[:, np.newaxis]

    coder1.fit(X1)
    trans = coder1.transform(X1)

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
    assert wl_formats == {'fa', 'fasta'}, wl_formats


def test_fasta_to_array_iterator_params():
    fasta_path = sequence_path
    generator = FastaDNABatchGenerator(fasta_path)
    generator.set_processing_attrs()
    X = np.arange(2, 8)[:, np.newaxis]
    y = np.array([1, 0, 0, 1, 0, 1])
    toarray_iterator = FastaToArrayIterator(
        X, generator, y=y, seed=42)

    params = list(toarray_iterator.get_params().keys())

    expect1 = ['X', 'batch_size', 'generator__fasta_path',
               'generator__seed', 'generator__seq_length',
               'generator__shuffle', 'generator', 'sample_weight',
               'seed', 'shuffle', 'y']

    assert params == expect1, params

    new_params = {
        'batch_size': 30,
        'seed': 999,
        'generator__seq_length': 500
    }

    toarray_iterator.set_params(**new_params)

    got1 = toarray_iterator.batch_size
    got2 = toarray_iterator.seed
    got3 = toarray_iterator.generator.seq_length
    assert got1 == 30, got1
    assert got2 == 999, got2
    assert got3 == 500, got3


def test_fasta_to_array_iterator_transform():

    generator = FastaDNABatchGenerator(sequence_path)
    generator.set_processing_attrs()
    X = np.arange(2, 8)[:, np.newaxis]
    y = np.array([1, 0, 0, 1, 0, 1])
    toarray_iterator = FastaToArrayIterator(
        X, generator, y=y, seed=42)

    arr0 = generator.apply_transform(0)
    # expect2 = np.array([[0., 1., 0., 0.],
    #                     [0., 0., 1., 0.],
    #                     [0., 1., 0., 0.]])
    assert arr0.shape == (1000, 4), arr0.shape

    index_array = [1, 3, 4]
    batch_X, batch_y = toarray_iterator.\
        _get_batches_of_transformed_samples(index_array)

    assert batch_X.shape == (3, 1000, 4), batch_X.shape
    assert np.array_equal(batch_y, np.array([0, 1, 0])), batch_y


def test_fasta_dna_batch_generator():
    fasta_path = sequence_path

    generator = FastaDNABatchGenerator(fasta_path, seq_length=1000,
                                       seed=42)
    params = generator.get_params()

    expect1 = {
        'fasta_path': './tools/test-data/regulatory_mutations.fa',
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
    assert np.array_equal(batch_y, np.array([1, 0, 1])), batch_y

    # test sample method
    retrived_seq_encodings, _ = generator.sample(X, sample_size=3)

    got4 = retrived_seq_encodings[0][3]
    got5 = retrived_seq_encodings[1][4]
    got6 = retrived_seq_encodings[2][7]

    assert retrived_seq_encodings.shape == (3, 1000, 4), \
        retrived_seq_encodings.shape
    assert got4.tolist() == [0., 0., 0., 1.], got4
    assert got5.tolist() == [0., 0., 1., 0.], got5
    assert got6.tolist() == [1., 0., 0., 0.], got6
    generator.close()


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

    assert clone(generator), "Clone generator failed!"
    generator.close()


def test_protein_one_hot_encoder():
    fasta_path = './tools/test-data/uniprot_sprot_10000L.fasta'
    coder = ProteinOneHotEncoder(fasta_path=fasta_path, padding=True)
    X = np.zeros((20, 1))
    X[:, 0] = np.arange(20)

    coder.fit(X)
    trans = coder.transform(X)
    # np.save('./tools/test-data/sequence_encoding02.npy', trans)

    expect = np.load('./tools/test-data/sequence_encoding02.npy')

    assert np.array_equal(trans, expect), trans

    fasta_file = pyfaidx.Fasta(fasta_path)
    X1 = np.array([str(fasta_file[i]) for i in range(20)])[:, np.newaxis]

    coder.set_params(fasta_path=None)

    coder.fit(X1)
    trans = coder.transform(X1)

    assert np.array_equal(trans, expect), trans


def test_genomic_interval_batch_generator(genomic_generator):
    generator = clone(genomic_generator)
    try:
        assert generator.get_params() == genomic_generator.get_params()
        generator.set_processing_attrs()
        assert generator.features_ == ['binding']
        assert generator.n_features_in_ == 1
        assert generator.bin_radius_ == 4
        assert generator.start_radius_ == generator.end_radius_ == 4
        assert generator.surrounding_sequence_radius_ == 12
        assert generator.target_.feature_thresholds == {'binding': 0.5}
        np.testing.assert_array_equal(
            generator.target_._feature_thresholds_vec, [0.5])
        assert generator.sample_from_intervals_[0] == ('chr1', 100, 120)
        assert generator.interval_lengths_ == list(range(20, 68, 4))

        X = np.arange(12)[:, None]
        indices, weights = generator.get_indices_and_probabilities(X)
        np.testing.assert_array_equal(indices, X[:, 0])
        lengths = np.arange(20, 68, 4)
        np.testing.assert_allclose(weights, lengths / lengths.sum())
        flow = generator.flow(X, batch_size=5)
        assert len(flow) == 3
        for batch, size in enumerate((5, 5, 2)):
            sequences, targets = next(flow)
            assert sequences.shape == (size, 32, 4)
            for row, idx in enumerate(range(batch * 5, batch * 5 + size)):
                # Independently derive the sequence at the interval midpoint.
                start = 110 + idx * 102 - 16
                bases = 'ACGT' if idx % 2 == 0 else 'TGCA'
                expected = np.eye(4)[[BASE_TO_INDEX[bases[j % 4]]
                                      for j in range(start, start + 32)]]
                np.testing.assert_array_equal(sequences[row], expected)
                assert targets[row, 0] == (idx % 2 == 0)
        sequences, targets = generator.sample(X, sample_size=14)
        assert sequences.shape == (14, 32, 4)
        np.testing.assert_array_equal(sequences.sum(axis=2), 1)
        np.testing.assert_array_equal(targets[:, 0], [1, 0] * 7)

        shuffled = generator.flow(X, batch_size=4, shuffle=True)
        expected_indices = np.random.RandomState(42).choice(
            12, size=12, replace=True, p=weights)
        np.testing.assert_array_equal(
            next(shuffled.index_generator), expected_indices[:4])
    finally:
        generator.close()


def test_genomic_variant_batch_generator(genomic_files):
    generator = GenomicVariantBatchGenerator(
        ref_genome_path=genomic_files['ref_genome_path'],
        vcf_path=genomic_files['vcf_path'], blacklist_regions=None,
        seq_length=32)
    reference = clone(generator).set_params(output_reference=True)
    try:
        assert clone(generator).get_params() == generator.get_params()
        generator.set_processing_attrs()
        assert generator.start_radius_ == generator.end_radius_ == 16
        assert len(generator.variants) == 4
        flow = generator.flow(batch_size=3)
        ref_flow = reference.flow(batch_size=3)
        assert len(flow) == len(ref_flow) == 2
        variants = np.concatenate([next(flow), next(flow)])
        references = np.concatenate([next(ref_flow), next(ref_flow)])
        assert variants.shape == references.shape == (4, 32, 4)
        sequence = 'ACGT' * 512
        for idx, (pos, alt) in enumerate(
                ((101, 'G'), (202, 'T'), (301, 'C'), (301, 'T'))):
            bases = sequence[pos - 16:pos + 16]
            expected_ref = np.eye(4)[[BASE_TO_INDEX[b] for b in bases]]
            expected_alt = expected_ref.copy()
            expected_alt[15] = np.eye(4)[BASE_TO_INDEX[alt]]
            np.testing.assert_array_equal(references[idx], expected_ref)
            np.testing.assert_array_equal(variants[idx], expected_alt)
        assert reference.unmatches == []
    finally:
        generator.close()
        reference.close()
