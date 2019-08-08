import numpy as np
import warnings

from sklearn.base import clone

from galaxy_ml.externals.selene_sdk.sequences._sequence import\
    _fast_sequence_to_encoding
from galaxy_ml.preprocessors import GenomeOneHotEncoder
from galaxy_ml.preprocessors import FastaIterator, FastaToArrayIterator
from galaxy_ml.preprocessors import FastaDNABatchGenerator
from galaxy_ml.preprocessors import FastaProteinBatchGenerator
from galaxy_ml.preprocessors import ProteinOneHotEncoder
from galaxy_ml.preprocessors import GenomicIntervalBatchGenerator
from galaxy_ml.preprocessors import GenomicVariantBatchGenerator

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
    generator.fit()
    X = np.arange(2, 8)[:, np.newaxis]
    y = np.array([1, 0, 0, 1, 0, 1])
    toarray_iterator = FastaToArrayIterator(
        X, generator, y=y, seed=42)

    params = list(toarray_iterator.get_params().keys())

    expect1 = ['X', 'batch_size', 'generator__fasta_path',
               'generator__seed', 'generator__seq_length',
               'generator__shuffle', 'generator', 'sample_weight',
               'seed', 'shuffle', 'steps', 'y']

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
    generator.fit()
    X = np.arange(2, 8)[:, np.newaxis]
    y = np.array([1, 0, 0, 1, 0, 1])
    toarray_iterator = FastaToArrayIterator(
        X, generator, y=y, seed=42)

    arr0 = generator.apply_transform(0)
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
    assert np.array_equal(batch_y,  np.array([1, 0, 1])), batch_y

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

    cloned_generator = clone(generator)


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


def test_genomic_interval_batch_generator():
    # selene case1 genome file, file not uploaded
    ref_genome_path = '/projects/selene/manuscript/case1/data/'\
        'GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta'
    intervals_path = './tools/test-data/hg38_TF_intervals_2000.txt'
    # selene case1 target bed file, file not uploaded
    target_path = '/projects/selene/manuscript/case1/data/'\
        'GATA1_proery_bm.bed.gz'
    seed = 42
    random_state = 0

    generator = GenomicIntervalBatchGenerator(
        ref_genome_path=ref_genome_path,
        intervals_path=intervals_path,
        target_path=target_path,
        seed=seed,
        features=['Proery_BM|GATA1'],
        random_state=random_state
    )
    generator1 = clone(generator)
    got = list(generator1.get_params().keys())
    expect = ['blacklist_regions', 'center_bin_to_predict',
              'feature_thresholds', 'features', 'intervals_path',
              'random_state', 'ref_genome_path', 'seed',
              'seq_length', 'shuffle', 'target_path']

    assert got == expect, got

    generator1.fit()

    features_ = generator1.features_
    n_features_ = generator1.n_features_
    bin_radius_ = generator1.bin_radius_
    start_radius_ = generator1.start_radius_
    end_radius_ = generator1.end_radius_
    surrounding_sequence_radius_ = generator1.surrounding_sequence_radius_
    target_ = generator1.target_
    sample_from_intervals_ = generator1.sample_from_intervals_
    intervals_lengths_ = generator1.interval_lengths_

    # test fit()
    assert features_ == ['Proery_BM|GATA1'], features_
    assert n_features_ == 1, n_features_
    assert bin_radius_ == 100, bin_radius_
    assert start_radius_ == 100, start_radius_
    assert end_radius_ == 100, end_radius_
    assert surrounding_sequence_radius_ == 400, surrounding_sequence_radius_
    assert target_.__class__.__name__ == 'GenomicFeatures', \
        target_.__class__.__name__
    assert target_._feature_thresholds_vec == [0.5], \
        target_._feature_thresholds_vec
    assert target_.feature_thresholds == {'Proery_BM|GATA1': 0.5}, \
        target_.feature_thresholds
    assert len(sample_from_intervals_) == 1878, len(sample_from_intervals_)
    assert sample_from_intervals_[0] == ('chr16', 19859514, 19860150), \
        sample_from_intervals_[0]
    assert len(intervals_lengths_) == 1878, len(intervals_lengths_)
    assert intervals_lengths_[0] == 636, intervals_lengths_[0]

    # test get_indices_and_probabilities()
    X = np.arange(2, 10)[:, np.newaxis]
    indices, weights = generator1.get_indices_and_probabilities(X)

    assert np.array_equal(indices, np.array([2, 3, 4, 5, 6, 7, 8, 9])),\
        indices
    assert [round(w, 3) for w in weights] == \
        [0.193, 0.023, 0.132, 0.049, 0.065, 0.195, 0.284, 0.058], weights

    # test flow()
    gen_flow = generator1.flow(X, batch_size=4)
    batch_X, batch_y = next(gen_flow)

    assert len(gen_flow) == 2, len(gen_flow)
    assert batch_X.shape == (4, 1000, 4), batch_X.shape
    assert batch_X[0][2].tolist() == [0, 0, 1, 0], batch_X[0][2]
    assert batch_X[2][4].tolist() == [0, 1, 0, 0], batch_X[2][4]
    assert batch_X[3][5].tolist() == [1, 0, 0, 0], batch_X[3][5]
    assert batch_y.tolist() == [[0], [0], [1], [0]], batch_y

    batch_X, batch_y = next(gen_flow)

    assert batch_X.shape == (4, 1000, 4), batch_X.shape
    assert batch_X[0][2].tolist() == [1, 0, 0, 0], batch_X[0][2]
    assert batch_X[2][4].tolist() == [0, 0, 0, 1], batch_X[2][4]
    assert batch_X[3][5].tolist() == [0, 0, 1, 0], batch_X[3][5]
    assert batch_y.tolist() == [[0], [0], [0], [1]], batch_y

    # test sample()
    retrieved_seq_encodings, targets = generator1.sample(X, sample_size=10)

    assert retrieved_seq_encodings.shape == (10, 1000, 4),\
        retrieved_seq_encodings.shape
    assert retrieved_seq_encodings[0][2].tolist() == [0, 1, 0, 0],\
        retrieved_seq_encodings[0][2]
    assert retrieved_seq_encodings[1][4].tolist() == [0, 0, 0, 1],\
        retrieved_seq_encodings[1][4]
    assert retrieved_seq_encodings[2][5].tolist() == [1, 0, 0, 0],\
        retrieved_seq_encodings[2][5]
    assert targets.tolist() == \
        [[0], [1], [0], [0], [0], [0], [0], [0], [0], [1]], targets

    # test steps_per_epoch
    generator2 = clone(generator)
    generator2.fit()
    gen_flow2 = generator2.flow(X, batch_size=2)

    index_arr = next(gen_flow2.index_generator)
    assert index_arr.tolist() == [3, 7], index_arr


def test_genomic_variant_batch_generator():
    # selene case2 and 3 genome file, file not uploaded
    ref_genome_path = "/projects/selene/manuscript/case3/"\
        "1_variant_effect_prediction/data/male.hg19.fasta"
    vcf_path = "./tools/test-data/lt0.05_igap_100.vcf"

    generator = GenomicVariantBatchGenerator(
        ref_genome_path=ref_genome_path, vcf_path=vcf_path,
        blacklist_regions='hg19', output_reference=False)

    generator1 = clone(generator)
    got = list(generator1.get_params().keys())
    expect = ['blacklist_regions', 'output_reference',
              'ref_genome_path', 'seq_length', 'vcf_path']

    assert got == expect, got

    generator1.fit()

    reference_genome_ = generator1.reference_genome_
    start_radius_ = generator1.start_radius_
    end_radius_ = generator1.end_radius_
    variants = generator1.variants

    assert reference_genome_.__class__.__name__ == 'Genome'
    assert start_radius_ == 500, start_radius_
    assert end_radius_ == 500, end_radius_
    assert len(variants) == 101, len(variants)

    gen_flow = generator1.flow(batch_size=4)

    n_batches = len(gen_flow)
    batch_X = next(gen_flow)
    with np.load('./tools/test-data/vcf_batch1.npz', 'r') as data:
        expect_X = data['arr_0']

    assert n_batches == 26, n_batches
    assert np.array_equal(batch_X, expect_X), batch_X
