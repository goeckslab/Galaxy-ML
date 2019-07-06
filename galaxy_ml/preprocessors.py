"""
Z_RandomOverSampler
"""
import numpy as np
import pyfaidx
from collections import Counter
from keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.utils import check_target_type
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing.data import _handle_zeros_in_scale
from sklearn.utils import check_array, safe_indexing, check_random_state
from sklearn.utils.validation import (check_is_fitted, check_X_y,
                                      FLOAT_DTYPES)

from galaxy_ml.externals import selene_sdk


__all__ = ('Z_RandomOverSampler', 'TDMScaler', 'GenomeOneHotEncoder',
           'ProteinOneHotEncoder', 'ImageBatchGenerator',
           'FastaIterator', 'FastaToArrayIterator',
           'FastaDNABatchGenerator', 'FastaRNABatchGenerator',
           'FastaProteinBatchGenerator', 'IntervalsToArrayIterator',
           'GenomicIntervalBatchGenerator')


class Z_RandomOverSampler(BaseOverSampler):

    def __init__(self, sampling_strategy='auto',
                 return_indices=False,
                 random_state=None,
                 ratio=None,
                 negative_thres=0,
                 positive_thres=-1):
        super(Z_RandomOverSampler, self).__init__(
            sampling_strategy=sampling_strategy, ratio=ratio)
        self.random_state = random_state
        self.return_indices = return_indices
        self.negative_thres = negative_thres
        self.positive_thres = positive_thres

    @staticmethod
    def _check_X_y(X, y):
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'], dtype=None)
        return X, y, binarize_y

    def _fit_resample(self, X, y):
        n_samples = X.shape[0]

        # convert y to z_score
        y_z = (y - y.mean()) / y.std()

        index0 = np.arange(n_samples)
        index_negative = index0[y_z > self.negative_thres]
        index_positive = index0[y_z <= self.positive_thres]
        index_unclassified = [x for x in index0
                              if x not in index_negative
                              and x not in index_positive]

        y_z[index_negative] = 0
        y_z[index_positive] = 1
        y_z[index_unclassified] = -1

        ros = RandomOverSampler(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            ratio=self.ratio)
        _, _ = ros.fit_resample(X, y_z)
        sample_indices = ros.sample_indices_

        print("Before sampler: %s. Total after: %s"
              % (Counter(y_z), sample_indices.shape))

        self.sample_indices_ = np.array(sample_indices)

        if self.return_indices:
            return (safe_indexing(X, sample_indices),
                    safe_indexing(y, sample_indices),
                    sample_indices)
        return (safe_indexing(X, sample_indices),
                safe_indexing(y, sample_indices))


def _get_quantiles(X, quantile_range):
    """
    Calculate column percentiles for 2d array

    Parameters
    ----------
    X : array-like, shape [n_samples, n_features]
    """
    quantiles = []
    for feature_idx in range(X.shape[1]):
        if sparse.issparse(X):
            column_nnz_data = X.data[
                X.indptr[feature_idx]: X.indptr[feature_idx + 1]]
            column_data = np.zeros(shape=X.shape[0], dtype=X.dtype)
            column_data[:len(column_nnz_data)] = column_nnz_data
        else:
            column_data = X[:, feature_idx]
        quantiles.append(np.nanpercentile(column_data, quantile_range))

    quantiles = np.transpose(quantiles)

    return quantiles


class TDMScaler(BaseEstimator, TransformerMixin):
    """
    Scale features using Training Distribution Matching (TDM) algorithm

    References
    ----------
    .. [1] Thompson JA, Tan J and Greene CS (2016) Cross-platform
           normalization of microarray and RNA-seq data for machine
           learning applications. PeerJ 4, e1621.
    """

    def __init__(self, q_lower=25.0, q_upper=75.0, ):
        self.q_lower = q_lower
        self.q_upper = q_upper

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
        """
        X = check_array(X, copy=True, estimator=self, dtype=FLOAT_DTYPES,
                        force_all_finite=True)

        if not 0 <= self.q_lower <= self.q_upper <= 100:
            raise ValueError("Invalid quantile parameter values: "
                             "q_lower %s, q_upper: %s"
                             % (str(self.q_lower), str(self.q_upper)))

        # TODO sparse data
        quantiles = np.nanpercentile(X, (self.q_lower, self.q_upper))
        iqr = quantiles[1] - quantiles[0]

        self.q_lower_ = quantiles[0]
        self.q_upper_ = quantiles[1]
        self.iqr_ = _handle_zeros_in_scale(iqr, copy=False)

        self.max_ = np.nanmax(X)
        self.min_ = np.nanmin(X)

        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix}
            The data used to scale along the specified axis.
        """
        check_is_fitted(self, 'iqr_', 'max_')
        X = check_array(X, copy=True, estimator=self, dtype=FLOAT_DTYPES,
                        force_all_finite=True)

        # TODO sparse data
        train_upper_scale = (self.max_ - self.q_upper_) / self.iqr_
        train_lower_scale = (self.q_lower_ - self.min_) / self.iqr_

        test_quantiles = np.nanpercentile(X, (self.q_lower, self.q_upper))
        test_iqr = _handle_zeros_in_scale(
            test_quantiles[1] - test_quantiles[0], copy=False)

        test_upper_bound = test_quantiles[1] + train_upper_scale * test_iqr
        test_lower_bound = test_quantiles[0] - train_lower_scale * test_iqr

        test_min = np.nanmin(X)
        if test_lower_bound < test_min:
            test_lower_bound = test_min

        X[X > test_upper_bound] = test_upper_bound
        X[X < test_lower_bound] = test_lower_bound

        X = (X - test_lower_bound) / (test_upper_bound - test_lower_bound)\
            * (self.max_ - self.min_) + self.min_

        return X

    def inverse_transform(self, X):
        """
        Scale the data back to the original state
        """
        raise NotImplementedError("Inverse transformation is not implemented!")


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


class ImageBatchGenerator(ImageDataGenerator, BaseEstimator):
    pass


##############################################################
from keras_preprocessing.image import Iterator
from keras.utils.data_utils import Sequence
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
    def __init__(self, X, generator, y=None, batch_size=32,
                 shuffle=True, sample_weight=None, seed=None,
                 steps=None):
        X, y, sample_weight = indexable(X, y, sample_weight)
        self.X = X
        self.generator = generator
        self.y = y
        self.sample_weight = sample_weight

        super(FastaToArrayIterator, self).__init__(
              X.shape[0], batch_size, shuffle, seed)

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

    def fit(self):
        self.fasta_file_ = pyfaidx.Fasta(self.fasta_path)
        return self

    def flow(self, X, y=None, batch_size=32, sample_weight=None,
             shuffle=None):
        if not hasattr(self, 'fasta_file_'):
            self.fit()

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


class IntervalsToArrayIterator(FastaToArrayIterator):
    """Iterator yielding Numpy array from intervals and reference
    sequences.

    Parameters
    ----------
    X : array
        Contains sequence indexes in the fasta file
    generator : fitted object
        instance of BatchGenerator, e.g.,  FastaDNABatchGenerator
        or FastaProteinBatchGenerator
    y : None
        The existing of y is due to inheritence, should be always None.
    batch_size : int, default=32
    shuffle : bool, default=True
        Whether to shuffle the data between epochs
    sample_weight : None or array
        Sample weight
    seed : int
        Random seed for data shuffling
    sample_probabilities : 1-D array or None, default is None.
        The probabilities to draw samples. Different from the sample
        weight, this parameter only changes the the frequency of
        sampling, won't the loss during training.
    """
    def __init__(self, X, generator, y=None, batch_size=32,
                 shuffle=True, sample_weight=None, seed=None,
                 sample_probabilities=None):
        super(IntervalsToArrayIterator, self).__init__(
            X, generator,
            y=y, batch_size=batch_size,
            shuffle=shuffle, seed=seed,
            sample_weight=sample_weight)

        self.sample_probabilities = sample_probabilities

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.choice(
                self.index_array, size=self.n,
                replace=True, p=self.sample_probabilities)

    def _get_batches_of_transformed_samples(self, index_array):
        generator = self.generator
        index_array = np.asarray(index_array)
        n_samples = index_array.shape[0]
        batch_x = np.zeros((n_samples,
                            generator.seq_length,
                            generator.n_bases), dtype='float32')
        batch_y = np.zeros((n_samples, generator.n_features_), dtype='int32')

        for i in range(n_samples):
            seq_idx = int(self.X[index_array[i]])
            rval = generator.apply_transform(seq_idx, shuffle=self.shuffle)
            # bad sample, sample next
            while rval is None:
                rand_idx = int(np.random.choice(
                    self.X[:, 0], size=1, p=self.sample_weight)[0])
                rval = generator.apply_transform(rand_idx,
                                                 shuffle=self.shuffle)
            batch_x[i, :, :] = rval[0]
            batch_y[i, :] = rval[1]

        output = (batch_x, batch_y)
        if self.sample_weight is not None:
            output += (self.sample_weight[index_array], )

        return output


class GenomicIntervalBatchGenerator(BaseEstimator):
    """Generate sequence array and target values from a reference
    genome, intervals and genomic feature dataset.
    Try to mimic the the
    `selene_sdk.samplers.interval_sampler.IntervalsSampler`.

    Parameters
    ----------
    ref_genome_path : str
        File path to the reference genomce, usually in fasta
        format.
    intervals_path : str
        File path to the intervals dataset.
    target_path : str
        File path to the dataset containing genomic features or
        target information, usually in `bed` ir `bed.gz` format.
    features : list of str or 'infer'
        A list of features to predict. If 'infer', retrieve all
        the unique features from the target file.
    blacklist_regions : str
        E.g., 'hg38'. For more info, refer to
        `selene_sdk.sequences.Genome`.
    shuffle : bool, default=True
        Whether to shuffle the data between epochs.
    seed : int or None, default=None
        Random seed for shuffling between epocks.
    seq_length : int, default=1000
        Retrived sequence length.
    center_bin_to_predict : int, default=200
        Query the tabix-indexed file for a region of length
        `center_bin_to_predict`.
    feature_thresholds : float, default=0.5
        Threshold values to determine target value.
    random_state : int or None, default=None
        Random seed for sampling sequences with changing position.
    """
    def __init__(self, ref_genome_path=None,
                 intervals_path=None, target_path=None,
                 features='infer', blacklist_regions='hg38',
                 shuffle=True, seed=None, seq_length=1000,
                 center_bin_to_predict=200,
                 feature_thresholds=0.5,
                 random_state=None):
        self.ref_genome_path = ref_genome_path
        self.intervals_path = intervals_path
        self.target_path = target_path
        self.features = features
        self.blacklist_regions = blacklist_regions
        self.shuffle = shuffle
        self.seed = seed
        self.seq_length = seq_length
        self.center_bin_to_predict = center_bin_to_predict
        self.feature_thresholds = feature_thresholds
        self.random_state = random_state
        self.STRAND_SIDES = ('+', '-')
        self.n_bases = 4

    def fit(self):
        if self.features == 'infer':
            features_set = set()
            with open(self.target_path, 'r') as features_file:
                for line in features_file:
                    feature = line.split('\t')[-1].strip()
                    features_set.add(feature)
            self.features_ = list(features_set)
        else:
            self.features_ = self.features

        self.n_features_ = len(self.features_)
        self.bin_radius_ = self.center_bin_to_predict // 2
        self.start_radius_ = self.bin_radius_
        self.end_radius_ = \
            self.bin_radius_ + self.center_bin_to_predict % 2
        surrounding_sequence_length = \
            self.seq_length - self.center_bin_to_predict
        if surrounding_sequence_length < 0:
            raise ValueError(
                "Sequence length of {0} is less than the center bin "
                "length of {1}.".format(
                    self.seq_length, self.center_bin_to_predict))
        self.surrounding_sequence_radius_ = \
            surrounding_sequence_length // 2

        self.reference_genome_ = selene_sdk.sequences.Genome(
            input_path=self.ref_genome_path,
            blacklist_regions=self.blacklist_regions)

        self.target_ = selene_sdk.targets.genomic_features.GenomicFeatures(
            self.target_path, self.features_,
            feature_thresholds=self.feature_thresholds)

        self.sample_from_intervals_ = []
        self.interval_lengths_ = []
        with open(self.intervals_path, 'r') as file_handle:
            for line in file_handle:
                cols = line.strip().split('\t')
                chrom = cols[0]
                start = int(cols[1])
                end = int(cols[2])
                self.sample_from_intervals_.append((chrom, start, end))
                self.interval_lengths_.append(end - start)

        self.rng_ = check_random_state(self.random_state)
        self.steps_per_epoch_ = None

    def flow(self, X, y=None, batch_size=32, sample_weight=None,
             shuffle=None):
        if not hasattr(self, 'reference_genome_'):
            self.fit()

        indices, weights = self.get_indices_and_probabilities(X)

        if shuffle is None:
            shuffle = self.shuffle
        return IntervalsToArrayIterator(
            indices[:, np.newaxis], self, y=y, batch_size=batch_size,
            sample_weight=sample_weight,
            sample_probabilities=weights,
            shuffle=shuffle,
            seed=self.seed)

    def get_indices_and_probabilities(self, X):
        """Return re-sampled indices and probabilities

        Parameters
        ----------
        X : 2-D array

        Returns
        -------
        (index_array, weights)
        index_array : 1-D array
            interval indices
        weights : 1-D array
            sample weights
        """
        # suppose the generator is fitted
        indices, weights = selene_sdk.utils.\
            get_indices_and_probabilities(
                self.interval_lengths_, X[:, 0].tolist())

        return np.asarray(indices), np.asarray(weights)

    def apply_transform(self, idx, shuffle=True):
        """
        Parameters
        ----------
        idx : int
            Index number in the intervals
        shuffle : bool, default is True
            Whether to shuffle the sequence. If False, position will
            be the middle point of the interval.
        """
        interval_info = self.sample_from_intervals_[idx]
        interval_length = self.interval_lengths_[idx]
        chrom = interval_info[0]
        if shuffle:
            position = int(interval_info[1] +
                           self.rng_.uniform(0, 1) * interval_length)
        else:
            position = int(interval_info[1] + 0.5 * interval_length)

        bin_start = position - self.start_radius_
        bin_end = position + self.end_radius_
        retrieved_targets = self.target_.get_feature_data(
            chrom, bin_start, bin_end)

        window_start = bin_start - self.surrounding_sequence_radius_
        window_end = bin_end + self.surrounding_sequence_radius_
        strand = self.STRAND_SIDES[self.rng_.randint(0, 1)]

        retrieved_seq = \
            self.reference_genome_.get_encoding_from_coords(
                chrom, window_start, window_end, strand)

        if retrieved_seq.shape[0] == 0:
            print("Full sequence centered at region \"{0}\" position "
                  "{1} could not be retrieved. Sampling again.".format(
                            chrom, position))
            return None
        elif np.sum(retrieved_seq) / float(retrieved_seq.shape[0]) < 0.60:
            print("Over 30% of the bases in the sequence centered "
                  "at region \"{0}\" position {1} are ambiguous ('N'). "
                  "Sampling again.".format(chrom, position))
            return None

        return (retrieved_seq, retrieved_targets)

    def sample(self, X, sample_size=None, by_sample_weight=False):
        """Output the number of sample arrays and their target values,
        excluding bad samples. If reaching the end of sample index,
        go back to index 0, until the number of sample_size is met.
        For prediction, validation purpose.

        Parameters
        ----------
        X : 2-D array like
            Array of interval indices.
        sample_size : int or None
            The number of output samples. If None,
            sample_size = X.shape[0]
        by_sample_weight : bool, default=False
            If True, samples are retrieved based on sample weight.

        Returns
        -------
        (retrieved_sequences, targets)
        """
        if not sample_size:
            sample_size = X.shape[0]

        retrieved_sequences = np.zeros(
            (sample_size, self.seq_length, 4), dtype='float32')
        targets = np.zeros(
            (sample_size, self.n_features_), dtype='int32')

        if by_sample_weight:
            indices, weights = self.get_indices_and_probabilities(X)
            sample_index = self.rng_.choice(indices, size=sample_size,
                                            replace=True, p=weights)
        else:
            sample_index = X[:, 0]

        n_samples_drawn = 0
        i = 0
        while n_samples_drawn < sample_size:
            interval_idx = int(sample_index[i])
            retrieve_output = self.apply_transform(interval_idx)
            if retrieve_output is not None:
                retrieved_sequences[n_samples_drawn, :, :] = \
                    retrieve_output[0]
                targets[n_samples_drawn, :] = retrieve_output[1]
                n_samples_drawn += 1
            elif by_sample_weight:
                while retrieve_output is None:
                    rand_idx = int(self.rng_.choice(
                        indices, size=1, p=weights)[0])
                    retrieve_output = self.apply_transform(rand_idx)
                    if retrieve_output is not None:
                        retrieved_sequences[n_samples_drawn, :, :]\
                            = retrieve_output[0]
                        targets[n_samples_drawn, :] = retrieve_output[1]
                n_samples_drawn += 1

            i += 1
            if i >= sample_index.shape[0]:
                i = 0

        return retrieved_sequences, targets
