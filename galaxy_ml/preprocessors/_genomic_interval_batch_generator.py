import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from ._fasta_iterator import FastaToArrayIterator
from ..externals import selene_sdk


class IntervalsToArrayIterator(FastaToArrayIterator):
    """Iterator yielding Numpy array from intervals and reference
    sequences.

    Parameters
    ----------
    X : array
        Contains sequence indexes in the fasta file
    generator : fitted object
        instance of GenomicIntervalBatchGenerator.
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
        batch_x = np.zeros(
            (n_samples, generator.seq_length, generator.n_bases),
            dtype='float32',
        )
        batch_y = np.zeros(
            (n_samples, generator.n_features_in_),
            dtype='int32',
        )

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

    def set_processing_attrs(self):
        if self.features == 'infer':
            features_set = set()
            with open(self.target_path, 'r') as features_file:
                for line in features_file:
                    feature = line.split('\t')[-1].strip()
                    features_set.add(feature)
            self.features_ = list(features_set)
        else:
            self.features_ = self.features

        self.n_features_in_ = len(self.features_)
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
            self.set_processing_attrs()

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
            position = int(
                interval_info[1]
                + self.rng_.uniform(0, 1) * interval_length
            )
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
            print(
                "Full sequence centered at region \"{0}\" position "
                "{1} could not be retrieved. Sampling again."
                .format(chrom, position)
            )
            return None
        elif np.sum(retrieved_seq) / float(retrieved_seq.shape[0]) < 0.60:
            print("Over 30% of the bases in the sequence centered "
                  "at region \"{0}\" position {1} are ambiguous ('N'). "
                  "Sampling again.".format(chrom, position))
            return None

        return (retrieved_seq, retrieved_targets)

    def sample(self, X, y=None, sample_size=None, by_sample_weight=False):
        """Output the number of sample arrays and their target values,
        excluding bad samples. If reaching the end of sample index,
        go back to index 0, until the number of sample_size is met.
        For prediction, validation purpose.

        Parameters
        ----------
        X : 2-D array like
            Array of interval indices.
        y : None
            For compatibility.
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
            (sample_size, self.n_features_in_), dtype='int32')

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

                retrieved_sequences[n_samples_drawn, :, :] = retrieve_output[0]
                targets[n_samples_drawn, :] = retrieve_output[1]
                n_samples_drawn += 1

            i += 1
            if i >= sample_index.shape[0]:
                i = 0

        return retrieved_sequences, targets

    def close(self):
        if hasattr(self, 'reference_genome_'):
            self.reference_genome_.genome.close()
