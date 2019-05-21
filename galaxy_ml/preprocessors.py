"""
Z_RandomOverSampler
"""
import numpy as np
from collections import Counter
from keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.utils import check_target_type
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing.data import _handle_zeros_in_scale
from sklearn.utils import check_array, safe_indexing
from sklearn.utils.validation import (check_is_fitted, check_X_y,
                                      FLOAT_DTYPES)
try:
    import pyfaidx
except ImportError:
    rval = __import__('os').system("pip install pyfaidx==0.5.5.2")
    if rval != 0:
        raise ImportError("module pyfaidx is not installed. "
                          "Galaxy attemped to install but failed."
                          "Please Contact Admin for manual "
                          "installation.")
try:
    # tools should pick up here
    from externals import selene_sdk
except ImportError:
    # nosetest picks here
    try:
        from .externals import selene_sdk
    except ImportError:
        pass


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
        File path to the fasta file. There could two alternative ways to set up
        `fasta_path`. 1) through fit_params; 2) set_params().
    padding : bool, default is False
        All sequences are expected to be in the same length, but sometimes not.
        If True, all sequences use the same length of first entry by either
        padding or truncating. If False, raise ValueError if different seuqnce
        lengthes are found.
    """
    BASE_TO_INDEX = {
        'A': 0, 'C': 1, 'G': 2, 'T': 3,
        'a': 0, 'c': 1, 'g': 2, 't': 3,
    }

    UNK_BASE = 'N'

    def __init__(self, fasta_path=None, padding=False):
        super(GenomeOneHotEncoder, self).__init__()
        self.fasta_path = fasta_path
        self.padding = padding

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

        if not self.fasta_path:
            raise ValueError("`fasta_path` can't be None!")

        fasta_file = pyfaidx.Fasta(self.fasta_path)
        # set up the sequence_length from the first entry
        sequence_length = len(fasta_file[int(X[0, 0])])
        if not self.padding:
            for idx in X[:, 0]:
                fasta_record = fasta_file[int(idx)]
                if len(fasta_record) != sequence_length:
                    raise ValueError("The first sequence record contains "
                                     "%d bases, while %s contrain %d bases"
                                     % (sequence_length,
                                        repr(fasta_record),
                                        len(fasta_record)))

        self.fasta_file = fasta_file
        self.sequence_length = sequence_length
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
                                        self.sequence_length,
                                        4))
        for i in range(X.shape[0]):
            cur_sequence = self.fasta_file[int(X[i, 0])]
            if len(cur_sequence) > self.sequence_length:
                cur_sequence = selene_sdk.predict._common._truncate_sequence(
                    cur_sequence,
                    self.sequence_length)

            elif len(cur_sequence) < self.sequence_length:
                cur_sequence = selene_sdk.predict._common._pad_sequence(
                    cur_sequence,
                    self.sequence_length,
                    GenomeOneHotEncoder.UNK_BASE)

            cur_sequence_encodeing = selene_sdk.sequences._sequence.\
                _fast_sequence_to_encoding(
                    str(cur_sequence),
                    GenomeOneHotEncoder.BASE_TO_INDEX,
                    4)

            sequences_endcoding[i, :, :] = cur_sequence_encodeing

        return sequences_endcoding


class ImageBatchGenerator(ImageDataGenerator, BaseEstimator):
    pass
