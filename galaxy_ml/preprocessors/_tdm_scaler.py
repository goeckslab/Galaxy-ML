import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted


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
        check_is_fitted(self, 'iqr_')
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
