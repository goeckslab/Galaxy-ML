import numpy as np

from sklearn.base import (
    BaseEstimator, RegressorMixin, TransformerMixin, clone)
from sklearn.utils.validation import (
    check_array, check_is_fitted, column_or_1d)


class BinarizeTargetClassifier(BaseEstimator, RegressorMixin):
    """
    Convert continuous target to binary labels (True and False)
    and apply a classification estimator.

    Parameters
    ----------
    classifier : object
        Estimator object such as derived from sklearn `ClassifierMixin`.
    z_score : float, default=-1.0
        Threshold value based on z_score. Will be ignored when
        fixed_value is set
    value : float, default=None
        Threshold value
    less_is_positive : boolean, default=True
        When target is less the threshold value, it will be converted
        to True, False otherwise.
    verbose : int, default=0
        If greater than 0, print discretizing info.

    Attributes
    ----------
    classifier_ : object
        Fitted classifier
    discretize_value : float
        The threshold value used to discretize True and False targets
    """

    def __init__(self, classifier, z_score=-1, value=None,
                 less_is_positive=True, verbose=0):
        self.classifier = classifier
        self.z_score = z_score
        self.value = value
        self.less_is_positive = less_is_positive
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None, **fit_params):
        """
        Convert y to True and False labels and then fit the classifier
        with X and new y

        Returns
        ------
        self: object
        """
        y = check_array(y, accept_sparse=False, force_all_finite=True,
                        ensure_2d=False, dtype='numeric')
        y = column_or_1d(y)

        if self.value is None:
            discretize_value = y.mean() + y.std() * self.z_score
        else:
            discretize_value = self.Value
        self.discretize_value = discretize_value

        if self.less_is_positive:
            y_trans = y < discretize_value
        else:
            y_trans = y > discretize_value

        n_positives = np.sum(y_trans)
        # for older version compatibility
        if self.verbose and self.verbose > 0:
            print("{0} out of total {1} samples are discretized into "
                  "positive.".format(n_positives, X.shape[0]))

        self.classifier_ = clone(self.classifier)

        keys = list(fit_params.keys())
        for key in keys:
            if not key.startswith('classifier__'):
                raise ValueError("fit_params for BinarizeTargetClassifier "
                                 "must start with `classifier__`")
            fit_params[key[12:]] = fit_params.pop(key)

        if sample_weight is not None:
            self.classifier_.fit(X, y_trans,
                                 sample_weight=sample_weight,
                                 **fit_params)
        else:
            self.classifier_.fit(X, y_trans, **fit_params)

        # Used in RFE or SelectFromModel
        if hasattr(self.classifier_, 'feature_importances_'):
            self.feature_importances_ = self.classifier_.feature_importances_
        if hasattr(self.classifier_, 'coef_'):
            self.coef_ = self.classifier_.coef_
        if hasattr(self.classifier_, 'n_outputs_'):
            self.n_outputs_ = self.classifier_.n_outputs_
        if hasattr(self.classifier_, 'n_features_in_'):
            self.n_features_in_ = self.classifier_.n_features_in_
        if hasattr(self.classifier_, 'classes_'):
            self.classes_ = self.classifier_.classes_

        return self

    def predict(self, X):
        """Predict using a fitted estimator
        """
        return self.classifier_.predict(X)

    def decision_function(self, X):
        """Predict using a fitted estimator
        """
        try:
            return self.classifier_.decision_function(X)
        except Exception:
            raise

    def predict_proba(self, X):
        """Predict using a fitted estimator
        """
        return self.classifier_.predict_proba(X)


class BinarizeTargetRegressor(BaseEstimator, RegressorMixin):
    """
    Extend regression estimator to have discretize_value

    Parameters
    ----------
    regressor : object
        Estimator object such as derived from sklearn `RegressionMixin`.
    z_score : float, default=-1.0
        Threshold value based on z_score. Will be ignored when
        value is set
    value : float, default=None
        Threshold value
    less_is_positive : boolean, default=True
        When target is less the threshold value, it will be converted
        to True, False otherwise.
    verbose : int, default=0
        If greater than 0, print discretizing info.

    Attributes
    ----------
    regressor_ : object
        Fitted regressor
    discretize_value : float
        The threshold value used to discretize True and False targets
    """

    def __init__(self, regressor, z_score=-1, value=None,
                 less_is_positive=True, verbose=0):
        self.regressor = regressor
        self.z_score = z_score
        self.value = value
        self.less_is_positive = less_is_positive
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None, **fit_params):
        """
        Calculate the discretize_value fit the regressor with traning data

        Returns
        ------
        self: object
        """
        y = check_array(y, accept_sparse=False, force_all_finite=True,
                        ensure_2d=False, dtype='numeric')
        y = column_or_1d(y)

        if not np.all((y >= 0) & (y <= 1)):
            raise ValueError("The target value of BinarizeTargetRegressor "
                             "must be in the range [0, 1]")

        if self.value is None:
            discretize_value = y.mean() + y.std() * self.z_score
        else:
            discretize_value = self.Value
        self.discretize_value = discretize_value
        if self.less_is_positive:
            n_positives = np.sum(y < discretize_value)
        else:
            n_positives = np.sum(y > discretize_value)

        # for older version compatibility
        if self.verbose and self.verbose > 0:
            print("{0} out of total {1} samples are discretized into "
                  "positive.".format(n_positives, X.shape[0]))

        self.regressor_ = clone(self.regressor)

        keys = list(fit_params.keys())
        for key in keys:
            if not key.startswith('regressor__'):
                raise ValueError("fit_params for BinarizeTargetClassifier "
                                 "must start with `regressor__`")
            fit_params[key[11:]] = fit_params.pop(key)

        if sample_weight is not None:
            self.regressor_.fit(X, y,
                                sample_weight=sample_weight,
                                **fit_params)
        else:
            self.regressor_.fit(X, y, **fit_params)

        # attach classifier attributes
        if hasattr(self.regressor_, 'feature_importances_'):
            self.feature_importances_ = self.regressor_.feature_importances_
        if hasattr(self.regressor_, 'coef_'):
            self.coef_ = self.regressor_.coef_
        if hasattr(self.regressor_, 'n_outputs_'):
            self.n_outputs_ = self.regressor_.n_outputs_
        if hasattr(self.regressor_, 'n_features_in_'):
            self.n_features_in_ = self.regressor_.n_features_in_

        return self

    def predict(self, X):
        """Predict target value of X
        """
        check_is_fitted(self, 'regressor_')
        return self.regressor_.predict(X)

    def decision_function(self, X):
        """
        Output the proba for True label
        For use in the binarize target scorers.
        """
        pred = self.predict(X)
        if self.less_is_positive:
            pred = 1 - pred

        return pred

    def predict_label(self, X, cutoff):
        """ output a label based on cutoff value

        Parameters
        ----------
        cutoff : float
        """
        scores = self.decision_function(X)
        return scores > cutoff


class BinarizeTargetTransformer(BaseEstimator, TransformerMixin):
    """
    Extend transformaer to work for binarized target.

    Parameters
    ----------
    transformer : object
        Estimator object such as derived from sklearn `TransformerMixin`,
        including feature_selector and preprocessor
    z_score : float, default=-1.0
        Threshold value based on z_score. Will be ignored when
        fixed_value is set
    value : float, default=None
        Threshold value
    less_is_positive : boolean, default=True
        When target is less the threshold value, it will be converted
        to True, False otherwise.

    Attributes
    ----------
    transformer_ : object
        Fitted regressor
    discretize_value : float
        The threshold value used to discretize True and False targets
    """
    def __init__(self, transformer, z_score=-1, value=None,
                 less_is_positive=True):
        self.transformer = transformer
        self.z_score = z_score
        self.value = value
        self.less_is_positive = less_is_positive

    def fit(self, X, y):
        """
        Convert y to True and False labels and then fit the transformer
        with X and new y

        Returns
        ------
        self: object
        """
        y = check_array(y, accept_sparse=False, force_all_finite=True,
                        ensure_2d=False, dtype='numeric')
        y = column_or_1d(y)

        if self.value is None:
            discretize_value = y.mean() + y.std() * self.z_score
        else:
            discretize_value = self.Value
        self.discretize_value = discretize_value

        if self.less_is_positive:
            y_trans = y < discretize_value
        else:
            y_trans = y > discretize_value

        self.transformer_ = clone(self.transformer)

        self.transformer_.fit(X, y_trans)

        return self

    def transform(self, X):
        """Transform X

        Parameters
        ----------
        X : array of shape [n_samples, n_features]

        Returns
        -------
        X_r : array
        """
        check_is_fitted(self, 'transformer_')
        X = check_array(X, dtype=None, accept_sparse='csr')

        return self.transformer_.transform(X)
