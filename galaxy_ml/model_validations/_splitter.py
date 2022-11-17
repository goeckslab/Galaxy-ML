import numbers

import numpy as np

from sklearn.model_selection._split import _BaseKFold, _RepeatedSplits
from sklearn.utils import check_random_state
from sklearn.utils.validation import _num_samples


class OrderedKFold(_BaseKFold):
    """
    Split into K fold based on ordered target value. Provide ranking
    stratification for regressions.

    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.
    shuffle : bool
    random_state : None or int
        Only relevant when shuffle is True.
    n_stratification_bins : None or positive integer
        Only relevant when shuffle is True. The number of bins into which
        samples are categorized for stratification. Valid in
        [2, `n_samples // n_splits`]. Default value is None, which is same
        as `n_samples // n_splits`. The higher the value is, the distribution
        of target values is more approximately the ame across all split folds.
        `ValueError` will be raised, if invalid value is given.
    """

    def __init__(
        self,
        n_splits=3,
        shuffle=False,
        random_state=None,
        n_stratification_bins=None,
    ):
        super(OrderedKFold, self).__init__(
            n_splits, shuffle=shuffle, random_state=random_state)
        if n_stratification_bins is not None:
            if not shuffle:
                raise ValueError("The n_stratification_bins is only relevant "
                                 "when shuffle is True. Set to None if shuffle"
                                 " is not needed!")
            if not isinstance(n_stratification_bins, numbers.Integral) or \
                    n_stratification_bins < 2:
                raise ValueError("The number of stratification bins must be "
                                 "None or an interger not less than 2. "
                                 "%s of type %s was passed."
                                 % (n_stratification_bins,
                                    type(n_stratification_bins)))

            n_stratification_bins = int(n_stratification_bins)

        self.n_stratification_bins = n_stratification_bins

    def _iter_test_indices(self, X, y, groups=None):
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        y = np.asarray(y)
        sorted_index = np.argsort(y)
        if self.shuffle:
            n_stratification_bins = self.n_stratification_bins
            rng = check_random_state(self.random_state)
            maxi_bins = n_samples // n_splits
            if n_stratification_bins in (None, maxi_bins):
                strides = [n_splits] * maxi_bins
                reminder = n_samples % n_splits
                if reminder:
                    strides.append(reminder)
            elif n_stratification_bins < maxi_bins:
                bin_size = n_samples // n_stratification_bins
                reminder = n_samples % n_stratification_bins
                strides = [bin_size] * n_stratification_bins
                for i in range(reminder):
                    strides[i] += 1
            else:
                ValueError("The highest allowed number of stratification "
                           "bins is %s, but %s was passed."
                           % (maxi_bins, n_stratification_bins))

            current = 0
            for step in strides:
                start, stop = current, current + step
                rng.shuffle(sorted_index[start:stop])
                current = stop

        for i in range(n_splits):
            yield sorted_index[i:n_samples:n_splits]


class RepeatedOrderedKFold(_RepeatedSplits):
    """ Repeated OrderedKFold runs mutiple times with different randomization.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    n_repeats : int, default=5
        Number of times cross-validator to be repeated.
    random_state: int, RandomState instance or None. Optional
    n_stratification_bins : None or positive integer
        Only relevant when shuffle is True. The number of bins into which
        samples are categorized for stratification. Valid in
        [2, `n_samples // n_splits`]. Default value is None, which is same as
        `n_samples // n_splits`. The higher the value is, the distribution of
        target values is more approximately the ame across all split folds.
        `ValueError` will be raised, if invalid value is given.
    """
    def __init__(self, n_splits=5, n_repeats=5, random_state=None,
                 n_stratification_bins=None):
        super(RepeatedOrderedKFold, self).__init__(
            OrderedKFold,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
            n_stratification_bins=n_stratification_bins,
        )
