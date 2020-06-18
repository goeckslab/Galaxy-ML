import numpy as np
import warnings

from itertools import chain
from sklearn.model_selection import (GroupShuffleSplit, ShuffleSplit,
                                     StratifiedShuffleSplit)
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils import indexable, safe_indexing
from sklearn.utils.validation import _num_samples, check_array


def train_test_split(*arrays, **options):
    """Extend sklearn.model_selection.train_test_slit to have group split.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    test_size : float, int or None, optional (default=None)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.
    train_size : float, int, or None, (default=None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    shuffle : None or str (default='simple')
        How to shuffle the data before splitting.
        None, no shuffle.
        For str, one of 'simple', 'stratified' and 'group', corresponding to
        `ShuffleSplit`, `StratifiedShuffleSplit` and `GroupShuffleSplit`,
        respectively.
    labels : array-like or None (default=None)
        Ignored if shuffle is None or 'simple'.
        When shuffle='stratified', this array is used as class labels.
        When shuffle='group', this array is used as groups.

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.

    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    test_size = options.pop('test_size', None)
    train_size = options.pop('train_size', None)
    random_state = options.pop('random_state', None)
    shuffle = options.pop('shuffle', 'simple')
    labels = options.pop('labels', None)

    if options:
        raise TypeError("Invalid parameters passed: %s" % str(options))

    arrays = indexable(*arrays)

    n_samples = _num_samples(arrays[0])
    if shuffle == 'group':
        if labels is None:
            raise ValueError("When shuffle='group', "
                             "labels should not be None!")
        labels = check_array(labels, ensure_2d=False, dtype=None)
        uniques = np.unique(labels)
        n_samples = uniques.size

    n_train, n_test = _validate_shuffle_split(n_samples, test_size, train_size,
                                              default_test_size=0.25)

    shuffle_options = dict(test_size=n_test,
                           train_size=n_train,
                           random_state=random_state)

    if shuffle is None or shuffle == 'None':
        if labels is not None:
            warnings.warn("The `labels` is ignored for "
                          "shuffle being None!")

        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)

    elif shuffle == 'simple':
        if labels is not None:
            warnings.warn("The `labels` is not needed and therefore "
                          "ignored for ShuffleSplit, as shuffle='simple'!")

        cv = ShuffleSplit(**shuffle_options)
        train, test = next(cv.split(X=arrays[0], y=None))

    elif shuffle == 'stratified':
        cv = StratifiedShuffleSplit(**shuffle_options)
        train, test = next(cv.split(X=arrays[0], y=labels))

    elif shuffle == 'group':
        cv = GroupShuffleSplit(**shuffle_options)
        train, test = next(cv.split(X=arrays[0], y=None, groups=labels))

    else:
        raise ValueError("The argument `shuffle` only supports None, "
                         "'simple', 'stratified' and 'group', but got `%s`!"
                         % shuffle)

    return list(chain.from_iterable((safe_indexing(a, train),
                                    safe_indexing(a, test)) for a in arrays))
