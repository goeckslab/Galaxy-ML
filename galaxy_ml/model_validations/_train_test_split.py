
from itertools import chain

import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils import _safe_indexing, indexable
from sklearn.utils.validation import check_array


def train_test_split(*arrays,
                     test_size=None,
                     train_size=None,
                     random_state=None,
                     shuffle=True,
                     labels=None):
    """Extend sklearn.model_selection.train_test_slit to have group split.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.
    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
    shuffle : str, default='simple'
        One of [None, 'simple', 'stratified', 'group']. Whether or not to
        shuffle the data before splitting. None: no shuffle; 'simple': non
        stratified shuffle; 'stratified': shuffle with class labels;
        'group': shuffle with group labels.
    labels : array-like or None, default=None
        If shuffle='simple' or shuffle=None, this must be None. If shuffle=
        'stratified', this array is used as class labels. If shuffle='group',
        this array is used as group labels

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.

    """
    if shuffle and shuffle not in \
            ['simple', 'stratified', 'group', True, False]:
        raise ValueError("The argument `shuffle` only supports None, "
                         "'simple', 'stratified' and 'group', but got `%s`!"
                         % shuffle)

    if shuffle != 'group':
        shuffle = False if not shuffle else True
        return sk_train_test_split(*arrays,
                                   test_size=test_size,
                                   train_size=train_size,
                                   random_state=random_state,
                                   shuffle=shuffle,
                                   stratify=labels)

    if labels is None:
        raise ValueError("When shuffle='group', labels should not be None!")

    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    arrays = indexable(*arrays)

    labels = check_array(labels, ensure_2d=False, dtype=None)
    n_samples = np.unique(labels).size

    n_train, n_test = _validate_shuffle_split(n_samples, test_size, train_size,
                                              default_test_size=0.25)

    cv = GroupShuffleSplit(n_splits=1,
                           test_size=n_test,
                           train_size=n_train,
                           random_state=random_state)

    train, test = next(cv.split(X=arrays[0], y=None, groups=labels))

    return list(chain.from_iterable((_safe_indexing(a, train),
                                     _safe_indexing(a, test)) for a in arrays))
