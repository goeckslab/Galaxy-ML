
<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/feature_selectors.py#L30)</span>
## DyRFE

```python
galaxy_ml.feature_selectors.DyRFE(estimator, n_features_to_select=None, step=1, verbose=0)
```


Mainly used with DyRFECV

**Parameters**

- **estimator**: object<br>
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.
- **n_features_to_select**: int or None (default=None)<br>
        The number of features to select. If `None`, half of the features
        are selected.
- **step**: int, float or list, optional (default=1)<br>
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of features to remove at each iteration.
        If list, a series of steps of features to remove at each iteration.
        Iterations stops when steps finish
- **verbose**: int, (default=0)<br>
        Controls verbosity of output.


----

<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/feature_selectors.py#L154)</span>
## DyRFECV

```python
galaxy_ml.feature_selectors.DyRFECV(estimator, step=1, min_features_to_select=1, cv='warn', scoring=None, verbose=0, n_jobs=None)
```


Compared with RFECV, DyRFECV offers flexiable `step` to eleminate
features, in the format of list, while RFECV supports only fixed number
of `step`.

**Parameters**

- **estimator**: object<br>
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.
- **step**: int or float, optional (default=1)<br>
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of features to remove at each iteration.
        If list, a series of step to remove at each iteration. iteration stopes
        when finishing all steps
        Note that the last iteration may remove fewer than ``step`` features in
        order to reach ``min_features_to_select``.
- **min_features_to_select**: int, (default=1)<br>
        The minimum number of features to be selected. This number of features
        will always be scored, even if the difference between the original
        feature count and ``min_features_to_select`` isn't divisible by
        ``step``.
- **cv**: int, cross-validation generator or an iterable, optional<br>
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. If the
        estimator is a classifier or if ``y`` is neither binary nor multiclass,
        :class:`sklearn.model_selection.KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        .. versionchanged:: 0.20
            ``cv`` default value of None will change from 3-fold to 5-fold
            in v0.22.

- **scoring**: string, callable or None, optional, (default=None)<br>
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
- **verbose**: int, (default=0)<br>
        Controls verbosity of output.
- **n_jobs**: int or None, optional (default=None)<br>
        Number of cores to run in parallel while fitting across folds.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    