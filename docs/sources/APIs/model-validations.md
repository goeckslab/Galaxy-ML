
<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/model_validations.py#L203)</span>
## OrderedKFold

```python
galaxy_ml.model_validations.OrderedKFold(n_splits=3, shuffle=False, random_state=None)
```


Split into K fold based on ordered target value

**Parameters**

- **n_splits**: int, default=3<br>
        Number of folds. Must be at least 2.
- **shuffle**: bool<br>
- **random_state**: None or int<br>
    
----

<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/model_validations.py#L236)</span>
## RepeatedOrderedKFold

```python
galaxy_ml.model_validations.RepeatedOrderedKFold(n_splits=5, n_repeats=5, random_state=None)
```

Repeated OrderedKFold runs mutiple times with different randomization.

**Parameters**

- **n_splits**: int, default=5<br>
        Number of folds. Must be at least 2.
- **n_repeats**: int, default=5<br>
        Number of times cross-validator to be repeated.
    random_state: int, RandomState instance or None. Optional
    