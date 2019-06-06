
<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/utils.py#L41)</span>
## _SafePickler

```python
galaxy_ml.utils._SafePickler(file)
```


Used to safely deserialize scikit-learn model objects
Usage:
eg.: _SafePickler.load(pickled_file_object)

----

<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/utils.py#L311)</span>
## SafeEval

```python
galaxy_ml.utils.SafeEval(load_scipy=False, load_numpy=False, load_estimators=False)
```

Customized symbol table for safely literal eval

**Parameters**

- **load_scipy**: bool, default=False<br>
        Whether to load globals from scipy
- **load_numpy**: bool, default=False<br>
        Whether to load globals from numpy
- **load_estimators**: bool, default=False<br>
        Whether to load globals for sklearn estimators
    