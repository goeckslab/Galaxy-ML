
<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/iraps_classifier.py#L4)</span>
## BinarizeTargetClassifier

```python
galaxy_ml.iraps_classifier.BinarizeTargetClassifier(classifier, z_score=-1, value=None, less_is_positive=True)
```


Convert continuous target to binary labels (True and False)
and apply a classification estimator.

**Parameters**

- **classifier**: object<br>
        Estimator object such as derived from sklearn `ClassifierMixin`.
- **z_score**: float, default=-1.0<br>
        Threshold value based on z_score. Will be ignored when
        fixed_value is set
- **value**: float, default=None<br>
        Threshold value
- **less_is_positive**: boolean, default=True<br>
        When target is less the threshold value, it will be converted
        to True, False otherwise.

**Attributes**

- **classifier_**: object<br>
        Fitted classifier
- **discretize_value**: float<br>
        The threshold value used to discretize True and False targets
    
----

<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/iraps_classifier.py#L5)</span>
## BinarizeTargetRegressor

```python
galaxy_ml.iraps_classifier.BinarizeTargetRegressor(regressor, z_score=-1, value=None, less_is_positive=True)
```


Extend regression estimator to have discretize_value

**Parameters**

- **regressor**: object<br>
        Estimator object such as derived from sklearn `RegressionMixin`.
- **z_score**: float, default=-1.0<br>
        Threshold value based on z_score. Will be ignored when
        fixed_value is set
- **value**: float, default=None<br>
        Threshold value
- **less_is_positive**: boolean, default=True<br>
        When target is less the threshold value, it will be converted
        to True, False otherwise.

**Attributes**

- **regressor_**: object<br>
        Fitted regressor
- **discretize_value**: float<br>
        The threshold value used to discretize True and False targets
    
----

<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/iraps_classifier.py#L567)</span>
## BinarizeTargetTransformer

```python
galaxy_ml.iraps_classifier.BinarizeTargetTransformer(transformer, z_score=-1, value=None, less_is_positive=True)
```


Extend transformaer to work for binarized target.

**Parameters**

- **transformer**: object<br>
        Estimator object such as derived from sklearn `TransformerMixin`,
        including feature_selector and preprocessor
- **z_score**: float, default=-1.0<br>
        Threshold value based on z_score. Will be ignored when
        fixed_value is set
- **value**: float, default=None<br>
        Threshold value
- **less_is_positive**: boolean, default=True<br>
        When target is less the threshold value, it will be converted
        to True, False otherwise.

**Attributes**

- **transformer_**: object<br>
        Fitted regressor
- **discretize_value**: float<br>
        The threshold value used to discretize True and False targets
    