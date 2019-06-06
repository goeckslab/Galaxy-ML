
<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/iraps_classifier.py#L2)</span>
## IRAPSCore

```python
galaxy_ml.iraps_classifier.IRAPSCore(n_iter=1000, positive_thres=-1, negative_thres=0, verbose=0, n_jobs=1, pre_dispatch='2*n_jobs', random_state=None)
```


Base class of IRAPSClassifier
From sklearn BaseEstimator:
get_params()
set_params()

**Parameters**

- **n_iter**: int<br>
        sample count
- **positive_thres**: float<br>
        z_score shreshold to discretize positive target values
- **negative_thres**: float<br>
        z_score threshold to discretize negative target values
- **verbose**: int<br>
        0 or geater, if not 0, print progress
- **n_jobs**: int, default=1<br>
        The number of CPUs to use to do the computation.
- **pre_dispatch**: int, or string.<br>
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
- **random_state**: int or None<br>
    
----

<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/iraps_classifier.py#L3)</span>
## IRAPSClassifier

```python
galaxy_ml.iraps_classifier.IRAPSClassifier(iraps_core, p_thres=0.0001, fc_thres=0.1, occurrence=0.8, discretize=-1, memory=None, min_signature_features=1)
```


Extend the bases of both sklearn feature_selector and classifier.
From sklearn BaseEstimator:
get_params()
set_params()
From sklearn _BaseFilter:
get_support()
fit_transform(X)
transform(X)
From sklearn RegressorMixin:
score(X, y): R2
New:
predict(X)
predict_label(X)
get_signature()
Properties:
discretize_value

**Parameters**

- **iraps_core**: object<br>
- **p_thres**: float, threshold for p_values<br>
- **fc_thres**: float, threshold for fold change or mean difference<br>
- **occurrence**: float, occurrence rate selected by set of p_thres and fc_thres<br>
- **discretize**: float, threshold of z_score to discretize target value<br>
- **memory**: None, str or joblib.Memory object<br>
- **min_signature_features**: int, the mininum number of features in a signature<br>
    