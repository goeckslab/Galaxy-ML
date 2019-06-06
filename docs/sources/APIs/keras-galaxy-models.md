
<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/keras_galaxy_models.py#L187)</span>
## SearchParam

```python
galaxy_ml.keras_galaxy_models.SearchParam(s_param, value)
```


Sortable Wrapper class for search parameters

----

<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/keras_galaxy_models.py#L210)</span>
## KerasLayers

```python
galaxy_ml.keras_galaxy_models.KerasLayers(name='sequential_1', layers=[])
```


**Parameters**

    name: str
    layers: list of dict, the configuration of model
    
----

<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/keras_galaxy_models.py#L290)</span>
## BaseKerasModel

```python
galaxy_ml.keras_galaxy_models.BaseKerasModel(config, model_type='sequential', optimizer='sgd', loss='binary_crossentropy', metrics=[], lr=None, momentum=None, decay=None, nesterov=None, rho=None, epsilon=None, amsgrad=None, beta_1=None, beta_2=None, schedule_decay=None, epochs=1, batch_size=None, seed=0, callbacks=None, validation_data=None)
```


Base class for Galaxy Keras wrapper

**Parameters**

- **config**: dictionary<br>
        from `model.get_config()`
- **model_type**: str<br>
        'sequential' or 'functional'
- **optimizer**: str, default 'sgd'<br>
        'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'
- **loss**: str, default 'binary_crossentropy'<br>
        same as Keras `loss`
- **metrics**: list of strings, default []<br>
- **lr**: None or float<br>
        optimizer parameter, default change with `optimizer`
- **momentum**: None or float<br>
        for optimizer `sgd` only, ignored otherwise
- **nesterov**: None or bool<br>
        for optimizer `sgd` only, ignored otherwise
- **decay**: None or float<br>
        optimizer parameter, default change with `optimizer`
- **rho**: None or float<br>
        optimizer parameter, default change with `optimizer`
- **epsilon**: None or float<br>
        optimizer parameter, default change with `optimizer`
- **amsgrad**: None or bool<br>
        for optimizer `adam` only, ignored otherwise
- **beta_1**: None or float<br>
        optimizer parameter, default change with `optimizer`
- **beta_2**: None or float<br>
        optimizer parameter, default change with `optimizer`
- **schedule_decay**: None or float<br>
        optimizer parameter, default change with `optimizer`
- **epochs**: int<br>
        fit_param from Keras
- **batch_size**: None or int, default=None<br>
        fit_param, if None, will default to 32
- **callbacks**: None or list of dict<br>
        fit_param, each dict contains one type of callback configuration.
        e.g. {"callback_selection":
                {"callback_type": "EarlyStopping",
                 "monitor": "val_loss"
                 "baseline": None,
                 "min_delta": 0.0,
                 "patience": 10,
                 "mode": "auto",
                 "restore_best_weights": False}}
- **validation_data**: None or tuple of arrays, (X_test, y_test)<br>
        fit_param
- **seed**: None or int, default 0<br>
        backend random seed
    
----

<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/keras_galaxy_models.py#L755)</span>
## KerasGClassifier

```python
galaxy_ml.keras_galaxy_models.KerasGClassifier(config, model_type='sequential', optimizer='sgd', loss='binary_crossentropy', metrics=[], lr=None, momentum=None, decay=None, nesterov=None, rho=None, epsilon=None, amsgrad=None, beta_1=None, beta_2=None, schedule_decay=None, epochs=1, batch_size=None, seed=0, callbacks=None, validation_data=None)
```


Scikit-learn classifier API for Keras

----

<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/keras_galaxy_models.py#L829)</span>
## KerasGRegressor

```python
galaxy_ml.keras_galaxy_models.KerasGRegressor(config, model_type='sequential', optimizer='sgd', loss='binary_crossentropy', metrics=[], lr=None, momentum=None, decay=None, nesterov=None, rho=None, epsilon=None, amsgrad=None, beta_1=None, beta_2=None, schedule_decay=None, epochs=1, batch_size=None, seed=0, callbacks=None, validation_data=None)
```


Scikit-learn API wrapper for Keras regressor

----

<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/keras_galaxy_models.py#L857)</span>
## KerasGBatchClassifier

```python
galaxy_ml.keras_galaxy_models.KerasGBatchClassifier(config, train_batch_generator, predict_batch_generator=None, model_type='sequential', optimizer='sgd', loss='binary_crossentropy', metrics=[], lr=None, momentum=None, decay=None, nesterov=None, rho=None, epsilon=None, amsgrad=None, beta_1=None, beta_2=None, schedule_decay=None, epochs=1, batch_size=None, seed=0, n_jobs=1, callbacks=None, validation_data=None)
```


keras classifier with batch data generator

**Parameters**

- **config**: dictionary<br>
        from `model.get_config()`
    train_batch_generator: instance of batch data generator
    predict_batch_generator: instance of batch data generator (default=None)
        if None, same as train_batch_generator
- **model_type**: str<br>
        'sequential' or 'functional'
- **optimizer**: str, default 'sgd'<br>
        'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'
- **loss**: str, default 'binary_crossentropy'<br>
        same as Keras `loss`
- **metrics**: list of strings, default []<br>
- **lr**: None or float<br>
        optimizer parameter, default change with `optimizer`
- **momentum**: None or float<br>
        for optimizer `sgd` only, ignored otherwise
- **nesterov**: None or bool<br>
        for optimizer `sgd` only, ignored otherwise
- **decay**: None or float<br>
        optimizer parameter, default change with `optimizer`
- **rho**: None or float<br>
        optimizer parameter, default change with `optimizer`
- **epsilon**: None or float<br>
        optimizer parameter, default change with `optimizer`
- **amsgrad**: None or bool<br>
        for optimizer `adam` only, ignored otherwise
- **beta_1**: None or float<br>
        optimizer parameter, default change with `optimizer`
- **beta_2**: None or float<br>
        optimizer parameter, default change with `optimizer`
- **schedule_decay**: None or float<br>
        optimizer parameter, default change with `optimizer`
- **epochs**: int<br>
        fit_param from Keras
- **batch_size**: None or int, default=None<br>
        fit_param, if None, will default to 32
- **callbacks**: None or list of dict<br>
        each dict contains one type of callback configuration.
        e.g. {"callback_selection":
                {"callback_type": "EarlyStopping",
                 "monitor": "val_loss"
                 "baseline": None,
                 "min_delta": 0.0,
                 "patience": 10,
                 "mode": "auto",
                 "restore_best_weights": False}}
- **validation_data**: None or tuple of arrays, (X_test, y_test)<br>
        fit_param
- **seed**: None or int, default 0<br>
        backend random seed
    