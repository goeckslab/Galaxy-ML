### Version 0.4.0 - dev / tool_main: 1.0.0.2

#### New Features

-

#### Changes

- Moves src to subfolder `Galaxy-ML`

#### Bug Fixes

- Fix safepickler classobj issue


### Version 0.3.0/ tool_main: 1.0.0.2

#### New Features

- Makes `RepeatedOrderedKFold`.
- Makes `train_test_split` tool and adds `train_test_split` to searchcv tool.
- Adds `jpickle` to persist sklearn objects.
- Makes `TDMScaler`.
- Makes search parameter `options` in `search_model_validation` tool using `from_dataset`, the `get_params` output of 
`estimator_attributes` tool.
- Restructures `estimator_attributes` tool to be workflow friendly

#### Changes

- Separate `OrderedKFold` into `model_validations` module
- Refactors `SafePickler` class and pickle white list loading system for better CPU and Memory efficiency.
- Separates `feature_selector` module out from `utils`

#### Bug Fixes

- Fix safepickler classobj issue


### Version 0.2.0 (03-24-2019)

##### New Features

- SearchCV tool selects param from `get_params()` dataset.
- Adds `extended_ensemble_ml` tool which wraps `StackingCVRegressor` to ensemble machine learning.
- Extends `estimator_attributes` tool to output `get_params()`
- Adds support of multipleprocessing in `IRAPSCore`

##### Changes

- Removes the limit of `n_jobs=1` for `IRAPSClassifier`
- Changes named estimators in `pipeline_builder` tool. Use `make_pipeline` instead of `Pipeline` initiation.


##### Bug Fixes

- 


### Version 0.1.0 (03-15-2019)

