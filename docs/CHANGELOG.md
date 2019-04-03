
### Version 0.3.0.dev

#### New Features

- Makes `TDMScaler`.
- Makes search parameter `options` in `search_model_validation` tool using `from_dataset`, the `get_params` output of 
`estimator_attributes` tool.
- Restructures `estimator_attributes` tool to be workflow friendly

#### Changes

- Refactors `SafePickler` class and pickle white list loading system for better CPU and Memory efficiency.
- Separates `feature_selector` module out from `utils`

#### Bug Fixes

- 


### Version 0.2.0 (03-24-2019)

##### New Features

- Adds `extended_ensemble_ml` tool which wraps `StackingCVRegressor` to ensemble machine learning.
- Extends `estimator_attributes` tool to output `get_params()`
- Adds support of multipleprocessing in `IRAPSCore`

##### Changes

- Removes the limit of `n_jobs=1` for `IRAPSClassifier`
- Changes named estimators in `pipeline_builder` tool. Use `make_pipeline` instead of `Pipeline` initiation.


##### Bug Fixes

- 


### Version 0.1.0 (03-15-2019)

