### Version 0.7.9 / tool_main: 1.0.7.9 / stacking: 0.2.0 / keras: 0.4.0 (07-25-2019)

#### New Features

- Adds searchcv tools to output `weights` for deep learning models.
- Makes `KerasGBatchClassifier.evalue` to support multi-class and multi-label classification problem.
- Adds parameter `verbose` in KerasG models to output device placement.
- Adds `metrics` in keras model building tools.

#### Changes

- Changes dependency `tensorflow` to `tensorflow-gpu`.
- Moves all tools to folder `tools`.
- Makes `sklearn.preprocessing.Imputer` deprecated.
- Updates dependencies in `requrements.txt`.

#### Bug Fixes

- Fixes `KerasGBatchClassifier` doesn't work with callbacks.
- Fixes `GenomicIntervalBatchGenerator` doesn't work in nested model validation.


### Version 0.7.5 / tool_main: 1.0.7.5 / stacking: 0.2.0 / keras: 0.3.0 (07-09-2019)

#### New Features

- Adds MIT license.
- Adds `setup.py` and `requirement.txt` for APIs installation.
- Makes Galaxy-ML APIs as a library and installable vis pypi and bioconda. 
- Adds `GenomicIntervalBatchGenerator`, an online data generator that provides online genomic sequences transformation from a reference genome and intervals. By trying to offer the same functionalities of [selene](https://github.com/FunctionLab/selene), `GenomicIntervalBatchGenerator` is implemented by, 1) reusing selene cython backend; 2) extending `keras.utils.Sequence`, multiple processing and queueing capable; 3) compatibilizing with sciKit-learn APIs, like KFold, GridSeearchCV, _etc_. `GenomicIntervalBatchGenerator` is supposed to be fast and memory-efficient.
- Adds parameter `steps_per_epoch`, `validation_steps` to `BaseKerasModel`.
- Adds parameter `prediction_steps` to `KerasGBatchClassifier`.
- Adds `class_weight`-like parameter `class_positive_factor` to `KerasGBatchGenerator` for imbalanced training.


#### Changes

- Refactor fast array generators, introduced `fit` method.
- Refactor iraps_classifier random index generator, reduce fit time by about 45%

#### Bug Fixes


### Version 0.7.1 / tool_main: 1.0.7.0 / stacking: 0.2.0 / keras: 0.3.0 (06-11-2019)

#### New Features

- Adds `validation_data` into keras galaxy models and supports gridsearch and `model_validation`.
- Adds fasta sequence batch generator and makes `FastaDNABatchGenerator`, `FastaRNABatchGenerator` and `FastaProteinBatchGenerator`.
- Adds keras galaxy batch classifier and `generator.flow`.
- Adds `keras_batch_models` tool.
- Adds `GenomeOneHotEncoder` and `ProteinOneHotEncoder` to `pipeline`.
- Adds API documentation at `https://goeckslab.github.io/Galaxy-ML/`.
- Extends `BinarizeTarget Classifier/Regressor` to support `fit_params`.
- Modifies `read_columns` function to avoid repeated input file reading.

#### Changes

- Changes `model_validation` tool name.


#### Bug Fixes

- Fixes CV groups file issue in searchcv tool.
- Fixes cheetah error in model_validation tool.


### Version 0.6.5 / tool_main: 1.0.6.2 / stacking: 0.2.0 / keras: 0.2.0 (05-26-2019)

#### New Features

- Adds `BinarizeTargetTransformer`.
- Adds support of binarize_scorers to `BaseSearchCV`.
- Adds `sklearn.ensemble.VotingClassifier` and `VotingRegressor` (will be available sklearn v0.21).
- Enhances security of `try_get_attr` by adding `check_def` argument.
- Adds `__all__` attribute together with `try_get_attr` to manage custom module and names.
- Adds keras callbacks. Now supports `EarlyStopping`, `RemoteMonitor`, `TerminateOnNaN`, `ReduceLROnPlateau` and partially support `ModelCheckpoint`, `CSVLogger`.

#### Changes

- Pumps `stacking_ensembles` too to version 0.2.0.
- Changes `KerasBatchClassifier` to `KerasGBatchClassifier`.

#### Bug Fixes

- Fix voting estimators duplicate naming problem.


### Version 0.6.0 / tool_main: 1.0.6.0 / keras: 0.2.0 (05-13-2019)

#### New Features

- Adds Nested CV to searchcv tool.
- Adds `BinarizeTargetClassifier`.classifier_, `BinarizeTargetRegressor`.regressor_ and `IRAPSClassifier`.get_signature() in estimator_attributes tool.
- Reformat the output of `corss_validate`.
- Adds `KerasBatchClassifier`.
- Makes `KerasGClassifier` and `KerasGRegressor` support multi-dimension array.

#### Changes

- Changes min value of `n_splits`  from 2 to 1.
- Main Tool version changes on the last second number instead of the last one.

#### Bug Fixes

- Fixes `train_test_split` which didn't work with `default` scoring.


### Version 0.5.0 / tool_main: 1.0.0.5 / keras: 0.2.0 (05-13-2019)

#### New Features

- Extend binarize target scorers to support stacking estimators, i.e., use binarize target estimator as meta estimator.
- Adds `cv_results` attributes to `estimator_attributes` tool.
- Adds loading prefitted model for prediction in `keras_model_builder` tool.
- Adds `save_weights` and `load_weights` for keras classifier/regressor models.
- Merges keras model builder

#### Changes

- Refactors the multiple scoring input for searchcv and simplify cv_results output.
- Refactors import system, get rid of exec import.

#### Bug Fixes

- Fixes stacking estimators whitelist issue and other import issues.
- Fixes bases typo error in stacking ensembles tool
- Fixes multiple scoring error in train_test_split mode


### Version 0.4.0 / tool_main: 1.0.0.4 (04-29-2019)

#### New Features

- Adds `StackingCVClassifier`, `StackingClassifier` and `StackingRegressor` to `Stacking_ensembles` tool, and makes explicit base estimator and meta estimator building options.
- Adds `.gitattributes` and `.gitignore`.

#### Changes

- Changes `extended_ensemble_ml.xml` to `stacking_ensembles.xml`.
- Moves src to subfolder `Galaxy-ML`

#### Bug Fixes

- Fix safepickler classobj issue


### Version 0.3.0/ tool_main: 1.0.0.3 (04-23-2019)

#### New Features

- Makes `RepeatedOrderedKFold`.
- Makes `train_test_split` tool and adds `train_test_split` to searchcv tool.
- Adds `jpickle` to persist sklearn objects.
- Makes `TDMScaler`.
- Makes search parameter `options` in `search_model_validation` tool using `from_dataset`, the `get_params` output of 
`estimator_attributes` tool.
- Restructures `estimator_attributes` tool to be workflow friendly.

#### Changes

- Separate `OrderedKFold` into `model_validations` module.
- Refactors `SafePickler` class and pickle white list loading system for better CPU and Memory efficiency.
- Separates `feature_selector` module out from `utils`.

#### Bug Fixes

- Fix safepickler classobj issue


### Version 0.2.0 (03-24-2019)

##### New Features

- SearchCV tool selects param from `get_params()` dataset.
- Adds `extended_ensemble_ml` tool which wraps `StackingCVRegressor` to ensemble machine learning.
- Extends `estimator_attributes` tool to output `get_params()`.
- Adds support of multipleprocessing in `IRAPSCore`.

##### Changes

- Removes the limit of `n_jobs=1` for `IRAPSClassifier`
- Changes named estimators in `pipeline_builder` tool. Use `make_pipeline` instead of `Pipeline` initiation.


##### Bug Fixes

- 


### Version 0.1.0 (03-15-2019)

