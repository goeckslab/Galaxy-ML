
# Galaxy-ML
Galaxy-ML is a web machine learning end-to-end pipeline building framework, with special support to biomedical data. Under the management of unified scikit-learn APIs, cutting-edge machine learning libraries are combined together to provide thousands of different pipelines suitable for various needs. In the form of [Galalxy](https://github.com/galaxyproject/galaxy) tools, Galaxy-ML provides scalabe, reproducible and transparent machine learning computations.


### Key features
-  simple web UI
-  no coding or minimum coding requirement
-  fast model deployment and model selection, specialized in hyperparameter tuning using `GridSearchCV`
-  high level of parallel and automated computation


### Supported modules
A typic machine learning pipeline is composed of a main estimator/model and optional preprocessing component(s).

##### Model
- _[scikit-learn](https://github.com/scikit-learn/scikit-learn)_
    - sklearn.ensemble
    - sklearn.linear_model
    - sklearn.naive_bayes
    - sklearn.neighbors
    - sklearn.svm
    - sklearn.tree
- _[xgboost](https://github.com/dmlc/xgboost)_
    - XGBClassifier
    - XGBRegressor
- _[mlxtend](https://github.com/rasbt/mlxtend)_
    - StackingCVClassifier
    - StackingClassifier
    - StackingCVRegressor
    - StackingRegressor

- _[Keras](https://github.com/keras-team/keras)_ (Deep learning models are re-implemented to fully support sklearn APIs. Supports parameter, including layer subparameter, swaps or searches.  Supports `callbacks`)
    - KerasGClassifier
    - KerasGRegressor
    - KerasGBatchClassifier (works best with online data generators, processing images, genomic sequences and so on)
    
- BinarizeTargetClassifier/BinarizeTargetRegressor
- [IRAPSClassifier](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5445594/)
  
##### Preprocessor
- _[scikit-learn](https://github.com/scikit-learn/scikit-learn)_
    - sklearn.preprocessing
    - sklearn.feature_selection
    - sklearn.decomposition
    - sklearn.kernel_approximation
    - sklearn.cluster
- _[imblanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn)_
    - imblearn.under_sampling
    - imblearn.over_sampling
    - imblearn.combine
- _[skrebate](https://github.com/EpistasisLab/scikit-rebate/tree/master/skrebate)_
    - ReliefF
    - SURF
    - SURFstar
    - MultiSURF
    - MultiSURFstar
- [TDMScaler](https://www.ncbi.nlm.nih.gov/pubmed/26844019)
- DyRFE/DyRFECV
- Z_RandomOverSampler
- GenomeOneHotEncoder
- ProteinOneHotEncoder
- FastaDNABatchGenerator
- FastaRNABatchGenerator
- FastaProteinBatchGenerator
- GenomicIntervalBatchGenerator
- GenomicVariantBatchGenerator
- ImageDataFrameBatchGenerator


### Installation
APIs for models, preprocessors and utils implemented in Galaxy-ML can be installed separately.

##### Installing using anaconda (recommended)
```
conda install -c bioconda -c conda-forge Galaxy-ML
```

##### Installing using pip
```
pip install -U Galaxy-ML
```

##### Installing from source
```
python setup.py install
```

##### Using source code inplace
```
python install -e .
```

To install Galaxy-ML tools in Galaxy, please refer to https://galaxyproject.org/admin/tools/add-tool-from-toolshed-tutorial/.

### Running the tests

Before running the tests, run the following commands:

```
conda create --name galaxy_ml python=3.9
conda activate galaxy_ml
pip install -e .
pip install nose nose-htmloutput pytest
cd galaxy_ml
```

To run all tests and generate an HTML report:
```
nosetests ./tests --with-html --html-file=./report.html
```

To run tests in a specific file (e.g., test_keras_galaxy.py file) and generate an HTML report
```
nosetests ./tests/test_keras_galaxy.py --with-html --html-file=./report.html
```

To run a specific test in a specific file (e.g., test_multi_dimensional_output test in test_keras_galaxy.py file) and generate an HTML report
```
nosetests ./tests/test_keras_galaxy.py:test_multi_dimensional_output --with-html --html-file=./report.html
```

### Examples for using Galaxy-ML custom models

```
# handle imports
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import GridSearchCV
from galaxy_ml.keras_galaxy_models import KerasGClassifier


# build a DNN classifier
model = Sequential()
model.add(Dense(64))
model.add(Activation(‘relu'))
model.add((Dense(1, activation=‘sigmoid’)))
config = model.get_config()

classifier = KerasGClassifier(config, random_state=42)


# clone a classifier
clf = clone(classifier)


# Get parameters
params = clf.get_params()


# Set parameters
new_params = dict(
    epochs=60,
    lr=0.01,
    layers_1_Dense__config__kernel_initializer__config__seed=999,
    layers_0_Dense__config__kernel_initializer__config__seed=999
)
clf.set_params(**new_params)


# model evaluation using GridSearchCV
grid = GridSearchCV(clf, param_grid={}, scoring=‘roc_auc’, cv=5, n_jobs=2)
grid.fit(X, y)
```

### Example for using Galaxy-ML to persist a sklearn/keras model

```
from galaxy_ml.model_persist import (dump_model_to_h5,
                                     load_model_from_h5)
                 
# dump model to hdf5
dump_model_to_h5(model, `save_path`,
                 store_hyperparameter=True)

# load model from hdf5
model = load_model_from_h5(`path_to_hdf5`)
```

#### Performance comparison

Galaxy-ML's HDF5 saving utils perform faster than cPickle for large, array-rich models.

```
Loading model using pickle...
(1.2471628189086914 s)

Dumping model using pickle...
(3.6942389011383057 s)
File size: 930712861

Dumping model to hdf5...
(3.006715774536133 s)
File size: 930729696

Loading model from hdf5...
(0.6420958042144775 s)

Pipeline(memory=None,
         steps=[('robustscaler',
                 RobustScaler(copy=True, quantile_range=(25.0, 75.0),
                              with_centering=True, with_scaling=True)),
                ('kneighborsclassifier',
                 KNeighborsClassifier(algorithm='auto', leaf_size=30,
                                      metric='minkowski', metric_params=None,
                                      n_jobs=1, n_neighbors=100, p=2,
                                      weights='uniform'))],
         verbose=False)
```

#### Publication


Gu Q, Kumar A, Bray S, Creason A, Khanteymoori A, Jalili V, et al. (2021) Galaxy-ML: An accessible, reproducible, and scalable machine learning toolkit for biomedicine. PLoS Comput Biol 17(6): e1009014. https://doi.org/10.1371/journal.pcbi.1009014
