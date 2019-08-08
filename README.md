
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
- ImageBatchGenerator


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
python setup.py build_ext --inplace
```

To install Galaxy-ML tools in Galaxy, please refer to https://galaxyproject.org/admin/tools/add-tool-from-toolshed-tutorial/.


### Examples for using Galaxy-ML APIs

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

