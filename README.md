
# Galaxy-ML
Galaxy-ML is a web machine learning pipeline building framework, with special support to biomedical data. Under the management of unified scikit-learn APIs, cutting-edge machine learning libraries are combined together to provide thousands of different pipelines suitable for various needs. In the form of [Galalxy](https://github.com/galaxyproject/galaxy) tools, Galaxy-ML provides scalabe, reproducible and transparent machine learning computations.


### Key features:
-  simple web UI
-  no coding or minimum coding requirement
-  fast model deployment and model selection, specialized in hyperparameter tuning using `GridSearchCV`
-  high level of parallel and automated computation


### Supported modules:
A typic machine learning pipeline is composed of a main estimator/model and optional preprocessing component(s).

##### Model:
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
- _[keras](https://github.com/keras-team/keras)_
  - KerasClassifier
  - KerasRegressor
  
##### Preprocessor:
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
  
##### Custom implementations for biomedical application
- [IRAPSClassifier](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5445594/)
- BinarizeTargetClassifier/BinarizeTargetRegressor
- [TDMScaler](https://www.ncbi.nlm.nih.gov/pubmed/26844019)
- DyRFE/DyRFECV
- Z_RandomOverSampler

### Examples
1. Build a simple randomforest model.
