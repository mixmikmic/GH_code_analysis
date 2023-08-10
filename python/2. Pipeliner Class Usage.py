from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedKFold

from reskit.core import Pipeliner


# Feature selection and feature extraction step variants (1st step)
feature_engineering = [('VT', VarianceThreshold()),
                       ('PCA', PCA())]

# Preprocessing step variants (2nd step)
scalers = [('standard', StandardScaler()),
           ('minmax', MinMaxScaler())]

# Models (3rd step)
classifiers = [('LR', LogisticRegression()),
               ('SVC', SVC()),
               ('SGD', SGDClassifier())]

# Reskit needs to define steps in this manner
steps = [('feature_engineering', feature_engineering),
         ('scaler', scalers),
         ('classifier', classifiers)]

# Grid search parameters for our models
param_grid = {'LR': {'penalty': ['l1', 'l2']},
              'SVC': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
              'SGD': {'penalty': ['elasticnet'],
                      'l1_ratio': [0.1, 0.2, 0.3]}}

# Quality metric that we want to optimize
scoring='roc_auc'

# Setting cross-validations
grid_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
eval_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

pipe = Pipeliner(steps, grid_cv=grid_cv, eval_cv=eval_cv, param_grid=param_grid)
pipe.plan_table

banned_combos = [('minmax', 'SVC')]
pipe = Pipeliner(steps, grid_cv=grid_cv, eval_cv=eval_cv, param_grid=param_grid, banned_combos=banned_combos)
pipe.plan_table

from sklearn.datasets import make_classification


X, y = make_classification()
pipe.get_results(X, y, scoring=['roc_auc'])

from sklearn.preprocessing import Binarizer

# Simple binarization step that we want ot cache
binarizer = [('binarizer', Binarizer())]

# Reskit needs to define steps in this manner
steps = [('binarizer', binarizer),
         ('classifier', classifiers)]

pipe = Pipeliner(steps, grid_cv=grid_cv, eval_cv=eval_cv, param_grid=param_grid)
pipe.plan_table

pipe.get_results(X, y, caching_steps=['binarizer'])

pipe._cached_X

