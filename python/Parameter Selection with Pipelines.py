import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

from sklearn.datasets import make_regression

X, y = make_regression(random_state=42, effective_rank=90)
print(X.shape)

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=.5)

from sklearn.feature_selection import SelectFpr, f_regression
from sklearn.linear_model import Ridge

fpr = SelectFpr(score_func=f_regression)
fpr.fit(X_train, y_train)
X_train_fpr = fpr.transform(X_train)
X_test_fpr = fpr.transform(X_test)

print(X_train_fpr.shape)

ridge = Ridge()
ridge.fit(X_train_fpr, y_train)
ridge.score(X_test_fpr, y_test)

from sklearn.pipeline import make_pipeline

pipe = make_pipeline(SelectFpr(score_func=f_regression), Ridge())

pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

from sklearn.grid_search import GridSearchCV
# without pipeline:
param_grid_no_pipeline = {'alpha': 10. ** np.arange(-3, 5)}

pipe.named_steps.keys()

# with pipeline
param_grid = {'ridge__alpha': 10. ** np.arange(-3, 5)}
grid = GridSearchCV(pipe, param_grid, cv=10)
grid.fit(X_train, y_train)

grid.score(X_test, y_test)

grid.best_params_

param_grid = {'ridge__alpha': 10. ** np.arange(-3, 5),
              'selectfpr__alpha': [0.01, 0.02, 0.05, 0.1, 0.3]}
grid = GridSearchCV(pipe, param_grid, cv=10)
grid.fit(X_train, y_train)
grid.score(X_test, y_test)

grid.best_params_

final_selectfpr = grid.best_estimator_.named_steps['selectfpr']
final_selectfpr.get_support()



