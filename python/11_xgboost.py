from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')
# load data
dataset = loadtxt("data.txt", delimiter=",")
# split data into X and y
X = dataset[:,0:8]
y = dataset[:,8]

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 7, test_size = 0.33)

# eval_metrics = rmse, logloss, error, auc, merror, mlogloss, custom
eval_set =  [(X_test, y_test)]
model = XGBClassifier()
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc", eval_set=eval_set, verbose=True)

import xgboost

y_pred = model.predict_proba(X_test)
y_pred[:20]

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print(model.feature_importances_)

from xgboost import plot_importance
plot_importance(model, )
plt.show()

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

max_depth = [2, 4, 6, 8]
reg_lambda = [0, 1, 2]
param_grid = dict(reg_lambda=reg_lambda, max_depth=max_depth, n_estimators=[200])

get_ipython().magic('pinfo XGBClassifier')

model = XGBClassifier(objective="binary:logistic")
kfold = StratifiedKFold(y, n_folds=5, shuffle=True)
grid_search = GridSearchCV(model, param_grid, scoring="roc_auc", n_jobs=1, cv=kfold, verbose = 1)
grid_result = grid_search.fit(X, y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

eval_set =  [(X_test, y_test)]
colsample_bytree = 0.7
model = XGBClassifier()
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc", eval_set=eval_set, verbose=True)

from xgboost import plot_tree
from matplotlib.pylab import rcParams

plot_tree(model, num_trees=1)
# plt.title("max_depth = 100, with gamma = 10")
# plt.savefig("tree_with_max_depth_gamma", dpi = 700)



