# Model Check
# Oversample Minority
# Shuffle Predictors and Targets such that predictors point to new targets
# Use dataset without Smoteen & Tomek

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
from sklearn import preprocessing as pp
import pickle
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV, train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# X = pd.read_pickle('./couple_data_xgboost_predictors')
# X = pd.read_pickle('./couple_data_predictors')
# X = pd.read_pickle('./couple_data_without_resample_predictors')
X = pd.read_pickle('./couple_data_rev_imbal_predictors')
# X = pd.read_pickle('./couple_data_lasso_predictors')
y = pd.read_pickle('./couple_data_rev_imbal_target')
# y = pd.read_pickle('./couple_data_without_resample_target')
# y = pd.read_pickle('./couple_data_target')

# Shuffling Predictors, swapping rows
# for verification only
# X = X.sample(frac=1, replace=False)

# Scale the resampled features
Xs = StandardScaler().fit_transform(X)
y = y.values.ravel()

# Training and Test set
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.33, random_state=42)

# Gridsearch for Ridge and Lasso Logistic Regression, optimize C

parameters = {
    'penalty':['l1','l2'],
    'solver':['liblinear'],
    'C':np.logspace(-5,0,100)
}

print ("GRID SEARCH:")
lr_grid_search = GridSearchCV(LogisticRegression(), parameters, cv=10, verbose=0)
lr_grid_search.fit(X_train, y_train)
print ("Best parameters set:")
print('f1 score:', lr_grid_search.best_score_)
lr_best_parameters = lr_grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ("\t%s: %r" % (param_name, lr_best_parameters[param_name]))
print ("-----------------------------------------")
print ("-----------------------------------------")
print ("GRIDSEARCHCV LOGREG RESULT:")
clf = lr_grid_search.best_estimator_
lr_gs_predicted = clf.predict(X_test)
print(metrics.classification_report(y_test, lr_gs_predicted, labels=[1,0], target_names=['break up','still together']))

# Get top 10 coefficients
lr_gs_coef = pd.DataFrame({
    'coef':clf.coef_.ravel(),
    'mag':np.abs(clf.coef_.ravel()),
    'pred':X.columns
})
lr_gs_coef.sort_values(by=['mag'], ascending=False, inplace=True)
lr_gs_coef.head(10)

# Gridsearch SGDclassifier with log loss and optimal learning rate
sgd_parameters = {
    'learning_rate': ['optimal'],
    'loss':['log','hinge'],
    'penalty': ['l1','l2','elasticnet'],
    'alpha': np.logspace(-10,5,100),
    'l1_ratio': np.logspace(-1,0,20)
}

print ("GRID SEARCH:")
sgd_grid_search = GridSearchCV(SGDClassifier(max_iter=10000, tol=0.0001), sgd_parameters, cv=10, verbose=0)
sgd_grid_search.fit(X_train, y_train)
print ("Best parameters set:")
print('f1 score:', sgd_grid_search.best_score_)
sgd_best_parameters = sgd_grid_search.best_estimator_.get_params()
for param_name in sorted(sgd_parameters.keys()):
    print ("\t%s: %r" % (param_name, sgd_best_parameters[param_name]))
print ("-----------------------------------------")
print ("-----------------------------------------")
print ("GRIDSEARCHCV SGDCLASSIFIER RESULT:")
sgd_clf = sgd_grid_search.best_estimator_
sgd_predicted = sgd_clf.predict(X_test)
print(metrics.classification_report(y_test, sgd_predicted, labels=[1,0], target_names=['break up','still together']))

# Get top 10 coefficients
sgd_coef = pd.DataFrame({
    'coef':sgd_clf.coef_.ravel(),
    'mag':np.abs(sgd_clf.coef_.ravel()),
    'pred':X.columns
})
sgd_coef.sort_values(by=['mag'], ascending=False, inplace=True)
sgd_coef.head(10)

# Reference link: https://www.kaggle.com/phunter/xgboost-with-gridsearchcv
# Credit to Shize's R code and the python re-implementation

xgb_model = xgb.XGBClassifier()

#brute force scan for all parameters, here are the tricks
#usually max_depth is 6,7,8
#learning rate is around 0.05, but small changes may make big diff
#tuning min_child_weight subsample colsample_bytree can have 
#much fun of fighting against overfit 
#n_estimators is how many round of boosting
#finally, ensemble xgboost with multiple seeds may reduce variance
parameters = {'nthread': [4], #when use hyperthread, xgboost may become slower
              'objective': ['binary:logistic'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [6,7,8],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [1000], #number of trees, change it to 1000 for better results
              'missing': [-999],
              'seed': [65]}

xgb_clf = GridSearchCV(xgb_model, parameters, n_jobs=5, scoring='f1', verbose=0, refit=True, cv=10)

xgb_clf.fit(X_train, y_train)

best_parameters, score, _ = max(xgb_clf.grid_scores_, key=lambda x: x[1])
print('f1 score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
    
print ("-----------------------------------------")
print ("-----------------------------------------")
print ("GRIDSEARCHCV XGBCLASSIFIER RESULT:")
xgb_predicted = xgb_clf.best_estimator_.predict(X_test)
print(metrics.classification_report(y_test, xgb_predicted, labels=[1,0], target_names=['break up','still together']))

# Get top 10 coefficients
xgb_featimpt = pd.DataFrame({
    'importance':xgb_clf.best_estimator_.feature_importances_,
    'pred':X.columns
})
xgb_featimpt = xgb_featimpt[xgb_featimpt.importance > 0]
xgb_featimpt.sort_values(by=['importance'], ascending=False, inplace=True)
xgb_featimpt.head(10)

# Only GridSearch LogReg & Gridsearch SGDClassifier performs normally in an imbalanced dataset

# Check for any feature that are removed in this round which might result in the drop in performance
xgb_featimpt = pd.DataFrame({
    'importance':xgb_clf.best_estimator_.feature_importances_,
    'pred':X.columns
})
xgb_featimpt[xgb_featimpt.importance <= 0]

# Drop in the above features might be the cause of the drop from previous score

# Try TPOT
from tpot import TPOTClassifier

pipeline_optimizer = TPOTClassifier()

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                    random_state=42, verbosity=2)

pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
# pipeline_optimizer.export('tpot_exported_pipeline.py')

yhat = pipeline_optimizer.predict(X_test)
print(metrics.classification_report(y_test, yhat, labels=[1,0], target_names=['break up','still together']))



