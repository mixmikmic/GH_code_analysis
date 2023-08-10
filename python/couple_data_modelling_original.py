# Model Check
# Oversample Minority
# Shuffle Predictors and Targets such that predictors point to new targets
# Use dataset without Smoteen & Tomek

import pandas as pd
import numpy as np
pd.set_option('max_colwidth',100)
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

# pred = pd.read_pickle('./couple_data_without_resample_lasso_feature_selection_predictors')
pred = pd.read_pickle('./couple_data_without_resample_lasso_feature_selection_predictors')
target = pd.read_pickle('./couple_data_without_resample_target')

X_train, X_test, y_train, y_test = train_test_split(pred.values, target.values.ravel(), test_size=0.3, random_state=42)

1 - target.mean()

# Gridsearch for Ridge and Lasso Logistic Regression, optimize C

parameters = {
    'penalty':['l1','l2'],
    'solver':['liblinear'],
    'C':np.logspace(-5,0,100)
}

print ("GRID SEARCH:")
lr_grid_search = GridSearchCV(LogisticRegression(), parameters, cv=5, verbose=1, n_jobs=-1)
lr_grid_search.fit(X_train, y_train)
print ("Best parameters set:")
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
    'pred':pred.columns
})
lr_gs_coef.sort_values(by=['mag'], ascending=False, inplace=True)
lr_gs_coef.head(10)

# Lasso LogReg factors that contribute to couple breakup
lr_gs_coef[lr_gs_coef.coef > 0]

# Race in Gender Race can either be white or non-white
# median hhincome is 67250

# Lasso LogReg factors that contribute to couple staying together
lr_gs_coef[lr_gs_coef.coef < 0]

# Gridsearch SGDclassifier with log loss and optimal learning rate
sgd_parameters = {
    'learning_rate': ['optimal'],
    'loss':['log','hinge'],
    'penalty': ['l1','l2','elasticnet'],
    'alpha': np.logspace(-10,5,100),
    'l1_ratio': np.logspace(-1,0,50)
}

print ("GRID SEARCH:")
sgd_grid_search = GridSearchCV(SGDClassifier(max_iter=10000, tol=0.0001), sgd_parameters, cv=5, verbose=1)
sgd_grid_search.fit(X_train, y_train)
print ("Best parameters set:")
sgd_best_parameters = sgd_grid_search.best_estimator_.get_params()
for param_name in sorted(sgd_parameters.keys()):
    print ("\t%s: %r" % (param_name, sgd_best_parameters[param_name]))
print ("-----------------------------------------")
print ("-----------------------------------------")
print ("GRIDSEARCHCV SGDCLASSIFIER RESULT:")
sgd_clf = sgd_grid_search.best_estimator_
sgd_predicted = sgd_clf.predict(X_test)
print(metrics.classification_report(y_test, sgd_predicted, labels=[1,0], target_names=['break up','still together']))

sgd_coef = pd.DataFrame({
    'coef':sgd_clf.coef_.ravel(),
    'mag':np.abs(sgd_clf.coef_.ravel()),
    'pred':pred.columns
})
sgd_coef.sort_values(by=['mag'], ascending=False, inplace=True)

sgd_coef.head(10)

# SGD factors that contribute to couple breakup
sgd_coef[sgd_coef.coef > 0]

# SGD factors that contribute to couple staying together
sgd_coef[sgd_coef.coef < 0]

from sklearn.svm import SVC

svc_params = {
    'C':np.logspace(-5, 2, 100),
    'gamma':np.logspace(-5, 2, 100),
    'kernel':['linear']
}

svc_gs = GridSearchCV(SVC(max_iter=10000, tol=0.0001), svc_params, cv=5, verbose=1, n_jobs=-1)
svc_gs.fit(X_train, y_train)

best_parameters = svc_gs.best_estimator_.get_params()
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

yhat = svc_gs.best_estimator_.predict(X_test)
print(metrics.classification_report(y_test, yhat, labels=[1,0], target_names=['break up','still together']))

svc_gs_coef = pd.DataFrame({
    'coef':svc_gs.best_estimator_.coef_.ravel(),
    'mag':np.abs(svc_gs.best_estimator_.coef_.ravel()),
    'pred':pred.columns
})
svc_gs_coef.sort_values(by=['mag'], ascending=False, inplace=True)

svc_gs_coef.head(10)

# SGD factors that contribute to couple breakup
svc_gs_coef[svc_gs_coef.coef > 0]

# SGD factors that contribute to couple staying together
svc_gs_coef[svc_gs_coef.coef < 0]

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
              'gamma': [0,0.1],
              'subsample': [0.8],
              'colsample_bytree': [0.8],
              'silent': [1],
              'scale_pos_weight': [1],
              'n_estimators': [1000], #number of trees, change it to 1000 for better results
              'missing': [-999],
              'seed': [65]}

xgb_clf = GridSearchCV(xgb_model, parameters, n_jobs=-1, verbose=1, cv=5)

xgb_clf.fit(X_train, y_train)

best_parameters = xgb_clf.best_estimator_.get_params()
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
    
print ("-----------------------------------------")
print ("-----------------------------------------")
print ("GRIDSEARCHCV XGBCLASSIFIER RESULT:")
xgb_predicted = xgb_clf.best_estimator_.predict(X_test)
print(metrics.classification_report(y_test, xgb_predicted, labels=[1,0], target_names=['break up','still together']))

xgb_featimpt = pd.DataFrame({
    'importance':xgb_clf.best_estimator_.feature_importances_,
    'pred':pred.columns
})
xgb_featimpt.sort_values(by=['importance'], ascending=False, inplace=True)

xgb_featimpt.head(10)

# Wrap up with Dashboard and Explaination

