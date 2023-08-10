# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve, ShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from IPython.display import display
import sklearn.cross_validation as cv

X = pd.read_csv('~/projects/capstone/model_selection/pro3_Data.csv')
y = pd.read_csv('~/projects/capstone/model_selection/pro3_y.csv', header= None)
y = y[0]

import sklearn
import sklearn.model_selection

X_train, X_test, y_train, y_test= sklearn.model_selection.train_test_split(X, y)


from __future__ import print_function
import sklearn.metrics
import time

def evaluate_model(clf):
    """Scores a model using log loss with the created train and test sets."""
    start = time.time()
    clf.fit(X_train, y_train)
    train_loss = sklearn.metrics.log_loss(y_train, clf.predict_proba(X_train))
    test_loss = sklearn.metrics.log_loss(y_test, clf.predict_proba(X_test))
    print("Train score:", train_loss)
    print("Test score:", test_loss)
    print("Total time:", time.time() - start)
    print()
    return test_loss

# grading boosting model 
from sklearn.ensemble import GradientBoostingClassifier

gbt = GradientBoostingClassifier()
evaluate_model(gbt)


def plot_gbt_learning(gbt):
    test_score = np.empty(len(gbt.estimators_))
    train_score = np.empty(len(gbt.estimators_))
    for i, pred in enumerate(gbt.staged_predict_proba(X_test)):
        test_score[i] = sklearn.metrics.log_loss(y_test, pred)
    for i, pred in enumerate(gbt.staged_predict_proba(X_train)):
        train_score[i] = sklearn.metrics.log_loss(y_train, pred)
    plt.figure(figsize=(8,6))
    plt.plot(np.arange(gbt.n_estimators) + 1, test_score, label='Test') 
    plt.plot(np.arange(gbt.n_estimators) + 1, train_score, label='Train')

plot_gbt_learning(gbt)

#Definitely overfitting, lets add more estimators:
gbt = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2)

evaluate_model(gbt)
plot_gbt_learning(gbt)

#run a grid search to find best params:
from sklearn.grid_search import GridSearchCV

params = {'max_depth' : [3,4,5], 'n_estimators':[60,80, 100],'learning_rate': [0.1,0.15], 
          'subsample': [0.5,1.0]}
grid = GridSearchCV(GradientBoostingClassifier(), params, scoring='log_loss')
evaluate_model(grid)
grid.grid_scores_

best_grid = grid.best_estimator_

params = {'max_depth' : 4, 'n_estimators': 60,'learning_rate': 0.1}
gbt = GradientBoostingClassifier(**params)
sample_weight = map(lambda x: 1 if x == 0 else 7, y_train)
#gbt.fit(x_train,y_train)
gbt.fit(x_train,y_train, sample_weight = sample_weight)
gbt.score(x_train,y_train)

best_grid.fit(X_train, y_train)

best_grid.score(X_train, y_train)

from sklearn.metrics import classification_report

gbt_pred_train = best_grid.predict(X_train)
target_names = ['class No','class Yes']
print (classification_report(y_train,gbt_pred_train, target_names = target_names))


gbt_pred_Test = best_grid.predict(X_test)
target_names = ['class No','class Yes']
print (classification_report(y_test,gbt_pred_Test, target_names = target_names))


feature_importance = gbt.feature_importances_
#make importances relative to max importance
feature_importance = 100.0*(feature_importance / feature_importance.max())
show_features = feature_importance[:10] #top twenty
sorted_idx =np.argsort(show_features)
pos = np.arange(sorted_idx.shape[0]) + 0.5
plt.barh(pos, show_features[sorted_idx], align='center')
plt.yticks(pos,X.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


feature_importance = best_grid.feature_importances_
#make importances relative to max importance
feature_importance = 100.0*(feature_importance / feature_importance.max())
show_features = feature_importance[:10] #top twenty
sorted_idx =np.argsort(show_features)
pos = np.arange(sorted_idx.shape[0]) + 0.5
plt.barh(pos, show_features[sorted_idx], align='center')
plt.yticks(pos,X.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()



