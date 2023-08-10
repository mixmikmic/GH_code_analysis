get_ipython().magic("run '6.1 Model Evaluation and HyperParameter Tuning.ipynb'")
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

p = Pipeline([('scl', StandardScaler()),
             ('clf', SVC(random_state=1))])

param_range=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = [{
        'clf__C': param_range,
        'clf__kernel': ['linear']
    }, {
        'clf__C': param_range,
        'clf__kernel': ['rbf'],
        'clf__gamma': param_range
    }]

# cv = 10 => k-fold with k=10
gs = GridSearchCV(estimator=p, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=10,
                  n_jobs=-1)

gs.fit(X_train, y_train)
print "GridSearch Results: \nBestScore = %s,\nBestParams= %s" %(gs.best_score_, gs.best_params_)

#Estimate test data
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print "Test data accuracy score with grid_search best_estimator: %.3f" %(clf.score(X_test, y_test))

from IPython.display import Image
Image("/Users/surthi/gitrepos/ml-notes/images/nested-cross-validation.jpg")

from sklearn.model_selection import cross_val_score
import numpy as np

# SVM
gs = GridSearchCV(estimator=p, 
                  param_grid=param_grid,
                  scoring='accuracy', 
                  cv=5, 
                  n_jobs=-1)
scores = cross_val_score(gs, X, y, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=0),
    param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
    scoring='accuracy', 
    cv=5)
scores = cross_val_score(gs, X, y, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

