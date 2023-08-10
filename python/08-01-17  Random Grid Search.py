import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, weights='uniform', 
                                 metric ='minkowski', p=2)

grid = {'n_neighbors':list(range(1,11)), 'weights':['uniform', 'distance'],
       'p':[1,2], }

from sklearn.grid_search import RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=classifier, 
                                   param_distributions = grid, 
                                   n_iter = 10, scoring = 'accuracy', 
                                   n_jobs=1, refit=True,
                                   cv = 10)
random_search.fit(X,y)

print('Best parameters: %s'%random_search.best_params_)
print('CV Accuracy of best parameters: %.3f'%random_search.best_score_)

from sklearn.cross_validation import cross_val_score
print ('Baesline with default parameters: %.3f' %np.mean(
        cross_val_score(classifier, X, y, cv=10, scoring='accuracy', n_jobs=1)))

random_search.grid_scores_



