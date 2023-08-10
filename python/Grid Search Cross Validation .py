from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x,y)

clf = KNeighborsClassifier()
grid = {"n_neighbors": [3,5,7,9,11]}
abc = GridSearchCV(clf, grid)
abc.fit(x_train, y_train)

abc.best_estimator_

abc.cv_results_

clf1 = svm.SVC()
grid = {'C' : [1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
       'gamma' : [1e-3, 5e-4, 1e-4, 5e-3] }

abc = GridSearchCV(clf1, grid)
abc.fit(x_train, y_train)

abc.best_estimator_

abc.cv_results_

