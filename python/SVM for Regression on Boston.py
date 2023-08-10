from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

boston = datasets.load_boston()
x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)

## SVR is Support Vector Regressor
clf = svm.SVR(kernel = "rbf")
clf.fit(x_train, y_train)

clf.score(x_test, y_test)

clf = svm.SVR(kernel = "linear")
clf.fit(x_train, y_train)

clf.score(x_test, y_test)

clf = svm.SVR()
grid = { 'C' : [1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
       'gamma': [1e-3, 5e-4, 1e-4, 5e-3]}

abc = GridSearchCV( clf, grid)
abc.fit(x_train, y_train)

abc.best_estimator_

abc.score(x_test, y_test)

