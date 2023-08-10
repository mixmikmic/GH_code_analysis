import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sklearn import linear_model

def func_predict(x):
    return 2*x + 5

inp = np.random.uniform(-10,10,10)
out = map(func_predict,inp)
print inp
print out
plt.plot(inp,out,'o');

lin_reg_model = linear_model.LinearRegression()
lin_reg_model.fit(inp.reshape(-1,1), out)

print lin_reg_model.coef_
print lin_reg_model.intercept_

plt.plot(inp,out,'o')
plt.plot(inp,lin_reg_model.predict(inp.reshape(-1,1)));

def func_predict_with_error(x):
    return 2*x + 5 + np.random.normal(scale=1,size=1)

inp_error = np.random.uniform(-10,10,10)
out_error = map(func_predict_with_error,inp_error)

lin_reg_model_error = linear_model.LinearRegression()
lin_reg_model_error.fit(inp_error.reshape(-1,1), out_error)

print lin_reg_model_error.coef_
print lin_reg_model_error.intercept_

plt.plot(inp_error, out_error, 'o')
plt.plot(inp_error, lin_reg_model_error.predict(inp_error.reshape(-1,1)));

from sklearn import metrics

X_nonlinear_train = np.random.random(size=(20, 1))
Y_nonlinear_train = 2 * X_nonlinear_train.squeeze() + 5 + 0.1*np.random.randn(20)

## We will use 20 points as training data and the rest 10 points as test data
X_nonlinear_test = np.ones((10,1)) + np.random.random(size=(10,1))
Y_nonlinear_test = 2 * X_nonlinear_test.squeeze() + 5 + 0.1*np.random.random(10)
plt.plot(X_nonlinear_train, Y_nonlinear_train,'o');

lin_model = linear_model.LinearRegression()
lin_model.fit(X_nonlinear_train, Y_nonlinear_train)

print "Mean absolute training error for linear model: ", metrics.mean_absolute_error(Y_nonlinear_train, lin_model.predict(X_nonlinear_train))
print "Mean absolute test error for linear model: ", metrics.mean_absolute_error(Y_nonlinear_test, lin_model.predict(X_nonlinear_test))
plt.plot(X_nonlinear_train, Y_nonlinear_train,'o')
plt.plot(X_nonlinear_train, lin_model.predict(X_nonlinear_train));

## Another way is numpy.polyfit
from numpy import linspace
from sklearn.preprocessing import PolynomialFeatures

def polynomial_regression(X_train, Y_train, X_test, Y_test, deg=2):
    poly = PolynomialFeatures(degree=deg)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.fit_transform(X_test)
    poly_model = linear_model.LinearRegression()
    poly_model.fit(X_poly_train, Y_train)
    
    x_sample = linspace(0, 1, 200)
    y_sample = poly_model.intercept_
    
    for i in range(deg+1):
        y_sample +=  poly_model.coef_[i]*x_sample**i
    
    print "Mean absolute training error for degree={} is: {}".format(deg, metrics.mean_absolute_error(Y_train, poly_model.predict(X_poly_train)))
    print "Mean absolute validation error for degree={} is: {}".format(deg, metrics.mean_absolute_error(Y_test, poly_model.predict(X_poly_test)))
    
    plt.plot(X_train, Y_train, 'o')
    plt.plot(x_sample, y_sample);

polynomial_regression(X_nonlinear_train, Y_nonlinear_train, X_nonlinear_test, Y_nonlinear_test, 2)

polynomial_regression(X_nonlinear_train, Y_nonlinear_train, X_nonlinear_test, Y_nonlinear_test, 10)

polynomial_regression(X_nonlinear, Y_nonlinear, X_nonlinear_test, Y_nonlinear_test, 4)

polynomial_regression(X_nonlinear, Y_nonlinear, X_nonlinear_test, Y_nonlinear_test, 5)

polynomial_regression(X_nonlinear, Y_nonlinear, X_nonlinear_test, Y_nonlinear_test, 6)

polynomial_regression(X_nonlinear, Y_nonlinear, X_nonlinear_test, Y_nonlinear_test, 30)

from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target

##Actual input values
## 64 features
## Value in the range of 0 to 100
print X[0,:].reshape(8,8)
print y[0]

from sklearn import neighbors
import numpy

KNN_classifier = neighbors.KNeighborsClassifier(n_neighbors=1)
#KNN_classifier = ensemble.RandomForestClassifier(n_estimators=50)
KNN_classifier.fit(X, y)
y_pred = KNN_classifier.predict(X)

from sklearn import metrics

print("{0:.2f}".format(metrics.accuracy_score(y, y_pred)))

from sklearn.cross_validation import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3)
X_train.shape, X_validation.shape

for i in range(1,51):
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    #knn = ensemble.RandomForestClassifier(n_estimators=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_validation)
    print "Validation Accuracy (neighbors = {}): {}".format(i, metrics.accuracy_score(y_validation, y_pred))

from sklearn.cross_validation import cross_val_score

for i in range(1,11):
    cv = cross_val_score(neighbors.KNeighborsClassifier(n_neighbors=i), X, y, cv=5)
    print "Accuracy (neighbors = {}): {}".format(i,cv.mean())



