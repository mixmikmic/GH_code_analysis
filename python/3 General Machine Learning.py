get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_process import arma_generate_sample
xs = arma_generate_sample([1.0, -0.6, 1.0, -0.6], [1.0], 200, 1.0, burnin=100)
plt.plot(xs)

def embed_data(x, steps):
    n = len(x)
    xout = np.zeros((n - steps, steps))
    yout = x[steps:]
    for i in np.arange(steps, n):
        xout[i - steps] = x[i-steps:i]
    return xout, yout

x, y = embed_data(np.array([1,2,3,4,5]), 2)

x, y

import sklearn
from sklearn import svm

model = svm.SVR(kernel='linear', C=1)

model.fit(x, y)

model.predict(x)

x, y = embed_data(xs, 5)

model.fit(x, y)

plt.plot(model.predict(x), y, '.')

from sklearn import metrics

metrics.mean_squared_error(model.predict(x), y)

np.linalg.norm(model.predict(x) - y)**2 / np.linalg.norm(y)**2

train = xs[:150]
test = xs[150:]
xtrain, ytrain = embed_data(train, 3)
xtest, ytest = embed_data(test, 3)

m = svm.SVR(kernel='rbf', C=1, gamma=0.1)
m.fit(xtrain, ytrain)

plt.plot(m.predict(xtrain), 'b-', ytrain, 'r-')

plt.plot(m.predict(xtest), 'b-', ytest, 'r-')



