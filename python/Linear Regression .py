import numpy as np 

data = np.loadtxt("data.csv", delimiter = ",")

data.shape

type(data)

x = data[:,0].reshape(-1,1)        #reshape from 2d array to 1d
y = data[:,1]

y.shape

from sklearn import model_selection 
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y)

from sklearn.linear_model import LinearRegression 
alg1 = LinearRegression()
alg1.fit(x_train, y_train)

alg1.coef_

alg1.intercept_

import matplotlib.pyplot as plt

m = alg1.coef_[0]
c = alg1.intercept_

x_line = np.arange(30,70,0.1)
y_line = m * x_line + c

plt.plot(x_line, y_line,"r", 30,70)


plt.scatter(x_train, y_train)
plt.show()

import matplotlib.pyplot as plt

m = alg1.coef_[0]
c = alg1.intercept_

x_line = np.arange(30,70,0.1)
y_line = m * x_line + c

plt.plot(x_line, y_line,"r", 30,70)


plt.scatter(x_test, y_test)
plt.show()

score_test = alg1.score(x_test, y_test)        #coefficient of determination
score_test

score_train = alg1.score(x_train, y_train)
score_train, score_test

