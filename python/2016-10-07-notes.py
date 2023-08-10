import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')

N = 100
x = np.linspace(0,1,N)
y = 3.21*x - 1.05 + np.random.randn(N) # Linear response variable with noise
plt.scatter(x,y);

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x.reshape(N,1),y.reshape(N,1))

print('The linear model is:', reg.coef_[0][0], 'x + (', reg.intercept_[0], ')')

X = np.linspace(0,1,10)
Y = reg.predict(X.reshape(10,1))

plt.plot(x,y,'b.',X,Y,'r--');

from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor() # Instantiate the model

clf.fit(x.reshape(N,1),y.reshape(N,1)) # Fit the model

X = np.linspace(0,1,100)
Y = clf.predict(X.reshape(100,1)) # Predict with the model

plt.plot(x,y,'b.',X,Y,'r--');

from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

diabetes.keys()

diabetes.data[:5]

diabetes.target[:5]

names = ['AGE','SEX','BMI','BP','S1','S2','S3','S4','S5','S6']

df = pd.DataFrame(diabetes.data,columns=names)

df['Y'] = diabetes.target.reshape(len(diabetes.target),1)

df.head()

from pandas.tools.plotting import scatter_matrix

scatter_matrix(df,figsize=(12,5));

reg_diabetes = LinearRegression()

reg_diabetes.fit(diabetes.data,diabetes.target)

reg_diabetes.coef_

reg_diabetes.intercept_

