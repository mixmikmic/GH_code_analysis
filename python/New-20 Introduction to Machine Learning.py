### Data representation in scikit-learn
### yxur9195
import seaborn as sns
iris = sns.load_dataset('iris')

iris

iris.shape

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np

rng = np.random.RandomState(11)
x = 10 * rng.rand(50)
x



x[:,np.newaxis].shape #add one more dim to data
#x.reshape((50,1))
#Another approach
x.reshape((50,1))

y = 8*x + 145

y



from sklearn.linear_model import LinearRegression



#create empty logic or brain
model = LinearRegression(fit_intercept=True)

#Trained the model with feature & target
#Input data should be in form of list of rows
model.fit(x.reshape(50,1),y)

x.reshape(50,1)
#or
x[:,np.newaxis]

model.coef_

model.intercept_

model.predict([[5.6181871]])

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.scatter(x,y)

model.predict([[5.6181871]])

y

y = ax^2 + d^x + bz + c

from statsmodels.tsa.arima_model import ARIMA

import pandas as pd

pd.read_csv('house_rental.csv.txt',index_col='Unnamed: 0')



