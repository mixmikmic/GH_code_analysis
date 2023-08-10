import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

df_train = pd.read_csv('car_prices_train.txt')
df_test = pd.read_csv('car_prices_test.txt')
df_train.head(5)

from sklearn.linear_model import LinearRegression 

y = df_train[' price']
x = df_train.drop([' price'],axis=1)

LinReg = LinearRegression()
LinReg.fit(x,y)

LinReg.score(x,y)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x,y)
rf.score(x,y)

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

df = pd.read_csv('dataset_2.txt')

df.head()

wine = pd.read_csv('wine_quality_red.csv', delimiter = ';')

wine.head()





