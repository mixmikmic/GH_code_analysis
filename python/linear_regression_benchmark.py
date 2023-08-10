import graphlab as gl

data = gl.SFrame('https://static.turi.com/datasets//airlines-2008.gl')

#Wrapper function for timeit module
#Uses GraphLab Create to compute a linear regression model on the data
def GLC_linear_regression(data, model):
    #Create linear regression model and save it in model
    model.append(gl.linear_regression.create(data, 
                                             target='ActualElapsedTime',
                                             verbose=False, 
                                             solver='newton',
                                             l1_penalty=0, 
                                             l2_penalty=0,
                                             validation_set=None, 
                                             feature_rescaling=False, 
                                             max_iterations=1))

import pandas as pd
from sklearn import linear_model

#Wrapper function for timeit module
#Uses Scikit-Learn to compute a linear regression model on the data
def SKL_linear_regression(features, target, model):
    #Create linear regression model and save it in model
    model.append(linear_model.LinearRegression())
    model[0].fit(features, target)

from timeit import timeit

def GLC_time_evaluate(data, time, rmse):
    #List to capture the model within the timeit function
    model = []
    
    #Time GLC linear regression training time
    time.append(timeit(lambda : GLC_linear_regression(data, model), number=1))
    
    #Save the training RMSE of the GLC model
    rmse.append(model[0].get('training_rmse'))

from sklearn.metrics import mean_squared_error

def SKL_time_evaluate(data, time, rmse):
    #Transform the data into a pandas data frame
    df = data.to_dataframe()
    
    #Split the target from the features for SKL
    target = df['ActualElapsedTime']
    features = df.drop('ActualElapsedTime', 1)
    
    #One-hot-encode categorical features
    features = pd.get_dummies(features)

    #List to capture the model within the timeit function
    model = []
    
    #Time SKL linear regression training time
    time.append(timeit(lambda : SKL_linear_regression(features, target, model), number=1))
    
    #Save the training RMSE of the GLC model
    rmse.append(mean_squared_error(target, model[0].predict(features))**0.5)

glc_time = []
glc_rmse = []

skl_time = []
skl_rmse = []

#Time both libraries on datasets of increasing size
rows_range = range(5000, 90001, 2500)
for n_rows in rows_range:  
    #Create a new SFrame with n_rows out of the data
    sf = data.head(n_rows)
    
    #Time and evaluate GLC linear regression model
    GLC_time_evaluate(sf, glc_time, glc_rmse)
    
    #Time and evaluate SKL linear regression model
    SKL_time_evaluate(sf, skl_time, skl_rmse)

import numpy as np
import graphlab.numpy

np_time = []
np_rmse = []

#Time skl using SFrame backed numpy arrays on datasets of increasing size
rows_range = range(5000, 90001, 2500)
for n_rows in rows_range:
    #Create a new SFrame with n_rows out of the data
    sf = data.head(n_rows)
    
    #Time and evaluate SKL linear regression model backed by SFrame
    SKL_time_evaluate(sf, np_time, np_rmse)

rows_range = range(92500, 100001, 2500)
for n_rows in rows_range:
    #Create a new SFrame with n_rows out of the data
    sf = data.head(n_rows)
    
    #Time and evaluate GLC linear regression model
    GLC_time_evaluate(sf, glc_time, glc_rmse)
    
    #Time and evaluate SKL linear regression model backed by SFrame
    SKL_time_evaluate(sf, np_time, np_rmse)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

skl_range = np.array(range(5000, 90001, 2500))
glc_range = np.array(range(5000, 100001, 2500))

plt.figure()

plt.scatter(skl_range, skl_rmse, label='scikit-learn', c='g', marker='o', s=100)
plt.scatter(glc_range, np_rmse, label='scikit-learn backed by SFrame', c='y', marker='*', s=100)
plt.scatter(glc_range, glc_rmse, label='GraphLab Create', c='b', marker='+', s=100)

plt.title('Linear Regression Error')
plt.xlabel('Rows of Data')
plt.ylabel('RMSE')

plt.ylim((min(glc_rmse) - 1, max(glc_rmse) + 1))
plt.xlim((0, max(glc_range)))

plt.legend(loc=4)

plt.show()

plt.figure()

plt.scatter(skl_range, skl_time, label='scikit-learn', c='g', marker='o', s=75)
plt.scatter(glc_range, np_time, label='scikit-learn backed by SFrame', c='y', marker='o', s=75)
plt.scatter(glc_range, glc_time, label='GraphLab Create', c='b', marker='o', s=75)

plt.annotate(s='Out of Memory', xy=(skl_range[-1], skl_time[-1]))

plt.title('Linear Regression Training Time')
plt.xlabel('Rows of Data')
plt.ylabel('Seconds')

plt.ylim((0, max(np_time)))
plt.xlim((0, max(glc_range)))

plt.legend(loc=2)

plt.show()

timeit(lambda : GLC_linear_regression(data, []), number=1)

