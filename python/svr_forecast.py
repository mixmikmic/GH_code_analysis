import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pylab as plt

import time
import datetime

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

'''
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
'''

##### read data #####
bitcoin=pd.read_csv("/Users/jinshuning/Documents/Semester6/CS5751/Project/data/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv")

print (bitcoin.head())
print (bitcoin.shape)

##### 1 Preprocessing #####

# unix timestamp to normal time
timestamp = [datetime.datetime.fromtimestamp(int(t)).strftime('%Y-%m-%d %H:%M:%S') for t in bitcoin['Timestamp']]

bitcoin['Timestamp'] =  timestamp

timestamp[:10]

'''
start =  timestamp.index('2017-01-01 00:00:00')
end = timestamp.index('2018-01-01 00:00:00') 
print (start,end)
'''

# daily data
# from 2017-01-01 00:00:00 to 2018-01-01 00:00:00
timestamp_subset = [timestamp.index(t) for t in timestamp 
     if '00:00:00' in t and '2017-01-01 00:00:00'<=t<='2018-01-01 00:00:00']

subset =  bitcoin.iloc[timestamp_subset,]

subset

x = subset['Close']

plt.plot(x)
plt.show()

# log transform to stablize variance
x_log =np.log(x)

plt.plot(x_log)
plt.show()

x_log = list(x_log)

##### 2 processing #####

#p: autoregression order for x (sliding window size)
def lag (x,p):
    t = len(x_log)
    n = t-p
    df = n 
    Y = x_log[p:t]
    #k = p+1
    X = np.matrix([float(0) for _ in range(n*p)]).reshape(n,p)
    for i in range(n):
        for j in range(p):
            X[i,j]=x_log[i+j]
    return X,Y

X,y = lag(x=x_log,p=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,shuffle=False)

print (len(X),len(y))
print (len(X_train),len(y_train))
print (len(X_test),len(y_test) )

clf = SVR(C=100, epsilon=0.01,gamma = 0.02, kernel = 'rbf')
#clf = SVR()
clf.fit(X_train, y_train)

### 3 recursive forecast: m step ahead ###

## 3.0 just for practice, ignore this ##

past  = X_train[-1,].tolist()
past

y1=clf.predict(past)
y1

past1 = past[0][1:]
past1.append(y1[0])
past1 
past1=[past1]
past1

past1
y2=clf.predict(past1)
y2

## 3.1 define a function ##

# m: m-step ahead forecast for y
# using recursive one-step-ahead strategy ??

def m_Forecast (m,last):
    y_pred_list = []
    past = last
    for i in range(m):
        y_pred = clf.predict(past)
        y_pred_list.append(y_pred[0])
        past = past[0][1:]
        past.append(y_pred[0]) 
        past=[past]
    return y_pred_list

# metric: MAPE
def mape (y_test, y_pred):
    m = len(y_test)
    error = [np.abs(1-y_pred[i]/y_test[i]) for i in range(m)]
    mape = 100/m * sum (error)
    return mape

# 3.2 Forecast on Data

# only use the last few points in training set
last  = X_train[-1,].tolist(); print (last)

m = len(y_test)
y_pred_m = m_Forecast(m,last)

#check: first 2 consistent with practice result y1, y2
print ('\n',y_pred_m)
print ('\n',y_test)

# m-Step-Ahead
# Test Error
# MSE
test_mse = mean_squared_error(y_test, y_pred_m)
print (test_mse)
# MAE
test_mae = mean_absolute_error(y_test, y_pred_m)
print (test_mae)
# MAPE
test_mape = mape(y_test, y_pred_m)
print (test_mape)

# Training Error
y_pred_train= clf.predict(X_train)
# MSE
train_mse = mean_squared_error(y_train, y_pred_train)
print (train_mse)

# 1-Step-Ahead
# Testing Error
y_pred_test= clf.predict(X_test)
# MSE
test_mse_1 = mean_squared_error(y_test, y_pred_test)
print (test_mse_1)
test_mape_1 = mape(y_test, y_pred_test)
print (test_mape_1)

plt.plot (y_train)
plt.plot (y_pred_train)
plt.plot (y_test)
plt.plot (y_pred_test)
plt.plot (y_pred_m)
plt.show()

# Create some test data
secret_data_X1 = np.linspace(0,327,328)
secret_data_Y1 = y_pred_train
secret_data_X2 = np.linspace(327,327+37-1,37)
secret_data_Y2 = y_pred_test

secret_data_Y10 = y_train
secret_data_Y20 = y_test
secret_data_Y22 = y_pred_m

# Show the secret data
#plt.subplot(2,1,1)
plt.figure(dpi=500)
plt.plot(secret_data_X1,secret_data_Y10,'black')
plt.plot(secret_data_X2,secret_data_Y20,'black')
plt.plot(secret_data_X1,secret_data_Y1,'g')
plt.plot(secret_data_X2,secret_data_Y2,'b')
plt.plot(secret_data_X2,secret_data_Y22,'r')

plt.show()

# Create some test data

secret_data_Y1 = y_pred_train
secret_data_X2 = np.linspace(327,327+37-1,37)

secret_data_Y20 = y_test
secret_data_Y22 = y_pred_m

# Show the secret data
#plt.figure(dpi=1000,figsize=(10, 6))#

plt.rcParams["figure.figsize"] = [10,5]
plt.plot(secret_data_X2,secret_data_Y20,'black')
plt.plot(secret_data_X2,secret_data_Y2,'b')
plt.plot(secret_data_X2,secret_data_Y22,'r')

plt.ylabel('Price')
plt.xlabel('Time')

plt.savefig('svm_fore.eps', format='eps', dpi=1000) 
plt.show()



