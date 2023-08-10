import pandas as pd
import numpy as np
# moving average in pandas http://chrisalbon.com/python/pandas_moving_average.html

# Problem Defination : 

#Our Aim is to predict if 'Future Daily-Returns' of S&P 500 are going to be +ve or -ve. This is
#binary classification.

# What is S&P 500? - Backtesting - Benchmark
# S&p is market index which contains list of top 500 companies having stocks in NYSE or or NASDAQ
# What is Binary Classification?

# Daily Return: 

# it is just the way that how we want to look for Value in the system. We are using following formula:

#Returni=AdjClosei–AdjClosei−1/AdjClosei−1

#Return at ith day is equal to return adjacent-close at ith day---adjacentclose at i-1th day // adjacentclose at i-1th day

# in other words it's just percent-change/ rate of change w.r.t to time-- What that is called in physics? :D

## What is Adjacent Close Price?
## Close price of the day modified by taking account into dividends. 
## What are dividends?
## How Adjacent-close price is calculated? 


df = pd.read_csv('WIKI-AAPL.csv')
df.columns.values[-2]

df['returns']=df['Adj. Close'].pct_change()

#pd.rolling_mean(df['returns'],4)

df['rolling'] = df['returns'].rolling(window = 4,center=False).mean()

#pd.rolling_mean(df['returns'],0) # what is the meaning o rolling mean?
df[['returns','rolling']].head()
#Multiple Day Returns: percentage difference of Adjusted Close Price of i-th day  compared to (i-delta)-th day. 
#Example: 3-days Return is the percentage difference of Adjusted Close Price of today compared to the one of 3 days ago
df['returns']=df['Adj. Close'].pct_change()
df['multiple_day']=df['Adj. Close'].pct_change(3)
# we are shifting for 2 so we are calculating for last 'two' days
df['time_lagged'] = df['Adj. Close']-df['Adj. Close'].shift(-2)

new_df = pd.DataFrame(columns=[df['Adj. Close'],df['returns'],df['rolling'],df['time_lagged'],df['multiple_day']])
new_df # we need horizontly , columns are only for names

#dataset.UpDown[dataset.UpDown >= 0] = 'Up'
df['updown'] = df['returns']
df.updown[df['returns']>=0]='up'
df.updown[df['returns']<0]='down'
seris_dict ={'close_value':df['Adj. Close'],'daily-returns':df['returns'],'rolling':df['rolling'],'time-lagged':df['time_lagged'],'multiple_day':df['multiple_day'],'updown':df['updown']}
new_df = pd.DataFrame(seris_dict)
new_df.head()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df['encoding']=le.fit(df['updown']).transform(df['updown']) # what is label encoding, ? also read more about preprocessing

seris_dict ={'close_value':df['Adj. Close'],'daily-returns':df['returns'],'rolling':df['rolling'],'time-lagged':df['time_lagged'],'multiple_day':df['multiple_day'],'updown':df['updown'],'encoding':df['encoding']}
new_df = pd.DataFrame(seris_dict)
new_df.head()

seris_dict ={'close_value':df['Adj. Close'],'daily-returns':df['returns'],'rolling':df['rolling'],'time-lagged':df['time_lagged'],'multiple_day':df['multiple_day']}
new_df = pd.DataFrame(seris_dict)
new_df.head()

features = new_df.columns

type(features)

features

new_df[features].head()

df.updown.head()

X = new_df[features].dropna()
Y =  df.updown[6:]
#new_df[features].dropna()

#dir(sklearn.model_selection)
from sklearn.model_selection import train_test_split,TimeSeriesSplit
# TimeSeries Split Scikit Learn
# What is the difference between tain_test_split and cross validation?

X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
#X_train,X_test,Y_train,Y_test = TimeSeriesSplit(X) - Why it is not working?

X_train.head()

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train,Y_train) # how to remove Nan's? from Pandas or somewhereelse use dropna() method or finance.fillna(finance.mean())

prediction.predict(X_test)

accuracy = rf.score(X_test,Y_test)

print accuracy

from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier()

ada.fit(X_train,Y_train)

ada.score(X_test,Y_test)

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

gbc.fit(X_train,Y_train)

gbc.score(X_test,Y_test)

from sklearn.svm import SVC

clf = SVC()

clf.fit(X_train,Y_train)

clf.score(X_test,Y_test)

from sklearn.qda import QDA

des_a = QDA()

des_a.fit(X_train,Y_train)

des_a.score(X_test,Y_test)

get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt

des_a.predict(X_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30),max_iter=200,validation_fraction=0.9)

mlp.fit(X_train,Y_train)

mlp.score(X_test,Y_test)



