import re
import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
import scipy.stats as ss
from glob import glob
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from scipy.stats import describe

import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import sys
sys.path.append('/Users/B/Documents')

from model_helpers.model_classes import *

SEG_CHARS = ['AADT', 'SPEEDLIMIT', 'Struct_Cnd', 'Surface_Tp', 'F_F_Class']

# Read in data
data = pd.read_csv('../../data_gen/data/vz_predict_dataset.csv.gz', compression='gzip', dtype={'segment_id':'str'})
data.sort_values(['segment_id', 'week'], inplace=True)

# get segments with non-zero crashes
data_nonzero = data.set_index('segment_id').loc[data.groupby('segment_id').crash.sum()>0]
data_nonzero.reset_index(inplace=True)

# scaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_nonzero[SEG_CHARS] = scaler.fit_transform(data_nonzero[SEG_CHARS])
data_nonzero['target'] = (data['crash']>0).astype(int)
data_model = data_nonzero[['target']+SEG_CHARS].values

# split into train and test sets
train_size = int(len(data_model) * 0.67)
test_size = len(data_model) - train_size
train, test = data_model[0:train_size,:], data_model[train_size:len(data_model),:]
print(len(train), len(test))

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 1:]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 12
trainX, trainY = create_dataset(train, look_back=look_back)
testX, testY = create_dataset(test, look_back=look_back)

trainX[0]

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[2]))
testX = np.reshape(testX, (testX.shape[0], look_back, testX.shape[2]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=trainX.shape[1:]))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY,
          validation_data=(testX, testY),
          epochs=10, batch_size=1000, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

a = testY - testPredict

np.sum(testY - np.reshape(testPredict, (a.shape[0],)))

testY.shape

a = model.predict_classes(testX)



# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

#Parameters for model
#Model parameters
params = dict()

#cv parameters
cvp = dict()
cvp['pmetric'] = 'roc_auc'
cvp['iter'] = 5 #number of iterations
cvp['folds'] = 5 #folds for cv (default)

#LR parameters
mp = dict()
mp['LogisticRegression'] = dict()
mp['LogisticRegression']['penalty'] = ['l1','l2']
mp['LogisticRegression']['C'] = ss.beta(a=5,b=2) #beta distribution for selecting reg strength

#RF model parameters
mp['RandomForestClassifier'] = dict()
mp['RandomForestClassifier']['n_estimators'] = [2**8] #number of trees in the forest
mp['RandomForestClassifier']['max_features'] = ss.beta(a=5,b=2) #number of features at split
mp['RandomForestClassifier']['max_leaf_nodes'] = ss.nbinom(n=2,p=0.001,loc=100) #max number of leaves to create

# Features
features = [u'pre_week', u'pre_month', u'pre_quarter', 'avg_week', u'AADT', u'SPEEDLIMIT',
       u'Struct_Cnd', u'Surface_Tp', u'F_F_Class']

#Initialize tuner
tune = Tuner(df)

#Base RF model
tune.tune('RF_base', 'RandomForestClassifier', features, cvp, mp['RandomForestClassifier'])

#Base LR model
tune.tune('LR_base', 'LogisticRegression', features, cvp, mp['LogisticRegression'])

#Display results
tune.grid_results

# Run test
test = Tester(df)
test.init_tuned(tune)
test.run_tuned('RF_base', cal=False)

def lift_chart(x_col, y_col, data, ax=None):

    p = sns.barplot(x=x_col, y=y_col, data=data, 
                    palette='Reds', ax = None, ci=None)
    vals = p.get_yticks()
    p.set_yticklabels(['{:3.0f}%'.format(i*100) for i in vals])
    xvals = [x.get_text().split(',')[-1].strip(']') for x in p.get_xticklabels()]
    xvals = ['{:3.0f}%'.format(float(x)*100) for x in xvals]
    p.set_xticklabels(xvals)
    p.set_axis_bgcolor('white')
    p.set_xlabel('')
    p.set_ylabel('')
    p.set_title('Predicted probability vs actual percent')
    return(p)
    
def density(data, score, ax=None):
    p = sns.kdeplot(risk_df['risk_score'], ax=ax)
    p.set_axis_bgcolor('white')
    p.legend('')
    p.set_xlabel('Predicted probability of crash')
    p.set_title('KDE plot predictions')
    return(p)

risk_scores = test.rundict['RF_base']['m_fit'].predict_proba(test.data.test_x[features])[:,1]
risk_df = pd.DataFrame({'risk_score':risk_scores, 'crash':test.data.test_y})
print risk_df.risk_score.describe()
risk_df['categories'] = pd.cut(risk_df['risk_score'], bins=[-1, 0, .01, .02, .05, max(risk_scores)])
risk_mean = risk_df.groupby('categories')['crash'].count()
print risk_mean

fig, axes = plt.subplots(1, 2)
lift_chart('categories', 'crash', risk_df, 
           ax=axes[1])
density(risk_df, 'risk_score', ax=axes[0])

# output predictions
# predict on all segments
data_model['risk_score'] = test.rundict['RF_base']['m_fit'].predict_proba(data_model[features])[:,1]
data_model.to_csv('seg_with_risk_score.csv', index=False)

for w in [20, 30, 40, 50]:
    print "week ", w
    crash_lags = format_crash_data(data_nonzero.set_index(['segment_id','week']), 'crash', w)
    data_model = crash_lags.merge(data_segs, left_on='segment_id', right_on='segment_id')
    df = Indata(data_model, 'target')
    # create train/test split
    df.tr_te_split(.7)
    test = Tester(df)
    test.init_tuned(tune)
    test.run_tuned('RF_base', cal=False)
    print '\n'

