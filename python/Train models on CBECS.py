get_ipython().magic('matplotlib inline')
import sys
import os
import time

import pandas as pd
import numpy as np

import cPickle as pickle

import CBECSLib

#sklearn base
import sklearn.base

#sklearn utility
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = "output/trainedModels/"

pbaLabels = CBECSLib.pbaLabels
pbaPlusLabels = CBECSLib.pbaPlusLabels

getDataset = CBECSLib.getDataset
getClassFrequencies = CBECSLib.getClassFrequencies
getDataSubset = CBECSLib.getDataSubset

regressors = CBECSLib.regressors
regressorNames = CBECSLib.regressorNames
numRegressors = CBECSLib.numRegressors

metrics = CBECSLib.metrics
metricNames = CBECSLib.metricNames
numMetrics = CBECSLib.numMetrics

X,Y,columnNames,classVals = getDataset(1,pbaOneHot=True)
print columnNames
classOrdering,classFrequencies = getClassFrequencies(classVals)
numClassVals = len(classFrequencies)
Y = np.log10(Y)

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
pickle.dump(scaler, open("output/scaler.p", "wb"))

for i in range(numRegressors):
    regressor = sklearn.base.clone(regressors[i])
    regressorName = regressorNames[i]

    print regressorName
    
    #train model
    regressor.fit(X_scaled,Y)

    #predict model
    predicted = regressor.predict(X_scaled)
    predicted[predicted<0] = 0

    #evaluate model
    scores = []
    for m,metric in enumerate(metrics):
        metricName = metricNames[m]
        score = metric(Y,predicted)
        scores.append(score)
    print scores
        
    pickle.dump(regressor, open(os.path.join(OUTPUT_DIR, "%s_trained.p" % (regressorName)), "wb"))



