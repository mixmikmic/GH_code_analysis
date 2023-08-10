get_ipython().magic('matplotlib inline')
import sys
import os
import time

import pandas as pd
import numpy as np

import CBECSLib

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')

#sklearn base
import sklearn.base

#sklearn utility
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

#IPython utilities
from IPython.display import HTML,display
def show(html):
    display(HTML(html))

RESULTS_DIR = "results/" # where output is written to
DATASET = "VAL" # for the extended set of features use 0, for the common set of features use 1

def load_NYC():
    X_val = np.load("output/nyc/ll84_X_2016.npy")
    Y_val = np.load("output/nyc/ll84_Y_2016.npy")
    valClassVals = np.load("output/nyc/ll84_classVals_2016.npy")
    return X_val, Y_val, valClassVals

pbaLabels = CBECSLib.pbaLabels
pbaPlusLabels = CBECSLib.pbaPlusLabels

getClassFrequencies = CBECSLib.getClassFrequencies
getDataSubset = CBECSLib.getDataSubset

regressors = CBECSLib.regressors
regressorNames = CBECSLib.regressorNames
numRegressors = CBECSLib.numRegressors

metrics = CBECSLib.metrics
metricNames = CBECSLib.metricNames
numMetrics = CBECSLib.numMetrics

X,Y,classVals = load_NYC()
classOrdering,classFrequencies = getClassFrequencies(classVals)
numClassVals = len(classFrequencies)

numSplits = 3
numRepeats = 10
outputFn = "test_all_VAL"

results = np.zeros((numRepeats, numSplits, numRegressors, numMetrics), dtype=float)

for i in range(numRepeats):
    print "Repetition %d" % (i)

    kf = StratifiedKFold(n_splits=numSplits)
    for j, (train, test) in enumerate(kf.split(X,classVals)):
        #print "\tSplit %d" % (j)
        X_train, X_test = X[train,:], X[test,:]
        Y_train, Y_test = Y[train], Y[test]
        classVals_train, classVals_test = classVals[train].copy(), classVals[test].copy()

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        for k in range(numRegressors):
            regressor = sklearn.base.clone(regressors[k])
            regressorName = regressorNames[k]

            #train model
            regressor.fit(X_train,Y_train)

            #predict model
            predicted = regressor.predict(X_test)
            predicted[predicted<0] = 0

            #evaluate model
            for m,metric in enumerate(metrics):
                metricName = metricNames[m]
                score = metric(Y_test,predicted)
                results[i,j,k,m] = score
        
results = np.array(results)

results = results.reshape(-1, numRegressors, numMetrics)

classNames = [pbaLabels[pbaLabel] for pbaLabel in classOrdering]

meanResults = results.mean(axis=0)
meanResultTable = pd.DataFrame(meanResults, index=regressorNames, columns=metricNames)
meanResultTable.to_csv(os.path.join(RESULTS_DIR, "%s_means.csv" % (outputFn)))

stdResults = results.std(axis=0)
stdResultTable = pd.DataFrame(stdResults, index=regressorNames, columns=metricNames)
stdResultTable.to_csv(os.path.join(RESULTS_DIR, "%s_stds.csv" % (outputFn)))

formattedResults = []
for i in range(numRegressors):
    row = []
    for j in range(numMetrics):
        row.append("%0.2f +/- %0.2f" % (meanResults[i,j], stdResults[i,j]))
    formattedResults.append(row)
formattedResults = np.array(formattedResults)
formattedResultsTable = pd.DataFrame(formattedResults, index=regressorNames, columns=metricNames)
formattedResultsTable.to_csv(os.path.join(RESULTS_DIR, "%s_formatted.csv" % (outputFn)))

display(formattedResultsTable)



