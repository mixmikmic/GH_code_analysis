import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
import datetime as dt
from sklearn.preprocessing import StandardScaler
import time

pay = pd.read_csv('../preprocess/payTH_parallel.txt', sep=" ", header = None)
trainFile = '../preprocess/trainValidFeatures_ensemble.csv'
testFile = '../preprocess/validFeatures_ensemble.csv'

trainLabel = pay[np.arange(447, 475)].values.reshape(1, -1)[0]
testLabel = pay[np.arange(475, 489)].values.reshape(1, -1)[0]

trainData = pd.read_csv(trainFile, header = None)
testData = pd.read_csv(testFile, header = None)

def detectNaN(a):
    for i in range(len(a[0])):
        e = True
        for j in range(len(a) - 1):
            if np.isnan(a[j][i]):
                e = False
                break
        if (not e):
            print(i)

def replace(a):
    for i in range(len(a[0])):
        e = True
        for j in range(len(a)):
            if np.isnan(a[j][i]):
                a[j][i] = a[j - 1][i]
    return a

trainDataArray = np.array(trainData)
trainDataArrayProcessed = np.delete(trainDataArray, [1, 2], 1)
trainDataProcessed = replace(trainDataArrayProcessed)
detectNaN(trainDataProcessed)
scaler = StandardScaler()
scaler.fit(trainDataProcessed)
trainDataNormalized = scaler.transform(trainDataProcessed)
detectNaN(trainDataNormalized)

testDataArray = np.array(testData)
testDataArrayProcessed = np.delete(testDataArray, [1, 2], 1)
testDataProcessed = replace(testDataArrayProcessed)
detectNaN(testDataProcessed)
scaler = StandardScaler()
scaler.fit(testDataProcessed)
testDataNormalized = scaler.transform(testDataProcessed)
detectNaN(testDataNormalized)

alphaList = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
iterationList = [500, 750, 1000, 1250, 1500, 1750, 2000]
minLoss = 10
bestAlpha = 0
bestIteration = 0
timeUsed = []
for alpha in alphaList:
    for iteration in iterationList:
        reg = MLPRegressor(hidden_layer_sizes=(100, 1000, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,100,100,100,100,100,100,100,100,100,100,1000,100,100), early_stopping = True,  alpha = alpha, learning_rate = 'adaptive', max_iter = iteration)
        t_start = time.time()
        reg.fit(trainDataNormalized, trainLabel)
        predictedLabel= reg.predict(testDataNormalized)
        t_finish = time.time()
        timeUsed.append(t_finish - t_start)
        result = sum(abs((abs(predictedLabel) - testLabel) / (abs(predictedLabel) + testLabel)) / (14 * 2000))
        if (result < minLoss):
            minLoss = result
            bestAlpha = alpha
            bestIteration = iteration     

