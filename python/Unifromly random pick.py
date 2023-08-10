#Read in the data from the files.
from mnist import MNIST
import numpy as np

mndata = MNIST('..\CSE 253\HW1\mnist')

trainData, trainLabel = mndata.load_training()

testData, testLabel = mndata.load_testing()

trainData = np.asarray(trainData) / 255.0
trainLabel = np.asarray(trainLabel)
testData = np.asarray(testData) / 255.0
testLabel = np.asarray(testLabel)

# 1-NN algorithm
def NN(dataSet, labelSet, testD):
    testL = np.zeros((len(testD), 1))
    for i in range(len(testD)):
        dist = np.sum(np.power(dataSet - testD[i], 2),axis=1)
        testL[i] = labelSet[np.argmin(dist)]
        if i % (len(testD) / 10) == 0:
            print (i)
    return testL

import random
def randomPick(data, label, M):
    rand = random.sample(range(0,len(data)), M)
    trainSet = data[rand]
    labelSet = label[rand]
    return trainSet, labelSet

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
acc1000 = np.zeros(5)
for i in range(5):
    xTrainRan, yTrainRan = randomPick(trainData, trainLabel, 1000)
    neigh.fit(xTrainRan, yTrainRan)
    prediction = neigh.predict(testData)
    acc1000[i] = sum([a == b for (a, b) in zip (prediction,testLabel)]) / len(testLabel)
print (acc1000)

np.mean(acc1000)

acc5000 = np.zeros(5)
for i in range(5):
    xTrainRan, yTrainRan = randomPick(trainData, trainLabel, 5000)
    neigh.fit(xTrainRan, yTrainRan)
    prediction = neigh.predict(testData)
    acc5000[i] = sum([a == b for (a, b) in zip (prediction,testLabel)]) / len(testLabel)
print (acc5000)

np.mean(acc5000)

acc6000 = np.zeros(5)
for i in range(5):
    xTrainRan, yTrainRan = randomPick(trainData, trainLabel, 6000)
    neigh.fit(xTrainRan, yTrainRan)
    prediction = neigh.predict(testData)
    acc6000[i] = sum([a == b for (a, b) in zip (prediction,testLabel)]) / len(testLabel)
print (acc6000)

np.mean(acc6000)

