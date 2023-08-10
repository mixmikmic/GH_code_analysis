"""
Author: Aditya Kotak
For the UnifyID ML Challenge
Implements a kNN classification to make predictions for which activity is being done.
Data from: http://ps.ewi.utwente.nl/Datasets.php
"""
import csv, random, math, operator
trainSet, testSet, predictions = [], [], []

k, split = 3, 0.7
wristFileName, beltFileName, pocketFileName, armFileName = "wrist.data.csv", "belt.data.csv", "pocket.data.csv", "arm.data.csv"

"""Functionality to load a CSV into an list of lists."""
def load(myfile, splitPercent, train=[] , test=[]):
    with open(myfile, 'r') as mycsv:
        iterator = csv.reader(mycsv)
        next(iterator, None)
        mydata = list(iterator)
        for i in range(len(mydata)-1):
            for j in range(0,10):
                mydata[i][j] = float(mydata[i][j])
            if random.random() < splitPercent:
                train.append(mydata[i])
            else:
                test.append(mydata[i])

"""Basis of the kNN algorithm. Calculates the 'distance' between two datapoints."""
def calcDist(dataSet1, dataSet2, mylen):
    dist = 0
    for i in range(mylen):
        dist += pow((dataSet1[i] - dataSet2[i]), 2)
    return math.sqrt(dist)

"""Finds the k nearest neighbors by finding the k elements with the shortest distance."""
def findNeighbors(trainSet, test, k):
    distances = []
    testLen = len(test)-1
    for i in range(len(trainSet)):
        dist = calcDist(test, trainSet[i], testLen)
        distances.append((trainSet[i], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

"""Makes the prediction for the activity by using the k nearest neighbors"""
def predict(neighbors):
    rawPredictions = {}
    for i in range(len(neighbors)):
        prediction = neighbors[i][-1]
        if prediction in rawPredictions:
            rawPredictions[prediction] += 1
        else:
            rawPredictions[prediction] = 1
    sortedPredictions = sorted(rawPredictions.items(), key=operator.itemgetter(1), reverse=True)
    return sortedPredictions[0][0]

"""Calculates how accurate the prediction is by cross referencing the actual test data set"""
def calcAccuracy(test, predicts, nums):
    accurate = 0
    for i in range(nums):
        if predicts[i] == test[i][-1]:
            accurate += 1
    return (accurate/float(nums)) * 100.0


"""Runs the actual predictions with a given dataset fileName"""
def run(fileName):
    load(fileName, split, trainSet, testSet)
    predictionNums = len(testSet)

    for i in range(predictionNums):
        neighbors = findNeighbors(trainSet, testSet[i], k)
        output = predict(neighbors)
        predictions.append(output)
        print("predicted = " + repr(output) + ", actual = " + repr(testSet[i][-1]))
    accuracy = calcAccuracy(testSet, predictions, predictionNums)
    print('Accuracy: ' + repr(accuracy) + '%')

"""Runs the wrist data"""
def runWrist():
    run(wristFileName)

"""Runs the belt data"""
def runBelt():
    run(beltFileName)

"""Runs the pocket data"""
def runPocket():
    run(pocketFileName)

"""Runs the arm data"""
def runArm():
    run(armFileName)

runBelt() # runs analysis on the Belt data. Change this command to work with the other datasets.



