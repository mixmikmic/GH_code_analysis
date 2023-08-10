#Implements functional expansions
from functions.FE import FE
#Evaluates accuracy in a dataset for a particular classifier
from fitness import Classifier
#Implements gafe using DEAP toolbox
import ga

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

iris = pd.read_csv("data/iris.data", sep=",")
#Isolate the attributes columns
irisAtts = iris.drop("class", 1)
#Isolate the class column
target = iris["class"]

scaledIris = MinMaxScaler().fit_transform(irisAtts)

bestSingleMatch = {'knn': [(1,5) for x in range(4)], 'cart': [(3,2) for x in range(4)], 'svm': [(7,4) for x in range(4)]}

functionalExp = FE()

for cl in ['knn', 'cart', 'svm']:
        #Folds are the number of folds used in crossvalidation
        #Jobs are the number of CPUS used in crossvalidation and some classifiers training step.
        #You can also change some classifier parameters, such as k_neigh for neighbors in knn, C in svm and others.
        #If you do not specify, it will use the articles default.        
        model = Classifier(cl, target, folds=10, jobs=6)
        #The class internally normalizes data, so no need to send normalized data when classifying 
        #accuracy without expanding
        print("original accuracy " + cl + " " + str(model.getAccuracy(irisAtts)))
        #Expand the scaled data 
        expandedData = functionalExp.expandMatrix(scaledIris, bestSingleMatch[cl])
        print("single match expansion accuracy " + cl + " " + str(model.getAccuracy(expandedData)))
        #If scaled is False, it will scale data in range [0,1]
        gafe = ga.GAFE(model, scaledIris, target, scaled=True)
        #Specify how many iterations of GAFE you wish with n_iter
        #Note that this is a slow method, so have patience if n_iter is high        
        avg, bestPair = gafe.runGAFE(n_population=21, n_iter=1, verbose=True)
        print("gafe " + cl + " " + str(avg) )

