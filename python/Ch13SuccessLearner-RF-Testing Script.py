#Copyright April 1. 2018, Warren E. Agin
#Code released under the Creative Commons Attribution-NonCommercial-
#ShareAlike 4.0 International License. You may obtain a copy of the license at 
#https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

DATA_URL = ''
FEATURE_NAMES = 'featureNames.csv'
TRAINING_FILE = 'trainingFile.csv'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)

#DELETED_FEATURES allows removal of selected features from the training set
DELETED_FEATURES = [
    'REALPROPVALUESQR',
    'REALPROPVALUELOG',
    'PERSPROPVALUESQR',
    'PERSPROPVALUELOG',
    'UNSECNPRVALUESQR',
    'UNSECNPRVALUELOG',
    'UNSECPRVALUESQR',
    'UNSECPRVALUELOG',
    'AVGMNTHIVALUESQR',
    'AVGMNTHIVALUELOG',
    'NTRDBT',
#    'JOINT',
#    'ORGD1FPRSE',
#    'PRFILE',
#    'DISTSUCCESS',
    'FEEP',
#    'FEEI',
    'FEEW',
    'REALPROPNULL',
#    'REALPROPNONE',
#    'REALPROPVALUE',
    'PERSPROPNULL',
#    'PERSPROPVALUE',
#    'UNSECNPRNULL',
#    'UNSECNPRVALUE',
    'UNSECEXCESS',
    'UNSECPRNULL',
#    'UNSECPRVALUE',
    'AVGMNTHINULL',
#    'AVGMNTHIVALUE',
#    'IEINDEX',
#    'IEGAP'
]

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import csv
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle

#identify features from data sets
features = pd.read_csv(FEATURE_NAMES)
featureNames = list(features.columns)

#read training set into panda arrays
training = pd.read_csv(TRAINING_FILE, names=featureNames)

#convert the panda arrays to numpy arrays for use with the learner
#creates training as a numpy array, trainingLabels as a numpy
#array holding the success field, and featureNames as a list of the features

def removeFeatures(file):
    for each in DELETED_FEATURES:
        file=file.drop(each, axis=1)
    return(file)

def convert2np(file):   
    labels = np.array(file['SUCCESS'])   #copy out the success column as a numpy array
    file = file.drop('SUCCESS', axis=1)  #remove the success column
    file = removeFeatures(file)          #remove the features not being used
    file = np.array(file)                #convert the data to a numpy array
    return(file,labels)
    
training,trainingLabels = convert2np(training)

finalFeatureNames = featureNames
for each in DELETED_FEATURES:
    finalFeatureNames.remove(each)
finalFeatureNames.remove('SUCCESS')

#create, run and evaluate the model

#set logging file and add a list of the features being used
log = 'Features used: ' + str(finalFeatureNames) + '\r\n'

#function to calculate metrics - calculates accuracy, AUC and a confusion matrix
def calcResults(set, labels):
    # Use the predict method on the training and test data
    predictions = rf.predict(set)
    # Calculate the number of errors
    numberErrors = sum(abs(predictions - labels))

    # Calculate and display accuracy and other statistics
    accuracy = (1-(numberErrors/len(set)))*100
    aucResult = metrics.roc_auc_score(labels, predictions)
    cMatrix = metrics.confusion_matrix(labels, predictions)

    return(accuracy, aucResult, cMatrix)


#define characteristics for the learner
n_estimators = 1000
max_features = 'auto' #default is 'auto' which considers sqrt(n_features) at each split - alt are 1 to consider n_features or a decimal
max_depth = 20    #default is 'None'
min_samples_split = 150 #default is 2
min_samples_leaf = 1  #default is 1
random_state = 26 #default is 'None' but use a number for testing variations

log += ' n_estimators: %s \r\n max_features: %s \r\n max_depth: %s \r\n min_samples_split: %s \r\n min_samples_leaf: %s \r\n' % (n_estimators,max_features,max_depth,min_samples_split,min_samples_leaf)

# Instantiate model with n_estimators decision trees
rf = sk.ensemble.RandomForestClassifier(criterion = 'gini', n_estimators = n_estimators, max_features = max_features, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, random_state = random_state)

# criterion = 'gini' or 'entropy'

# Train the model on training data
rf.fit(training, trainingLabels);

#run predictions on the training set and the test set and calculate metrics

accuracy, aucResult, cMatrix = calcResults(training, trainingLabels)

log += 'Train Set Accuracy: %s \r\n' %  round(accuracy, 2)
log += 'Train set AUC: %s  \r\n' % round(aucResult, 4)

# Print out the confusion matrix.
print('accuracy: %s' % round(accuracy, 2))


# Get numerical feature importances 
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 6)) for feature, importance in zip(finalFeatureNames, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Log the features and importances 

for pair in feature_importances:
    log += 'Variable: {:20} Importance: {}\r\n'.format(*pair)
log += '\r\n'

#run the model against the three test sets

#define the sources for the test sets
EVAL_FILE1 = 'test1File.csv'
EVAL_FILE2 = 'test2File.csv'
EVAL_FILE3 = 'test3File.csv'

def evaluateModel(set):
    
    results=''
    
    #load the test file into panda array and convert into x and label numpy arrays
    testing = pd.read_csv(set, names=featureNames)
    testing,testingLabels = convert2np(testing)

    #run the model on the test set and return accuracy, AUC, and confusion matrix
    accuracy, aucResult, cMatrix = calcResults(testing, testingLabels)

    #log the results
    results += 'Test Set Accuracy: %s \r\n' %  round(accuracy, 2)
    results += 'Test set AUC: %s  \r\n' % round(aucResult, 4)
    results += str(cMatrix)
    results += '\r\n\r\n'
    print('Test Set Evaluated')
    return(results)

log += evaluateModel(EVAL_FILE1)
log += evaluateModel(EVAL_FILE2)
log += evaluateModel(EVAL_FILE3)

#print the log file for review   
print(log)

#write the log file to LOG.txt
with open('LOG.txt', 'w', newline='') as f:
    f.write(log)
    

#save the model for deployment

filename = 'finalRFModel.sav'
pickle.dump(rf, open(filename,'wb'))

#run probabilities and log out for analysis in Excel

#load the test file into panda array and convert into x and label numpy arrays
testingProb = pd.read_csv(EVAL_FILE1, names=featureNames)
testingProb,testingLabels = convert2np(testingProb)

#compute the failure, success probabilities for each in testingProb
probabilities = rf.predict_proba(testingProb)

#write probabilities and actual result to a csv
with open('ProbLog.csv', 'w', newline='') as f:
    writer=csv.writer(f, delimiter=',')
    for key in range(0,len(probabilities)):
        writer.writerow([probabilities[key][0],probabilities[key][1],testingLabels[key]])

#sample code for reloading the saved model
testing = '' #name of the data file with data for prediction
rfLoaded = pickle.load(open('finalRFModel.sav', 'rb'))
rfLoaded.predict_proba(testing)

