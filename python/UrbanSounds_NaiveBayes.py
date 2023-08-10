# Base
import numpy as np
import pandas as pd
import json
import re
import string
from os import listdir
import math
import time
import csv
import sys
import datetime

# Plotting
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Machine Learning
from sklearn import metrics, cross_validation

from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder

from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

#from sknn.mlp import Regressor, Classifier, Layer
#from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Import Pre-Processed Wav File Data Set
wavData = pd.read_csv('feature_quant.csv')

wavData[0:5]

# Remove Empty Rows
wavData = wavData[-np.isnan(wavData['mean'])]

feat = list(wavData.columns)
feat.remove('class')
feat.remove('Unnamed: 0')
feat

X_train, X_test, y_train, y_test = train_test_split(wavData.loc[:,feat], wavData.loc[:,'class'],                                                     test_size=0.3, random_state=0)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_predict = gnb.predict(X_test)

classes = set(wavData['class']); classes

y_probs = gnb.predict_proba(X_test)
list(zip(y_probs[1], gnb.classes_))

y_probs[0]

y_logprobs = gnb.predict_log_proba(X_test)
zip(y_logprobs[1],gnb.classes_)

y_predict[1]

print('Accuracy: %.2f' % accuracy_score(y_test,y_predict))
#print('Precision: %.2f' % precision_score(y_test,y_predict))
#print('Recall: %.2f' % recall_score(y_test,y_predict))
#print('F1: %.2f' % f1_score(y_test,y_predict))
confmat=confusion_matrix(y_true=y_test, y_pred=y_predict)
print(confmat)

# Set up loop to get Accuracy for each class as 1-vs-All
def runNBonevsall(var, DF, featList):
    # Create new response variable
    DF[var] = 0
    DF.loc[DF['class'] == var,var] = 1
    #feat = list(DF.columns)
    #print(feat)
    #feat.remove('class')
    #feat.remove('Unnamed: 0')
    #feat.remove(var)
    #print(feat)
    X_train, X_test, y_train, y_test = train_test_split(DF.loc[:,featList], DF.loc[:,var],                                                         test_size=0.35, random_state=0)
    print(var)
    
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_predict = gnb.predict(X_test)
    
    print('Accuracy: %.2f' % accuracy_score(y_test,y_predict))
    print('Precision: %.2f' % precision_score(y_test,y_predict))
    print('Recall: %.2f' % recall_score(y_test,y_predict))
    print('F1: %.2f' % f1_score(y_test,y_predict))
    
    confmat=confusion_matrix(y_true=y_test, y_pred=y_predict, labels = [1,0])
    print(confmat)
    print('\n')
    
    return {var:(accuracy_score(y_test,y_predict),precision_score(y_test,y_predict),                  recall_score(y_test,y_predict), f1_score(y_test,y_predict))}

resultsNB = [runNBonevsall(var, wavData, feat) for var in classes]

# Set up loop to get Accuracy for each class as 1-vs-All
def runNBstacked(var, X_train, X_test, y_train, y_test):
    y_train = y_train[var]
    y_test = y_test[var] 
    
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_predict = gnb.predict(X_test)
    y_probs = gnb.predict_proba(X_test)
    
    probs1 = list(zip(*y_probs))[1]
    print(gnb.classes_)
    
    #print('Accuracy: %.2f' % accuracy_score(y_test,y_predict))
    #print('Precision: %.2f' % precision_score(y_test,y_predict))
    #print('Recall: %.2f' % recall_score(y_test,y_predict))
    #print('F1: %.2f' % f1_score(y_test,y_predict))
    
    #confmat=confusion_matrix(y_true=y_test, y_pred=y_predict, labels = [1,0])
    #print(confmat)
    #print('\n')
    
    #return {var:(accuracy_score(y_test,y_predict),precision_score(y_test,y_predict), \
    #             recall_score(y_test,y_predict), f1_score(y_test,y_predict))}
    return probs1

for var in classes:
    wavData[var] = 0
    wavData.loc[wavData['class'] == var,var] = 1

X_train, X_test, y_train, y_test = train_test_split(wavData.loc[:,feat], wavData.loc[:,list(classes)],                                                         test_size=0.3, random_state=0)

y_train[0:5]

probsNB = {}
for var in classes:
    probsNB[var] = runNBstacked(var, X_train, X_test, y_train, y_test)

probsNB_DF = pd.DataFrame(probsNB)

probsNB_DF[0:5]

probsNB_DF['response'] = probsNB_DF.idxmax(axis=1)

y_predict = probsNB_DF['response']

y_predict[0:5]

y_test = y_test.idxmax(axis=1)

y_test[0:5]

print('Accuracy: %.2f' % accuracy_score(y_test,y_predict))
print('Precision: %.2f' % precision_score(y_test,y_predict))
print('Recall: %.2f' % recall_score(y_test,y_predict))
print('F1: %.2f' % f1_score(y_test,y_predict))



