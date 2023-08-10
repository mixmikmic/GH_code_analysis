#Importing libraries

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.cross_validation import train_test_split

#Loading the labelled datasets

train = pd.read_csv('Data2/Data2_train.csv')
validation = pd.read_csv('Data2/Data2_validation.csv')
train.head()

#Concatenating the 2 sets to form ONE TRAINING SET
trainCombined = pd.concat([train, validation], axis=0)

# Splitting train and test to take the first 10 features only
trainSet = trainCombined.iloc[:,:10]

#Not required, just getting the names

train.scenes.unique()

# Getting the corresponding Y scenes(text)

Y_labels = trainCombined.scenes
Y_labels[:15]

#The function that assigns numbers to our categories

def numericLabels(x):
     return {
        ourLabels[0]: 1,
        ourLabels[1]: 2,
        ourLabels[2]: 3,
        ourLabels[3]: 4,
        ourLabels[4]: 5,
        ourLabels[5]: 5,
        'unknown': 6,
    }[x]

#The function that assigns numerical values to our labels
ourLabels = ['tubestation', 'quietstreet', 'busystreet', 'restaurant', 'market', 'openairmarket']

def manageLabels(labelsText, labelsNum):
    i = 0;
    while i < labelsText.size:
        if labelsText[i] not in ourLabels:
            labelsText.replace(labelsText[i],'unknown',inplace=True)
        labelsNum[i] = numericLabels(labelsText[i])
        i += 1

validationSize = [0.30, 0.28, 0.26, 0.24, 0.22, 0.20]
softMargin = [1.5, 2.5]

for i in validationSize:
    #splitting the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(trainSet, Y_labels, test_size=i, random_state=2891)
    
    #resetting indices for the labelled sets so that they work with the pre written functions
    Y_train.reset_index(drop=True, inplace=True)
    Y_test.reset_index(drop=True, inplace=True)
    
    
    #Converting the labels to numerical values
    Y_train1 = Y_train
    Y_test1 = Y_test
    manageLabels(Y_train, Y_train1)
    manageLabels(Y_test, Y_test1)
    
    #converting type of Y to int
    Y_train1 = Y_train1.astype('int64')
    Y_test1 = Y_test1.astype('int64')
    
    #Readability shenanigans
    testing = 100 * i
    training = 100 - testing
    print '\n\n **** For %d/%d data split ratio **** : \n' %(training, testing)
    
    #Train the model (Poly with degree=3)
    for kernel in ('linear', 'poly', 'rbf'):
        clf = svm.SVC(kernel=kernel, C=2.5, degree=3)
        clf.fit(X_train, Y_train1)
        print "We successfully predict {0}% of data using {1} kernel for the training data".format(100-abs(clf.predict(X_train)-Y_train1).sum()/len(Y_train1), kernel)
    
    #Fit the model (Poly with degree=3,, C=1.5 and 2.5)
    for c in softMargin:
        print '\n With C = %.1f' %c   
        for kernel in ('linear', 'poly', 'rbf'):
            clf = svm.SVC(kernel=kernel, C=c, degree=3)
            clf.fit(X_test, Y_test1)
            correct=1.0*(clf.predict(X_test)==np.asarray(Y_test1)).sum()/len(Y_test1)
            print "We successfully predict {0}% of data using {1} kernel for the test data".format((correct)*100, kernel)

test = pd.read_csv('Data2/Data2_test.csv')

testSet = test.iloc[:,:10]

test.scenes.unique()

Y_labels = test.scenes
Y_labels[:15]

Y_labels.reset_index(drop=True, inplace=True)

#Converting the labels to numerical values
Y_test = Y_labels
manageLabels(Y_test, Y_labels)

#converting type of Y to int
Y_test = Y_test.astype('int64')

clf = svm.SVC(kernel='poly', C=2.5, degree=3)
clf.fit(X_train, Y_train1)
correct=1.0*(clf.predict(testSet)==np.asarray(Y_test)).sum()/len(Y_test)
print "We successfully predict {0}% of data using {1} kernel for the test data".format((correct)*100, 'poly')

