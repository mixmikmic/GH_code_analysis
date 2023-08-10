import datetime
import gc
import numpy as np
import os
import random
from scipy import misc
import string
import time
import sys
import sklearn.metrics as skm
import collections
from sklearn.svm import SVC
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn import metrics
import dwdii_bc_model_helper as bc

random.seed(20275)
np.set_printoptions(precision=2)

imagePath = "png"
trainImagePath = imagePath
trainDataPath = "data/ddsm_train.csv"
testDataPath = "data/ddsm_test.csv"
categories = bc.bcNumerics()
imgResize = (150, 150)
normalVsAbnormal=False

os.listdir('data')

metaData, meta2, mCounts = bc.load_training_metadata(trainDataPath, balanceViaRemoval=True, verbose=True, 
                                                     normalVsAbnormal=normalVsAbnormal)

thesePathos = ['benign','malignant']

# Actually load some representative data for model experimentation
maxData = len(metaData)
X_data, Y_data = bc.load_data(trainDataPath, trainImagePath, 
                              categories=categories,
                              maxData = maxData, 
                              verboseFreq = 50, 
                              imgResize=imgResize, 
                              thesePathos=thesePathos,
                              normalVsAbnormal=normalVsAbnormal)
print X_data.shape
print Y_data.shape

# Actually load some representative data for model experimentation
maxData = len(metaData)
X_test, Y_test = bc.load_data(testDataPath, imagePath, 
                              categories=categories,
                              maxData = maxData, 
                              verboseFreq = 50, 
                              imgResize=imgResize, 
                              thesePathos=thesePathos,
                              normalVsAbnormal=normalVsAbnormal)
print X_test.shape
print Y_test.shape

X_train = X_data
Y_train = Y_data

print X_train.shape
print X_test.shape

print Y_train.shape
print Y_test.shape

def yDist(y):
    bcCounts = collections.defaultdict(int)
    for a in range(0, y.shape[0]):
        bcCounts[y[a][0]] += 1
    return bcCounts

print "Y_train Dist: " + str(yDist(Y_train))
print "Y_test Dist: " + str(yDist(Y_test))

X_train_s = X_train.reshape((1270,-1))

X_test_s = X_test.reshape((321,-1))

Y_train_s = Y_train.ravel()

model = SVC(C=1.0, gamma=0.001, kernel='rbf')

model.fit(X_train_s,Y_train_s)

predicted = model.predict(X_test_s)
expected = Y_test

svm_matrix = skm.confusion_matrix(Y_test, predicted)
svm_matrix

print metrics.accuracy_score(expected,predicted)

numBC = bc.reverseDict(categories)
class_names = numBC.values()
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
bc.plot_confusion_matrix(svm_matrix, classes=class_names[1:3],
                      title='Confusion Matrix without normalization')
plt.savefig('raw_class2_bVm_o_norm.png')

from IPython.display import Image
Image(filename='raw_class2_bVm_o_norm.png')

plt.figure()
bc.plot_confusion_matrix(svm_matrix, classes=class_names[1:3], normalize=True,
                      title='Confusion Matrix with normalization')
plt.savefig('raw_class2_bVm_norm.png')

# Load the image we just saved
from IPython.display import Image
Image(filename='raw_class2_bVm_norm.png')

imagePath = "DDSM_threshold"
trainImagePath = imagePath
trainDataPath = "data/ddsm_train.csv"
testDataPath = "data/ddsm_test.csv"
categories = bc.bcNumerics()
imgResize = (150, 150)
normalVsAbnormal=False

os.listdir('data')

metaData, meta2, mCounts = bc.load_training_metadata(trainDataPath, balanceViaRemoval=True, verbose=True, 
                                                     normalVsAbnormal=normalVsAbnormal)

# Actually load some representative data for model experimentation
maxData = len(metaData)
X_data, Y_data = bc.load_data(trainDataPath, trainImagePath, 
                              categories=categories,
                              maxData = maxData, 
                              verboseFreq = 50, 
                              imgResize=imgResize, 
                              thesePathos=thesePathos,
                              normalVsAbnormal=normalVsAbnormal)
print X_data.shape
print Y_data.shape

# Actually load some representative data for model experimentation
maxData = len(metaData)
X_test, Y_test = bc.load_data(testDataPath, imagePath, 
                              categories=categories,
                              maxData = maxData, 
                              verboseFreq = 50, 
                              imgResize=imgResize, 
                              thesePathos=thesePathos,
                              normalVsAbnormal=normalVsAbnormal)
print X_test.shape
print Y_test.shape

X_train = X_data
Y_train = Y_data

print X_train.shape
print X_test.shape

print Y_train.shape
print Y_test.shape

def yDist(y):
    bcCounts = collections.defaultdict(int)
    for a in range(0, y.shape[0]):
        bcCounts[y[a][0]] += 1
    return bcCounts

print "Y_train Dist: " + str(yDist(Y_train))
print "Y_test Dist: " + str(yDist(Y_test))

X_train_s = X_train.reshape((1196,-1))
X_test_s = X_test.reshape((309,-1))
Y_train_s = Y_train.ravel()

model = SVC(C=1.0, gamma=0.001, kernel='rbf')
model.fit(X_train_s,Y_train_s)

predicted = model.predict(X_test_s)
expected = Y_test

svm_matrix = skm.confusion_matrix(Y_test, predicted)
svm_matrix

print metrics.accuracy_score(expected,predicted)

numBC = bc.reverseDict(categories)
class_names = numBC.values()
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
bc.plot_confusion_matrix(svm_matrix, classes=class_names[1:3],
                      title='Confusion Matrix without normalization')
plt.savefig('threshold_class2_bVm_o_norm.png')

from IPython.display import Image
Image(filename='threshold_class2_bVm_o_norm.png')

plt.figure()
bc.plot_confusion_matrix(svm_matrix, classes=class_names[1:3], normalize=True,
                      title='Confusion Matrix with normalization')
plt.savefig('threshold_class2_bVm_norm.png')

# Load the image we just saved
from IPython.display import Image
Image(filename='threshold_class2_bVm_norm.png')



