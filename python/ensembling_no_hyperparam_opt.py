import wget
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

# Import the dataset
data_url = 'https://raw.githubusercontent.com/nslatysheva/data_science_blogging/master/datasets/wine/winequality-red.csv'
dataset = wget.download(data_url)
dataset = pd.read_csv(dataset, sep=";")

# Using a lambda function to bin quality scores
dataset['quality_is_high'] = dataset.quality.apply(lambda x: 1 if x >= 6 else 0)

# Convert the dataframe to a numpy array and split the
# data into an input matrix X and class label vector y
npArray = np.array(dataset)
X = npArray[:,:-2].astype(float)
y = npArray[:,-1]

# Split into training and test sets
XTrain, XTest, yTrain, yTest = train_test_split(X, y, random_state=1)

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

# Build rf model
best_n_estimators, best_max_features = 73, 5
rf = RandomForestClassifier(n_estimators=best_n_estimators, max_features=best_max_features)
rf.fit(XTrain, yTrain)
rf_predictions = rf.predict(XTest)

# Build SVM model
best_C_svm, best_gamma = 1.07, 0.01
rbf_svm = svm.SVC(kernel='rbf', C=best_C_svm, gamma=best_gamma)
rbf_svm.fit(XTrain, yTrain)
svm_predictions = rbf_svm.predict(XTest)

# Build LR model
best_penalty, best_C_lr = "l2", 0.52
lr = LogisticRegression(penalty=best_penalty, C=best_C_lr)
lr.fit(XTrain, yTrain)
lr_predictions = lr.predict(XTest)

# Train SVM and output predictions
# rbfSVM = svm.SVC(kernel='rbf', C=best_C, gamma=best_gamma)
# rbfSVM.fit(XTrain, yTrain)
# svm_predictions = rbfSVM.predict(XTest)

print (classification_report(yTest, svm_predictions))
print ("Overall Accuracy:", round(accuracy_score(yTest, svm_predictions),4))

print(best_C, best_C_svm)

import collections

# stick all predictions into a dataframe
predictions = pd.DataFrame(np.array([rf_predictions, svm_predictions, lr_predictions])).T
predictions.columns = ['RF', 'SVM', 'LR']

# initialise empty array for holding predictions
ensembled_predictions = np.zeros(shape=yTest.shape)

# majority vote and output final predictions
for test_point in range(predictions.shape[0]):
    row = predictions.iloc[test_point,:]
    counts = collections.Counter(row)
    majority_vote = counts.most_common(1)[0][0]
    
    # output votes
    ensembled_predictions[test_point] = majority_vote.astype(int)
    #print "The majority vote for test point", test_point, "is: ", majority_vote

print(ensembled_predictions)

# Get final accuracy of ensembled model
from sklearn.metrics import classification_report, accuracy_score

for individual_predictions in [rf_predictions, svm_predictions, lr_predictions]:
#     classification_report(yTest.astype(int), individual_predictions.astype(int))
    print "Accuracy:", round(accuracy_score(yTest.astype(int), individual_predictions.astype(int)),2)


print classification_report(yTest.astype(int), ensembled_predictions.astype(int))
print "Ensemble Accuracy:", round(accuracy_score(yTest.astype(int), ensembled_predictions.astype(int)),2)

# from sklearn.ensemble import VotingClassifier
import sklearn.ensemble.VotingClassifier

# Build and fit majority vote classifier
# ensemble_1 = VotingClassifier(estimators=[('rf', rf), ('svm', rbf_svm), ('lr', lr)], voting='hard')
# ensemble_1.fit(XTrain, yTrain)

# simple_ensemble_predictions = ensemble_1.predict(XTest)
# print metrics.classification_report(yTest, simple_ensemble_predictions)
# print "Ensemble_2 Overall Accuracy:", round(metrics.accuracy_score(yTest, simple_ensemble_predictions),2)

# Getting weights

ensemble_1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], weights=[1,1,1], voting='hard')
ensemble_1.fit(XTrain, yTrain)

simple_ensemble_predictions = ensemble_1.predict(XTest)
print metrics.classification_report(yTest, simple_ensemble_predictions)
print "Ensemble_2 Overall Accuracy:", round(metrics.accuracy_score(yTest, simple_ensemble_predictions),2)

ensemble_1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], weights=[1,1,1], voting='soft')
ensemble_1.fit(XTrain, yTrain)

simple_ensemble_predictions = ensemble_1.predict(XTest)
print metrics.classification_report(yTest, simple_ensemble_predictions)
print "Ensemble_2 Overall Accuracy:", round(metrics.accuracy_score(yTest, simple_ensemble_predictions),2)

## Model stacking



