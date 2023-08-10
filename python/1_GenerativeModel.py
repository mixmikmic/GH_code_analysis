from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load the wine dataset
data = pd.read_csv('wine_original.csv')
labels = data['class']
del data['class']

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=10)

print (X_train.shape)
X_train.head()

from sklearn.naive_bayes import GaussianNB, MultinomialNB

# Initialize Gaussian Naive Bayes
gnb = GaussianNB()
# Train the classifier
gnb.fit(X_train, y_train)
# Make predictions on test data
y_pred = gnb.predict(X_test)
y_train_pred = gnb.predict(X_train)

# print the accuracy
print ('Training accuracy = ' + str(np.sum(y_train_pred == y_train)/len(y_train)))
print ('Test accuracy = ' + str(np.sum(y_pred == y_test)/len(y_test)))

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=5)

parameters = { 'alpha' : [0, 0.1, 0.5, 1, 2, 3, 4, 5, 10, 100] }
mnb = MultinomialNB()
clf = GridSearchCV(mnb, parameters, verbose=True, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#accuracy = np.sum(y_pred == y_test)/len(y_test)
accuracy = accuracy_score(y_pred, y_test)
train_acc = accuracy_score(clf.predict(X_train), y_train)
print ('Test accuracy = ' + str(accuracy))# + ' at alpha = ' + str(alpha))
print ('Train accuracy = ' + str(train_acc)) 
print (clf.best_params_)

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=20)



