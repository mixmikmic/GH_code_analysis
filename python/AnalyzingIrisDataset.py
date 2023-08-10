import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from matplotlib import pyplot as plt
import random

df = pd.read_csv("iris-species/Iris.csv")

df.describe()

df.head()

data = df.iloc[:,:4]
(data.head(10))

target = df.iloc[:,5:]
target.head()

target.replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],[0.0,1.0,2.0],inplace=True)

print(target.head())

# create training and testing vars
n = random.randint(1,100)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state = n)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#Linear Regression

lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
print("Accuracy for linear regression: " + str(model.score(X_test, y_test)))

#Logistic Regression
n = random.randint(1,100)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state = n)

lg = linear_model.LogisticRegression()
model = lg.fit(X_train,y_train.values.ravel())
predictions = lg.predict(X_test)
print("Accuracy for logistic regression: " + str(model.score(X_test,y_test)))

#Decision Tree Classifier

n = random.randint(1,100)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state = 40)

dTree = tree.DecisionTreeClassifier()
model = dTree.fit(X_train,y_train)
predictions = dTree.predict(X_test)
print("Accuracy for Decision Tree Classifier: " + str(model.score(X_test,y_test)))


#Naive Bayes

n = random.randint(1,100)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state = n)

nb = GaussianNB()
model = nb.fit(X_train,y_train.values.ravel())
predictions = nb.predict(X_test)
print("Accuracy for Naive Bayes: " + str(model.score(X_test,y_test)))

#Support Vector Machine

n = random.randint(1,100)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state = n)

supportVm = svm.SVC()
model = supportVm.fit(X_train,y_train.values.ravel())
predictions = supportVm.predict(X_test)
print("Accuracy for Support Vector Machine: " + str(model.score(X_test,y_test)))



