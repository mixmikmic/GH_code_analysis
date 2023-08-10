import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('C:/Users/R AMARTYA/Desktop/Artificial_Neural_Networks/Churn_Modelling.csv')    

dataset.head(5)

train_features = dataset.iloc[:, 3:13].values
train_target = dataset.iloc[:, 13].values

train_features

train_target

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le=LabelEncoder()

train_features[:,1]=le.fit_transform(train_features[:,1])
train_features[:,2]=le.fit_transform(train_features[:,2])

train_features

oe = OneHotEncoder(categorical_features = [1])

train_features=oe.fit_transform(train_features).toarray()

train_features=train_features[:,1:]

train_features

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(train_features, train_target, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
#clf = clf.fit(train_x, train_y)
#predict_y = clf.predict(test_x)
#print ("Naive Bayes Acuracy = %.2f" % (accuracy_score(test_y, predict_y)))
#cm = confusion_matrix(test_y, predict_y)
#print(cm)

#from sklearn.ensemble import RandomForestClassifier
#RandomForest = RandomForestClassifier(n_estimators=1000)
#RandomForest.fit(train_x,train_y)
#predict_y = RandomForest.predict(test_x)
#print ("RandomForest Accuracy = %.2f" % (accuracy_score(test_y, predict_y)))
#cm = confusion_matrix(test_y, predict_y)
#print(cm)

import keras
from keras.models import Sequential
from keras.layers import Dense

clf = Sequential()
clf.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
clf.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
clf.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

clf.fit(train_x, train_y, batch_size = 10, nb_epoch = 100)

from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

predict_y = clf.predict(test_x)
predict_y = (predict_y > 0.5)
print (" Accuracy = %.2f" % (accuracy_score(test_y, predict_y)))
cm = confusion_matrix(test_y, predict_y)
print(cm)

print(f1_score(test_y, predict_y, average="macro"))
print(precision_score(test_y, predict_y, average="macro"))
print(recall_score(test_y, predict_y, average="macro"))

