import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

dataset = datasets.load_breast_cancer()
X_train, X_test, Y_train, Y_test = train_test_split(dataset.data, dataset.target, test_size = 0.2, random_state = 0)

clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(X_train, Y_train)

clf.score(X_test, Y_test)

def train(x,y):
    return 

def predict_one(x_train, y_train, x_test, k):
    distances = []
    
    for i in range(len(x_train)):
        distance = ((x_train[i, :] - x_test)**2).sum()
        distances.append([distance, i])
    
    distances = sorted(distances)
    
    targets = []
    for i in range(k):
        index_of_training_data = distances[i][1]
        targets.append(y_train[index_of_training_data])
        
    return Counter(targets).most_common(1)[0][0] 

def predict(x_train, y_train, x_test_data, k):
    predictions = []
    
    for x_test in x_test_data:
        predictions.append(predict_one(x_train, y_train, x_test, k))
        
    return predictions
    

y_pred = predict(X_train, Y_train, X_test, 7)
accuracy_score(Y_test, y_pred)

a = [1,1,0,1,0,1,1]

Counter(a).most_common(1)[0][0]         

##counter gives count of all the unique enteries
## most commmon gives tuples of most common k enteries where k is passed as parameter

