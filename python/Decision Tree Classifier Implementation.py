import numpy as np
import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

My_data =  pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
                           sep= ',', header= None)

#See my data, its a random table consisting of many values .
#Let's Analyze its properties
My_data

#As seen , but if you wanna check its dimensions and shape

print "Dataset Lenght = ", len(My_data)
print "Dataset Shape = ", My_data.shape

# The Top rows of the data by

print "Dataset =  "
My_data.head()

#Slicing and showing you the rows : 
X = My_data.values[:, 1:5]
Y = My_data.values[:,0]

Y

# Main Part :

#####Training and test set :

## split the data to : X_train, y_train are training data &  X_test, y_test belongs to the test dataset

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

# The parameter test_size is given value 0.3; it means test sets will be 30% of whole dataset 
# & training dataset’s size will be 70% of the entire dataset. random_state variable is 
# a pseudo-random number generator state used for random sampling. 

# DecisionTreeClassifier(): This is the classifier function for DecisionTree. It is the main function for implementing
# the algorithms. Some important parameters are:







clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)



clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

# Now, we have modeled 2 classifiers. One classifier with gini index & another one with information gain as the
# criterion. We are ready to predict classes for our test set. We can use predict() method. Let’s try to predict. 

#with gini index :
clf_gini.predict([[4, 4, 3, 3]])

#### Prediction for Decision Tree classifier with criterion as gini index

y_pred = clf_gini.predict(X_test)
y_pred

#### Prediction for Decision Tree classifier with criterion as information gain

y_pred_en = clf_entropy.predict(X_test)
y_pred_en

# Accuracy for Decision Tree classifier with criterion as gini index

#  The parameter y_true  accepts an array of correct labels and y_pred takes an array of predicted labels 
#  that are returned by the classifier. It returns accuracy as a float value.

print "Accuracy is ", accuracy_score(y_test,y_pred)*100

## Accuracy for Decision Tree classifier with criterion as information gain

print "Accuracy is ", accuracy_score(y_test,y_pred_en)*100

##That's it :))



