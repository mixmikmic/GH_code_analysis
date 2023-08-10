import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, log_loss, average_precision_score

from textblob import TextBlob

import logging

get_ipython().magic('matplotlib inline')

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Set up my data directories from different machines

mac_data_dir = '/Users/christopherallison/Documents/Coding/Data'
linux_data_dir = '/home/chris/data'
win_data_dir = u'C:\\Users\\Owner\\Documents\\Data'

# Set data directory for example

data_dir = mac_data_dir

# Load our prepared dataset and reference data

X = pd.read_csv(os.path.join(data_dir, "prepared_animals_df.csv"),index_col=0)

# Check our data

X.columns

# Double check our data

X.head()

outcomes = X.OutcomeType.unique()

from sklearn import preprocessing

# This code takes our text labels and creates an encoder that we use
# To transform them into an array

encoder = preprocessing.LabelEncoder()
encoder.fit(outcomes)

encoded_y = encoder.transform(outcomes)
encoded_y

#We can also inverse_transform them back.
list(encoder.inverse_transform([0, 1, 2, 3]))

X.OutcomeType = encoder.transform(X.OutcomeType)

# Check our work

X.head()

outcomes = encoder.inverse_transform([0, 1, 2, 3, 4])
outcomes

train_features = X.values[:,1:]
train_features[:5]

train_target = X['OutcomeType'].values
train_target

# Set up our train_test_split

X_train, x_test, y_train, y_test = train_test_split(train_features,
                                          train_target,
                                          test_size=0.4,
                                          random_state=42)

X.drop('OutcomeType', axis=1, inplace=True)

# Let's try a different estimator
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

dt_clf = DecisionTreeClassifier()
dt_clf = dt_clf.fit(X_train, y_train)

score = dt_clf.score(X_train, y_train)
"Mean accuracy of Decision Tree: {0}".format(score)

dt_y_predict = dt_clf.predict(x_test)

from sklearn.metrics import accuracy_score
from sklearn import metrics
print ("Accuracy = %.2f" % (accuracy_score(y_test, dt_y_predict)))

from sklearn.cross_validation import cross_val_predict

predicted = cross_val_predict(dt_clf, X_train, y_train, cv=10)
metrics.accuracy_score(y_train, predicted) 



from sklearn.externals.six import StringIO

with open(os.path.join(data_dir, "shelter.dot"), 'w') as f:
    f = tree.export_graphviz(dt_clf, out_file=f)

# Warning - this code block is system intensive and may not work
# on lower spec systems.

import pydotplus as pydot
dot_data = StringIO() 
tree.export_graphviz(dt_clf, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf(os.path.join(data_dir, "shelter.pdf"))

# Warning - this code block is system intensive and may not work
# on lower spec systems.

from IPython.display import Image  
dot_data = StringIO()  
tree.export_graphviz(dt_clf, out_file=dot_data,  
                         feature_names=X.columns,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

# Evaluate the model
print (X_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)



model_score = dt_clf.score(x_test, y_test)
print ("Model Score %.2f \n" % (model_score))

confusion_matrix = metrics.confusion_matrix(y_test, dt_y_predict)
print ("Confusion Matrix \n", confusion_matrix)

print ("          Predicted")
print ("         |  0  |  1  |")
print ("         |-----|-----|")
print ("       0 | %3d | %3d |" % (confusion_matrix[0, 0],
                                   confusion_matrix[0, 1]))
print ("Actual   |-----|-----|")
print ("       1 | %3d | %3d |" % (confusion_matrix[1, 0],
                                   confusion_matrix[1, 1]))
print ("         |-----|-----|")

import itertools

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    # Simple plot based on the Iris sample CM
    plt.figure(figsize=(8, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(confusion_matrix, outcomes, title="Animal Sanctuary Confusion Matrix")

# Plot the importance of the different features

importances = dt_clf.feature_importances_
std = np.std(dt_clf.feature_importances_)
indices = np.argsort(importances)[::-1]

for f in range(X.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

