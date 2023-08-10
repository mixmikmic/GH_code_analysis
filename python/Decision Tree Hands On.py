from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load the wine dataset
data = pd.read_csv('wine_original.csv')
labels = data['class']
del data['class']

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=5)

X_train

# Define Model
clf = DecisionTreeClassifier(criterion='entropy')
# Train
clf.fit(X_train, y_train)
# Predict
y_pred = clf.predict(X_test)
#Evaluate
y_pred_train = clf.predict(X_train)
print ('Train accuracy = ' + str(np.sum(y_pred_train == y_train)*1.0/len(y_train)))

print ('Test accuracy = ' + str(np.sum(y_pred == y_test)*1.0/len(y_test)))

from IPython.display import Image  
dot_data = export_graphviz(clf, out_file=None, 
                         feature_names=X_train.columns,  
                         class_names=['1','2','3'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())  

# Max Depth of tree

# Define Model
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
# Train
clf.fit(X_train, y_train)
# Predict
y_pred = clf.predict(X_test)
# Evaluate
print ('Test accuracy = ' + str(np.sum(y_pred == y_test)*1.0/len(y_test)))

dot_data = export_graphviz(clf, out_file=None, 
                         feature_names=X_train.columns,  
                         class_names=['1','2','3'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())  

X_test.loc[1]

y_pred[0]

#Min Sample Split

# Define Model
clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=10)
# Train
clf.fit(X_train, y_train)
# Predict
y_pred = clf.predict(X_test)
#Evaluate
print ('Test accuracy = ' + str(np.sum(y_pred == y_test)*1.0/len(y_test)))

dot_data = export_graphviz(clf, out_file=None, 
                         feature_names=X_train.columns,  
                         class_names=['1','2','3'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())

#min samples leaf

# Define Model
clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10)
# Train
clf.fit(X_train, y_train)
# Predict
y_pred = clf.predict(X_test)
# Evaluate
print ('Test accuracy = ' + str(np.sum(y_pred == y_test)*1.0/len(y_test)))

dot_data = export_graphviz(clf, out_file=None, 
                         feature_names=X_train.columns,  
                         class_names=['1','2','3'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=5)



