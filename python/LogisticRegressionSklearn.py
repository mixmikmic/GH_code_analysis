# Import Dependecies

import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, linear_model
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Load Data

df = pd.read_csv('dataset/breast-cancer-wisconsin-data.csv')

# Let's have a look at the data
# Let's print first 5 row values from data.

print(df.head())

# Load data with Column names
# Here we provide the names for each Column.

df = pd.read_csv('dataset/breast-cancer-wisconsin-data.csv', names=['id', 'clump_thickness','unif_cell_size',
                                                                           'unif_cell_shape', 'marg_adhesion', 'single_epith_cell_size',
                                                                           'bare_nuclei', 'bland_chromatin', 'normal_nucleoli','mitoses','class'])

# Let's check the data again

print(df.head())

# Correlation between different features
correlation = df.corr()
print(df.corr())

# Let's see this in a heatmap

plt.figure(figsize=(15, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm')

# Feature Selection

df.drop(['id'], 1, inplace=True)

# Filtered Dataset
print(df.head())

# Replace empty values with Outlier value

df.replace('?', -99999, inplace=True)

# Features
X = np.array(df.drop(['class'],1))

# Labels
y = np.array(df['class'])

# Let's have a look at our Features and Labels
print('Features: \n',X)
print('Labels: \n',y)

# Cross Validation
# Test Data: 20% of total data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

# Define the Classifier
clf = linear_model.LogisticRegression(n_jobs = 10)

# Train the Classifier
clf.fit(X_train,y_train)

# Predcitions on Test Data
y_pred = clf.predict(X_test)
print('Predicted Labels: ',y_pred)

# Number of Misclassified Labels
print('Number of Misclassified Labels: {}'.format((y_pred != y_test).sum()))

print('Total Predicted Values: ',len(y_pred))

# Confidence
confidence = clf.score(X_test, y_test)
print('Confidence of Classifier: ',confidence)

# Accuracy Score
acc = accuracy_score(y_test,y_pred)
print('Accuracy Score of Classifier: ',acc)

# Confusin Matrix
conf = confusion_matrix(y_test, y_pred)
print('Confusion Matrix: \n',conf)

