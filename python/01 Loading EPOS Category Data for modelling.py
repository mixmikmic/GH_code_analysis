# Imports
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Training Data
training_raw = pd.read_table("../data/training_data.dat")
df_training = pd.DataFrame(training_raw)
df_training.head()

# test Data
test_raw = pd.read_table("../data/test_data.dat")
df_test = pd.DataFrame(test_raw)
df_test.head()

# target names
target_categories = ['Unclassified','Art','Aviation','Boating','Camping /Walking /Climbing','Collecting']
target_values = ['1','528','529','530','531','532']

# features
feature_names = ['Barcode','Description','UnitRRP']

# Extract features from panda
training_data = df_training[feature_names].values
training_data[:3]

# Extract target results from panda
target = df_training["CategoryID"].values

# Create classifier class
model_dtc = DecisionTreeClassifier()

# train model
model_dtc.fit(training_data, target)

# features
feature_names_integers = ['Barcode','UnitRRP']

# Extra features from panda (without description)
training_data_integers = df_training[feature_names_integers].values
training_data_integers[:3]

# train model again
model_dtc.fit(training_data_integers, target)

# Extract test data and test the model
test_data_integers = df_test[feature_names_integers].values
test_target = df_test["CategoryID"].values
expected = test_target
predicted_dtc = model_dtc.predict(test_data_integers)

print(metrics.classification_report(expected, predicted_dtc,    target_names=target_categories))

print(metrics.confusion_matrix(expected, predicted_dtc))

metrics.accuracy_score(expected, predicted, normalize=True, sample_weight=None)

predicted[:5]

from sklearn.linear_model import SGDClassifier

# Create classifier class
model_sgd = SGDClassifier()

# train model again
model_sgd.fit(training_data_integers, target)

predicted_sgd = model_sgd.predict(test_data_integers)

print(metrics.classification_report(expected, predicted_sgd,    target_names=target_categories))

print(metrics.confusion_matrix(expected, predicted_sgd))

metrics.accuracy_score(expected, predicted_sgd, normalize=True, sample_weight=None)



