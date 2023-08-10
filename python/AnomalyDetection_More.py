import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
get_ipython().magic('matplotlib inline')

boston = load_boston()

print(boston.DESCR)

indices = []

for i in range(len(boston["feature_names"])):
    indices.append(i)

columns = np.array(list(zip(boston["feature_names"], indices)))

print(columns)

# print out the column indices so it's easier to refer to

boston["target"][0:5]

boston_X = boston.data[:, (4, 6, 8, 10, 12)]
# Select out just a few of the features 
# NOX      nitric oxides concentration (parts per 10 million)
# AGE      proportion of owner-occupied units built prior to 1
# RAD      index of accessibility to radial highways
# PTRATIO  pupil-teacher ratio by town
# LSTAT    % lower status of the population

boston_y = boston["target"]

combined = np.column_stack([boston_X, boston_y])

dataset = pd.DataFrame(combined).sample(frac = 1).reset_index(drop = True)

dataset.columns = ["NOX", "AGE", "RAD", "PTRATIO", "LSTAT", "TARGET"]

dataset.head()

X = np.array(dataset.iloc[:, :-1]) # everything except for TARGET
y = np.array(dataset["TARGET"])

print(X.shape)

print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)

naive_model = Ridge()

naive_model

scores = cross_val_score(naive_model, X, y, cv = 3) # 3 folds

print("Scores:", scores)

print("Mean Scores:", np.mean(scores))

# fit the model with training data
naive_model.fit(X_train, y_train)

# predict 
naive_predictions = naive_model.predict(X_test)
print(X_test.shape, naive_predictions.shape)

plt.scatter(y_test, naive_predictions)
z = np.polyfit(y_test, naive_predictions, 1)
p = np.poly1d(z)
plt.title("Predicted vs actual target values")
plt.xlabel("Actual y value")
plt.ylabel("Predicted y value")
plt.plot(y_test, p(y_test), "k--")

naive_model.score(X_test, y_test)

from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(n_estimators = 250, bootstrap = True)

iso_forest.fit(X, y)

# get outliers
iso_outliers = iso_forest.predict(X) == -1

iso_outliers[0:10]

# remove outliers from the dataset

X_iso = X[~iso_outliers]
y_iso = y[~iso_outliers]

X_train_iso, X_test_iso, y_train_iso, y_test_iso = train_test_split(X_iso, y_iso, test_size = 0.3)

print(X_train_iso.shape) # new X_train without outliers

print(X_train.shape) # old X_train with outliers

iso_model = Ridge()
iso_model.fit(X_train_iso, y_train_iso)

iso_scores = cross_val_score(estimator = iso_model, X = X_test_iso, y = y_test_iso)

print("Iso Scores:", iso_scores)

print("Mean Iso Scores", np.mean(iso_scores))

# predict 
iso_predictions = iso_model.predict(X_test)

iso_predictions[0:10]

plt.scatter(y_test, iso_predictions)
z = np.polyfit(y_test, iso_predictions, 1)
p = np.poly1d(z)
plt.title("Predicted vs actual target values")
plt.xlabel("Actual y value")
plt.ylabel("Model y value")
plt.plot(y_test, p(y_test), "k-")

iso_model.score(X_test_iso, y_test_iso)

from sklearn.svm import OneClassSVM

svm = OneClassSVM(kernel = "rbf")

svm.fit(X, y)

svm_outliers = svm.predict(X) == -1
svm_outliers[0:10]

# remove outliers
X_svm = X[~svm_outliers]
y_svm = y[~svm_outliers]

print(X_svm.shape) # without outliers

print(X.shape) # with outliers

X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_svm, y_svm, test_size = 0.3)

svm_model = Ridge().fit(X_train_svm, y_train_svm)

svm_scores = cross_val_score(estimator = svm_model, X = X_test_svm, y = y_test_svm)

print("svm scores", svm_scores)

print("Mean svm scores", np.mean(svm_scores))

svm_predictions = svm_model.predict(X_test)

plt.scatter(y_test, svm_predictions)
z = np.polyfit(y_test, svm_predictions, 1)
p = np.poly1d(z)
plt.title("Predicted vs actual target values")
plt.xlabel("Actual y value")
plt.ylabel("Model y value")
plt.plot(y_test, p(y_test), "k-")

svm_model.score(X_test, y_test)



