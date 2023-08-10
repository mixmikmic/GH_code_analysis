# Import Dependencies
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Import Data
data = pd.read_csv('cal_housing_clean.csv')

data.head()

data.describe()

# Features
X = data.drop('medianHouseValue', axis=1)

# Labels 
y = data['medianHouseValue']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)

print(X_train)

print(y_train)

# Scale the Feature Data
# Do scaling only on Training Data

# MinMaxScaler: Transforms features by scaling each feature to a given range.

# Transformation is Given by:
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)

# Make X_train to be the Scaled Version of Data
# X_train has 6 columns
# This process scales all the values in all 6 columns and replaces them with the new values
X_train = pd.DataFrame(data=scaler.transform(X_train), columns=X_train.columns, index=X_train.index)

print(X_train)

# Do same Scaling for Test Features
scal = MinMaxScaler()
scal.fit(X_test)
X_test = pd.DataFrame(data=scal.transform(X_test), columns=X_test.columns, index=X_test.index)

print(X_test)

data.columns

# Make Feature Columns
age = tf.feature_column.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
population = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
med_income = tf.feature_column.numeric_column('medianIncome')

feat_cols = [age, rooms, bedrooms, population, households, med_income]

input_func = tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size=10, num_epochs=1000, shuffle=True)

model = tf.estimator.DNNRegressor(hidden_units=[10,20,20,10], feature_columns=feat_cols)

model.train(input_fn=input_func, steps=2000)

predict_input_func = tf.estimator.inputs.pandas_input_fn(X_test, batch_size=10, num_epochs=1, shuffle=False)

pred = model.predict(input_fn=predict_input_func)

predictions = list(pred)

predictions

final_preds = []

for pred in predictions:
    final_preds.append(pred['predictions'])

print(len(final_preds))

from sklearn.metrics import mean_squared_error, classification_report

# Root Mean Squared Error
mean_squared_error(y_test,final_preds)**0.5

