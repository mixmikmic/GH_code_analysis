import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import theano

import matplotlib.pyplot as plt
import os

# Set up my data directories from different machines

mac_data_dir = '/Users/christopherallison/Documents/Coding/Data'
linux_data_dir = '/home/chris/data'
win_data_dir = u'C:\\Users\\Owner\\Documents\\Data'

# Set data directory for example

data_dir = mac_data_dir

# Load our prepared dataset and reference data

df = pd.read_csv(os.path.join(data_dir, "prepared_animals_df.csv"),index_col=0)

df.head()

# Drop uneccesary columns
X = df.drop('OutcomeType', axis=1)
X.dtypes

# We now have a dataframe with 7 features.

X.head()

X = X.as_matrix()

outcomes = df.OutcomeType.unique()

from sklearn import preprocessing

# This code takes our text labels and creates an encoder that we use
# To transform them into an array

encoder = preprocessing.LabelEncoder()
encoder.fit(outcomes)

encoded_y = encoder.transform(outcomes)
encoded_y

#We can also inverse_transform them back.
list(encoder.inverse_transform([0, 1, 2, 3, 4]))

#We still need to transform the array into a matrix - this is called 
# one hot encoding. It allows us to track the probability of each possible outcome separately.

#First, we'll transform the labels into their array value.
df.OutcomeType = encoder.transform(df.OutcomeType)

from keras.utils import np_utils

train_target = np_utils.to_categorical(df['OutcomeType'].values)
train_target

model = Sequential()
model.add(Dense(5, input_dim=7, init='normal', activation="relu"))
model.add(Dense(5, init='normal', activation='sigmoid'))

# Compile model
print("Compiling model...")
model.compile(loss='categorical_crossentropy', optimizer='adam',
             metrics=['accuracy'])

hist = model.fit(X, train_target, validation_split=0.2)
print("")
print(hist.history)

model.evaluate(X, train_target)

model.predict_classes(X[0:10], verbose=1)

encoder.inverse_transform([0, 0, 3, 4, 3, 4, 4, 0, 0, 0])

model.predict_proba(X[0:2], verbose=1)

