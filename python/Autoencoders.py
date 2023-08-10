#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import keras

#Import Keras module
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, advanced_activations
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

#pretty plots
get_ipython().run_line_magic('matplotlib', 'inline')

#Load the index data
raw_data = pd.read_csv('price_only.csv', skiprows=1, parse_dates=['Date']).set_index(['Date'])
raw_data.tail()

#Data and Labels split
y = raw_data['P']
X = raw_data

#Training, Validation, test split
train_size = 0.9
timestep = 5

#Calcualte cutoff
train_cut_index = int(train_size * X.shape[0] - train_size * X.shape[0] % timestep)
print(train_cut_index)

#Split train, validation
X_train, X_val = X.iloc[0:train_cut_index,:], X.iloc[train_cut_index:,:]
y_train, y_val = y[0:train_cut_index], y[train_cut_index:]



