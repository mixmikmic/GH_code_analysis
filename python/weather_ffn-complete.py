import numpy as np
import os
from datetime import date
import re

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import pandas as pd
pd.options.display.max_rows = 30

training_df = pd.read_pickle("../data/weather/ts_temp_dp_press.p")

import random

HISTORY_LEN = 14
PREDICTION_LEN = 7
NUM_SAMPLES = 2000
NUM_INPUTS = len(training_df.columns)

indices = random.sample(range(0,len(training_df) - HISTORY_LEN - PREDICTION_LEN), NUM_SAMPLES)

ts_data = training_df.values
ts_inputs = np.array([ts_data[i:i + HISTORY_LEN] for i in indices]).reshape((NUM_SAMPLES, HISTORY_LEN * NUM_INPUTS))
ts_outputs = np.array([ts_data[i + HISTORY_LEN: i + HISTORY_LEN + PREDICTION_LEN] for i in indices]).reshape((NUM_SAMPLES, PREDICTION_LEN * NUM_INPUTS))

from keras import Model
from keras.layers import Input, Dense, Dropout

inputs = Input(shape=ts_inputs.shape[1:])
x = Dense(200, activation='tanh')(inputs)
x = Dropout(0.1)(x)
x = Dense(100, activation='tanh')(x)
x = Dropout(0.1)(x)
x = Dense(50, activation='tanh')(x)
x = Dropout(0.1)(x)
outputs = Dense(PREDICTION_LEN * NUM_INPUTS, activation='linear')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(ts_inputs, ts_outputs, validation_split=0.1, epochs=400)

import datetime
from datetime import timedelta

prediction_date = date(2013, 1, 7)

prediction_input = training_df[prediction_date - timedelta(days=HISTORY_LEN - 1): prediction_date]
prediction = model.predict(prediction_input.values.reshape(1, HISTORY_LEN * NUM_INPUTS))

history = training_df[prediction_date - timedelta(days=HISTORY_LEN - 1): prediction_date + timedelta(days=PREDICTION_LEN)]
prediction_df = pd.DataFrame(prediction.reshape((PREDICTION_LEN, NUM_INPUTS)), index=pd.DatetimeIndex(start=prediction_date + timedelta(days=1), freq='D', periods=PREDICTION_LEN))

plt.figure(figsize = (12, 6))
history_plt = plt.plot(history.iloc[:,0:3], linestyle='--')
plt.plot(history.iloc[:,3], 'oc')
plt.plot(prediction_df.iloc[:,0], color=history_plt[0].get_color())
plt.plot(prediction_df.iloc[:,1], color=history_plt[1].get_color())
plt.plot(prediction_df.iloc[:,2], color=history_plt[2].get_color())
plt.plot(prediction_df.iloc[:,3], 'or')
plt.show()

