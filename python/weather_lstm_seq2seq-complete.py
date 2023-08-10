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

PREDICTION_LEN = 7
HISTORY_LEN = 14
NUM_SAMPLES = 2000
NUM_INPUTS = len(training_df.columns)

indices = random.sample(range(0,len(training_df) - HISTORY_LEN - 1), NUM_SAMPLES)

ts_data = training_df.values
ts_inputs = np.array([ts_data[i:i + HISTORY_LEN] for i in indices])
ts_outputs = np.array([ts_data[i + 1: i + 1 + HISTORY_LEN] for i in indices])

from keras import Model
from keras.layers import LSTM, GRU, Input, Dense, CuDNNGRU

inputs = Input(shape=ts_inputs.shape[1:])
x = GRU(64, return_sequences=True, recurrent_activation='sigmoid')(inputs)
x = GRU(32, return_sequences=True, recurrent_activation='sigmoid')(x)
outputs = Dense(NUM_INPUTS)(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(ts_inputs, ts_outputs, validation_split=0.1, epochs=100)

model.save_weights('model_plus1_predition.hdf')

inputs = Input(batch_shape=(1, 1, NUM_INPUTS))
x = GRU(64, return_sequences=True, stateful=True)(inputs)
x = GRU(32, return_sequences=True, stateful=True)(x)
outputs = Dense(NUM_INPUTS)(x)

pred_model = Model(inputs, outputs)
pred_model.summary()
pred_model.load_weights('model_plus1_predition.hdf')

import datetime
from datetime import timedelta

prediction_date = date(2014, 12, 23)

prediction_input = training_df[prediction_date - timedelta(days=HISTORY_LEN - 1): prediction_date]

pred_model.reset_states()
for i in range(0, HISTORY_LEN):
    prediction = pred_model.predict(prediction_input.values[i].reshape(1, 1, NUM_INPUTS))
    
self_predicition = [np.copy(prediction)]

for i in range(0, PREDICTION_LEN - 1):
    prediction = pred_model.predict(self_predicition[i])
    self_predicition.append(np.copy(prediction))
    


history = training_df[prediction_date - timedelta(days=HISTORY_LEN - 1): prediction_date + timedelta(days=PREDICTION_LEN)]
prediction_df = pd.DataFrame(np.vstack(self_predicition).reshape((PREDICTION_LEN, NUM_INPUTS)), index=pd.DatetimeIndex(start=prediction_date + timedelta(days=1), freq='D', periods=PREDICTION_LEN))

plt.figure(figsize = (12, 6))
history_plt = plt.plot(history.iloc[:,0:3], linestyle='--')
plt.plot(history.iloc[:,3], 'oc')
plt.plot(prediction_df.iloc[:,0], color=history_plt[0].get_color())
plt.plot(prediction_df.iloc[:,1], color=history_plt[1].get_color())
plt.plot(prediction_df.iloc[:,2], color=history_plt[2].get_color())
plt.plot(prediction_df.iloc[:,3], 'or')
plt.show()

