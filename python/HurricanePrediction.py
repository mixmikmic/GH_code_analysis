from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.noise import GaussianNoise, GaussianDropout
from datetime import datetime, timedelta
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
import pyowm
import yaml
import time
import os
import csv

with open('config.yml') as f:
    # use safe_load instead load
    config = yaml.safe_load(f)
    
owm_headers = ['timestamp', 'Max.TemperatureF', 'Min.TemperatureF', 'status_short', 'status', 'wind_speed', 'wind_dir', 'cloud_coverage', 'humidity', 'pressure', 'sea_level', 'rain', 'snow']
owm_data_path = 'HurricaneData/owm_houston.csv'

def render_plots_numbers(data):
    values = data.values
    cols_to_plot = [col for col in range(1, len(data.columns)) if type(values[0, col]) != str]
    pyplot.figure(figsize=(10, len(cols_to_plot) * 1.2))
    i = 1
    for group in cols_to_plot:
        pyplot.subplot(len(cols_to_plot), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(data.columns[group], y=1, loc='right')
        i += 1
    pyplot.tight_layout(h_pad=1)
    pyplot.show()

houston_weather = read_csv('HurricaneData/Preprocessed/houston.csv', index_col=0)
hurdat_houston = read_csv('HurricaneData/Preprocessed/hurdat_houston.csv', index_col=0)

houston_weather.index = pd.to_datetime(houston_weather.index)
hurdat_houston.index = pd.to_datetime(hurdat_houston.index)

hurdat_houston

render_plots_numbers(hurdat_houston)

houston_weather

render_plots_numbers(houston_weather)

houston_to_supervise = houston_weather.loc[:, ['Events', 'Max.TemperatureF', 'CloudCover', 'WindDirDegrees.br...', 'Max.Wind.SpeedMPH', 'Max.Sea.Level.PressureIn']]
houston_to_supervise

if 'Hurricane_Started' in houston_to_supervise:
    del houston_to_supervise['Hurricane_Started']
houston_to_supervise.insert(len(houston_to_supervise.columns), 'Hurricane_Started', 0)
hurricane_started = np.zeros(len(houston_to_supervise.index))
date_range = timedelta(days=2)
for i, row in enumerate(houston_to_supervise.itertuples()):
    start_date = row[0] - date_range
    end_date = row[0] + date_range
    mask = (hurdat_houston.index > start_date) & (hurdat_houston.index <= end_date)
    found_hurricane = len(hurdat_houston[mask].index) > 0
    if found_hurricane:
        hurricane_started[i] = 1
hurricane_started.shape, houston_to_supervise.shape

houston_to_supervise

values = houston_to_supervise.values
encoder = LabelEncoder()
values[:,0] = encoder.fit_transform(values[:,0])

# convert series to superv ised learning
def series_to_supervised(data, dataset_cols, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('%s(t-%d)' % (dataset_cols[j], i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('%s(t)' % (dataset_cols[j])) for j in range(n_vars)]
		else:
			names += [('%s(t+%d)' % (dataset_cols[j], i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, houston_to_supervise.columns, 1, 1)

# drop columns we don't want to predict
reframed.drop(reframed.loc[:, 'Events(t)' : 'WindDirDegrees.br...(t)'].head(0).columns, axis=1, inplace=True)
reframed.drop(reframed.loc[:, ['Max.Sea.Level.PressureIn(t)']].head(0).columns, axis=1, inplace=True)

reframed.drop(reframed.loc[:, ['Hurricane_Started(t-1)']].head(0).columns, axis=1, inplace=True)
#reframed.drop(reframed.loc[:, ['Max.Wind.SpeedMPH(t-1)']].head(0).columns, axis=1, inplace=True)
print(reframed.head(), reframed.shape, values.shape)

# split into train and test sets
train_percent = 0.8
train_amount = math.floor(len(reframed.values) * train_percent)
train = reframed.values[:train_amount, :]
test = reframed.values[train_amount:, :]
# split into input and outputs
train_X, train_y = train[:, :-2], train[:, -2:]
test_X, test_y = test[:, :-2], test[:, -2:]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(15, recurrent_dropout=0.2, dropout=0.2, return_sequences=False, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(2))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=5, batch_size=128, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# Clear the screen so we can see the charts
clear_output()

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

def weather_row_to_pandas(rows):
    def val_by_name(r, name):
        return r[owm_headers.index(name)]
    
    def to_normal_event(status_short):
        d = {
            'Rain': 'Rain'           
        }
        if status_short in d:
            return d[status_short]
        return '0'
    
    data = []
    index = []
    for r in rows:
        index.append(pd.to_datetime(r[0], unit='s'))
        data.append([to_normal_event(val_by_name(r, 'status_short')), val_by_name(r, 'Max.TemperatureF'),
                     val_by_name(r, 'cloud_coverage'), val_by_name(r, 'wind_dir'), val_by_name(r, 'wind_speed'), val_by_name(r, 'sea_level')])
        
    data = np.array(data) # make sure all data are floats or ints by converting to a numpy array
    return (pd.DataFrame(data=data,
                      index=index,
                      columns=['Events', 'Max.TemperatureF', 'CloudCover', 'WindDirDegrees.br...', 'Max.Wind.SpeedMPH', 'Max.Sea.Level.PressureIn']))

if os.path.isfile(owm_data_path):
    with open(owm_data_path, 'r') as csv_file:
        reader = csv.reader(csv_file,  delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        print(next(reader)) # skip header
        owm_data = weather_row_to_pandas(reader)
owm_data

# ensure all data is float
data = owm_data.apply(pd.to_numeric, args=('coerce',))
data.fillna(0, inplace=True)
values = data.values
values = values.astype('float32')
scaled = scaler.fit_transform(values)

reframed = series_to_supervised(scaled, data.columns, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.loc[:, 'Events(t)' : 'WindDirDegrees.br...(t)'].head(0).columns, axis=1, inplace=True)
reframed.drop(reframed.loc[:, ['Max.Sea.Level.PressureIn(t)']].head(0).columns, axis=1, inplace=True)
pred = reframed.values
pred = pred[:, :-1]
pred = pred.reshape((pred.shape[0], 1, pred.shape[1]))
yhat = model.predict(pred)

# Here we create an empty array with the same dimensions as where we had our original data.
# We then place the values that have been predicted (wind speed) under the same columns as in the original data.
# This is at column 4 and 5.
# We can then use the scaler to inverse the normalization of the numbers.
inv_yhat = np.zeros((yhat.shape[0], values.shape[1]))
inv_yhat[:, 4] = yhat[:, 0]

inv_yhat = scaler.inverse_transform(inv_yhat)
yhat[:, 0] = inv_yhat[:, 4]

df = pd.DataFrame(data=yhat,
                  index=data.index[:-1],
                  columns=['Predicted Wind Speed (MPH)', 'Hurricane Chance'])
df



