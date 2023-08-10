# Imports
import pandas as pd
import numpy as np

# Load Data
weather_2015_2017 = pd.read_csv('cleaned_weather_2015_2017.csv', index_col=0)

weather_2015_2017.head()

len(weather_2015_2017)

beijing_spatial_15_17 = pd.read_csv('merged_spatial_weather_2015_2017.csv', index_col=0)

beijing_spatial_15_17.head()

weather_2015_2017['pollution'] = beijing_spatial_15_17['avg_air_pollution']

weather_2015_2017.head()

UCI_2010_2014 = pd.read_csv('pollution.csv', index_col=0).reset_index(drop=True)

UCI_2010_2014.head()

merged_final_pollution = weather_2015_2017.copy()

merged_final_pollution.columns = ['dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain',
                                 'year', 'month', 'day', 'hour', 'pollution']

merged_final_pollution.head()

merged_final_pollution.to_csv('UCI_year_mo_day_pollution_weather_2015_2017.csv')

UCI_2010_2014.head()

UCI_2015_2017 = merged_final_pollution.copy()

UCI_2015_2017 = UCI_2015_2017.drop(['year', 'month', 'day', 'hour'], axis=1, inplace=False)

UCI_2015_2017 = UCI_2015_2017[['pollution','dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']]

def preprocess_parsed_col(df, column='wnd_dir'):
    '''
    Redo parsing for wind direction
    '''
    df[column] = df[column].apply(lambda x: wind_categories(x))
    return df

def wind_categories(x):
    x = int(x)
    if x >= 0 and x <= 90:
        # Angular degrees from True north
        y = 'NE'
    if x > 90 and x <= 180:
        y = 'SE'
    if x > 180 and x <=270:
        y = 'SW'
    if x > 270 and x <=360:
        y = 'NW'
    return y

UCI_2015_2017 = preprocess_parsed_col(UCI_2015_2017)

merged_final_pollution.head()

UCI_2015_2017.head()

UCI_2010_2017 = pd.concat((UCI_2010_2014, UCI_2015_2017), axis=0)

UCI_2010_2017 = UCI_2010_2017.reset_index()

#UCI_2010_2017.drop(['level_0','index'], axis=1, inplace=True)

U

len(UCI_2010_2017)

UCI_2010_2017

UCI_2015_2017.to_csv('merged_final_UCI_format.csv')

len(UCI_2010_2017)

# Let's just drop snow upfront:
data_no_snow = UCI_2010_2017.drop('snow', inplace=False, axis=1)

def cast_float_col(df, column='dew'):
    '''
    Redo parsing for dew
    '''
    df[column] = df[column].apply(lambda x: float(x))
    return df

def fix_snow_values(df, column='snow'):
    df[column] = df[column].apply(lambda x: 0 if x == ' ' else x)
    df[column] = df[column].apply(lambda x: 0 if x in ['O','R','E'] else x)
    df[column] = df[column].apply(lambda x: 0 if type(x) != str else int(x))
    return df

def what_type(df, col='snow'):
    return df[col].apply(lambda x: type(x))

snow_fixed_2010_2017 = fix_snow_values(UCI_2010_2017, column='snow')

cast_float_col(snow_fixed_2010_2017, column='snow').head()

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import pandas as pd
import time

#UCI_2015_2017.to_csv('merged_final_UCI_format.csv')

data = data_no_snow

values = data.values
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

####### Can change t_input timesteps here ##########  ### I changed it to time lag = 4
reframed = series_to_supervised(values, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[-6,-5,-4,-3,-2,-1]], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values
#values = scaled
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
scaled_features = scaler.fit_transform(values[:,:-1])
scaled_label = scaler.fit_transform(values[:,-1].reshape(-1,1))
values = np.column_stack((scaled_features, scaled_label))

n_train_hours = 365 * 24 + (365 * 48)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
# features take all values except the var1
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

from sklearn.svm import SVR

x = train_X
y = train_y

regr = SVR(C = 2.0, epsilon = 0.1, kernel = 'rbf', gamma = 0.5, 
           tol = 0.001, verbose=False, shrinking=True, max_iter = 10000)

regr.fit(x, y)
data_pred = regr.predict(x)
y_pred = scaler.inverse_transform(data_pred.reshape(-1,1))
y_inv = scaler.inverse_transform(y.reshape(-1,1))

mse = mean_squared_error(y_inv, y_pred)
rmse = np.sqrt(mse)
print('Mean Squared Error: {:.4f}'.format(mse))
print('Root Mean Squared Error: {:.4f}'.format(rmse))

print('Variance score: {:2f}'.format(r2_score(y_inv, y_pred)))

def plot_preds_actual(preds, actual):
    fig, ax = plt.subplots(figsize=(17,8))
    ax.plot(preds, color='red', label='Predicted data')
    ax.plot(actual, color='green', label='True data')
    ax.set_xlabel('Hourly Timestep in First Month of Predicted Year', fontsize=16)
    ax.set_ylabel('Pollution [pm2.5]', fontsize=16)
    ax.set_title('Nonlinear Regression using SVR on Test set', fontsize=16)
    ax.legend()
    plt.show()

plot_preds_actual(y_pred[:24*31*1,], y_inv[:24*31*1,])

def run_test_nonlinear_reg(x, y):
    data_pred = regr.predict(x)
    y_pred = scaler.inverse_transform(data_pred.reshape(-1,1))
    y_inv = scaler.inverse_transform(y.reshape(-1,1))

    mse = mean_squared_error(y_inv, y_pred)
    rmse = np.sqrt(mse)
    print('Mean Squared Error: {:.4f}'.format(mse))
    print('Root Mean Squared Error: {:.4f}'.format(rmse))

    #Calculate R^2 (regression score function)
    #print('Variance score: %.2f' % r2_score(y, data_pred))
    print('Variance score: {:2f}'.format(r2_score(y_inv, y_pred)))
    return y_pred, y_inv

y_pred, y_inv = run_test_nonlinear_reg(test_X, test_y)

plot_preds_actual(y_pred[:24*31*1,], y_inv[:24*31*1,])





