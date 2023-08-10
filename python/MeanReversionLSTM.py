import quandl
from datetime import datetime, timedelta
from pandas import Series, concat, DataFrame
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

def get_stock(ticker, start, end, time_range):
    #Retrive Stock Data Using Quandl
    stock = quandl.get('WIKI/{}'.format(ticker), start_date=start, end_date=end)
    #Keep Only Adjusted Closing Price And Adjusted Volume
    stock = stock.iloc[:,10:12]
    #Resample For Given Time Period
    stock = stock.resample('{}'.format(time_range)).mean()
    stock.dropna(inplace=True)
    return stock

def preprocess(data):
    #Transform Data and Calculate Percent Change
    data = np.log(data)
    percent_changes = np.empty(data.shape)
    for i in range(1, len(data)):
        percent_changes[i,:] = (data[i,:] - data[i-1,:])/data[i-1,:]
    return percent_changes

def series_to_supervised(data, n_in, n_out):
    #Convert The Data To A Supervised Learning Format
    cols = []
    df = DataFrame(data)
    for i in range(0, n_in+n_out):
        cols.append(df.shift(-i))
    data = concat(cols, axis=1)
    data.dropna(inplace=True)
    return data.values

def scale(train,test):
    #Scale The Data Using MinMax
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

def forecast_lstm(model, batch_size, X):
    #Forecasts The Average Price Of The Next Time Period
    X = X.reshape(1, len(X), 1)
    prediction = model.predict(X, batch_size=batch_size)
    return prediction[0,0]

def inverse_scale(predict, X, scaler):
    #Inverse Scales The Data
    new_row = [x for x in X] + [predict]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inv_scale = scaler.inverse_transform(array)
    return inv_scale[0,-1]

def inverse_preprocess(orig_data, predict, previous):
    #Inverse Percent Change And Transformation To Obtain The Actual Value
    inv_change = (predict + 1)*np.log(orig_data[-previous,0])
    return np.exp(inv_change)

#Stock Ticker
ticker = 'FMC'
#Timeframe of stock price data to r
train_start = datetime(2010,1,1)
train_end = datetime(2015,12,31)
test_end = datetime(2017,10,19)
#Time range ['M', 'W', 'D', 'A', 'Q']
time_range = 'D'
#Name of lstm model to be saved
model_name = 'model_10_15_D.h5'
#Number of time instances to test on
test_inst = 12
#Number of past stock prices and volumes to use to predict future prices
num_prev = 4
#Number of future stock prices to predict
#Keep at 1 for now, script not updated yet
num_pred = 1
#Batch Size
batch_size = 1

print('Retrieving Stock Data...')
data = get_stock(ticker, train_start, test_end, time_range)
train, test = data[:train_end-timedelta(days=test_inst)].values, data[train_end+timedelta(days=test_inst+num_prev):].values

print('Preprocessing and Formatting Data...')
train_preprocessed = preprocess(train)
test_preprocessed = preprocess(test)

train_formatted = series_to_supervised(train_preprocessed, num_prev, num_pred)
test_formatted = series_to_supervised(test_preprocessed, num_prev, num_pred)
#Drop Last Volume Row As We Are Only Predicting Stock Price
train_formatted = train_formatted[:,:-1]
test_formatted = test_formatted[:,:-1]
scaler, train_scaled, test_scaled = scale(train_formatted, test_formatted)

print('Fitting and Testing Model...')
error_scores = []
model = load_model(model_name)
predictions = []
for j in range(len(test_scaled)):
    X = test_scaled[j,0:-1]
    pred = forecast_lstm(model, batch_size, X)
    pred_invScale = inverse_scale(pred, X, scaler)
    pred_actual = inverse_preprocess(data.values, pred_invScale,len(test_scaled)+1-j)
    predictions.append(pred_actual)

results = DataFrame()
results['Diffs'] = test[num_prev:,0] - np.array(predictions)

plt.hist(results['Diffs'],20)
plt.show()

sns.kdeplot(results['Diffs'])

results.describe()

#Calculate Percent of the time the past Market price moves in the 
#same direction to the next Market Price and to the next LSTM price
direction = DataFrame()
#Change from between past Market Price and next LSTM Price
direction['lstm'] = np.array(predictions)[1:] - test[num_prev:-1,0]
#Change from between past Market Price and next Market Price
direction['market'] = np.diff(test[num_prev:,0])
#Positive means they moved in same direction, negative opposite direction
direction['product'] = direction['market']*direction['lstm']
#All the times they moved in same direction
same_direction = direction[direction['product']>=0]
percent = 100*len(same_direction)/len(direction)
print('Percent of Market Price and LSTM Price Moving in the same Direction: {}'.format(round(percent,3)))

