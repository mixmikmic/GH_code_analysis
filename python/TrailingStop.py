import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from matplotlib import pyplot
from math import sqrt
from pytz import timezone

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Conv1D
from keras import optimizers
from keras.utils.np_utils import to_categorical

# Fix the random seed to reproducibility
np.random.seed(7)

import dovahkiin as dk
from dovahkiin.strategy import StrategyBase
from dovahkiin.feature.Amibroker import *
from dovahkiin.feature.StrategyUtility import *
from dovahkiin.OptimizeParam import OptimizeParam


class CrossOver(StrategyBase):

    """
    样本策略，均线交叉
    """

    params = {}
    params["stop_ratio"] = OptimizeParam("stop", 9.7, 0.1, 12, 0.1)
    params["short_ratio"] = OptimizeParam("shortPeriod", 0.19, 0.1, 0.5, 0.01)
    params["longPeriod"] = OptimizeParam("longPeriod", 24, 20, 60, 1)
    params["threshold_multiplier"] = OptimizeParam("threshold_multiplier", 1.1, 0.1, 4, 0.1)
    params["linreg_lookback"] = OptimizeParam("linreg_lookback", 46, 10, 60, 1)
    params["linreg_slope_coeff"] = OptimizeParam("slope coeff", 0.15, 0.05, 0.5, 0.05)
    params["cond3_coeff"] = OptimizeParam("cond3_coeff", 1.75, 1, 4, 0.25)


    def __init__(self, dataframe, params=None):
        super().__init__(dataframe, params)

    def strategy(self):

        """
        策略的逻辑
        """

        recentATR = ATR(self.C, self.H, self.L, 100, False)
        threshold = self.optimize("threshold_multiplier") * recentATR
        linreg_slope_coeff = self.optimize("linreg_slope_coeff")
        linreg_lookback = int(self.optimize("linreg_lookback"))
        long_period = int(self.optimize("longPeriod"))

        short_period = int(self.optimize("short_ratio") * long_period)
        short_line = MA(self.C, short_period)
        long_line = MA(self.C, self.optimize("longPeriod"))
        
        print("short period", short_period)
        print("linreg lookback", linreg_lookback)
        print("long period", long_period)

        close_slope = LinRegSlope(self.C, short_period)
        short_slope = LinRegSlope(short_line, linreg_lookback)

        # Long logic
        bcond1_1 = (self.C > long_line) & (self.C > short_line)
        bcond1_2 = long_line < short_line
        bcond1_3 = abs(short_line - long_line) > threshold
        bcond1 = bcond1_1 & bcond1_2 & bcond1_3
        bcond2 = LinRegSlope(self.C, short_period) > linreg_slope_coeff * self.optimize("cond3_coeff") * recentATR
        bcond3 = short_slope > linreg_slope_coeff * recentATR
        BSIG = bcond1 & bcond2 & bcond3

        # Short logic
        scond1_1 = (self.C < long_line) & (self.C < short_line)
        scond1_2 = long_line > short_line
        scond1_3 = abs(short_line - long_line) > threshold
        scond1 = scond1_1 & scond1_2 & scond1_3
        scond2 = LinRegSlope(self.C, short_period) < (-1) * linreg_slope_coeff * self.optimize("cond3_coeff") * recentATR
        scond3 = short_slope < (-1) * linreg_slope_coeff * recentATR
        SSIG = scond1 & scond2 & scond3

        self.BUY = BSIG
        self.SHORT = SSIG
        
        self.SELL = self.COVER = (self.C==0).astype(int)
        
        sigs = MoveStop(self.C, self.BUY, self.SHORT, self.C==0 , 100)
        return sigs.values    

dp = dk.DataParser()
dataframe = dp.get_data("cu")



strategy = CrossOver(dataframe)

sigs = strategy.strategy();

dataframe["buy"] = strategy.BUY.values.astype(int)
dataframe["short"] = strategy.SHORT.values.astype(int)
y = sigs.values

y = y.astype(int)

del dataframe["open"]
del dataframe["high"]
del dataframe["low"]
del dataframe["volume"]
del dataframe["openint"]

X = dataframe.values

def FullyConnected_Model():
    model = Sequential()
    model.add(Dense(128, input_shape=(3,), activation="relu"))
    model.add(Dense(128, input_shape=(3,), activation="relu"))
    model.add(Dense(128, input_shape=(3,), activation="relu"))
#     model.add(Dense(128, input_shape=(3,), activation="relu"))
#     model.add(Dense(128, input_shape=(3,), activation="relu"))
#     model.add(Dense(128, input_shape=(3,), activation="relu"))
#     model.add(Dense(128, input_shape=(3,), activation="relu"))
#     model.add(Dense(128, input_shape=(3,), activation="relu"))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation="softmax"))
    return model

def non_shuffling_train_test_split(X, y, test_size=0.2):
    i = int((1 - test_size) * X.shape[0]) + 1
    X_train, X_test = np.split(X, [i])
    y_train, y_test = np.split(y, [i])
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = non_shuffling_train_test_split(X, y, test_size=0.3)

y_train_category = to_categorical(y_train, num_classes=3)

y_test_category = to_categorical(y_test, num_classes=3)

adam = optimizers.adam(lr=1e-9)
sgd = optimizers.SGD(lr=1e-9, decay=1e-9, momentum=0.9, nesterov=True, clipnorm=1.)
model = FullyConnected_Model()
# model.compile(loss="mse", optimizer=adam, metrics=["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=[ 'accuracy'])

batch_size = 10

def trainFullyConnected_network():
    num_epochs = 1
    for i in range(num_epochs):
        model.fit(
            np.nan_to_num(X_train),
            np.nan_to_num(y_train_category),
            epochs=10,
            batch_size=batch_size,
            verbose=1,
            shuffle=False,
            validation_split=0.2
        )
        
    return model

model = trainFullyConnected_network()

model.evaluate(X_test, y_test_category)





lag=100
time_series_step = lag

def timeseries_to_supervised(raw_time_series, lag):
    p = {}
    for i in range(1, lag+1):
        p["{}".format(i)] = raw_time_series.shift(i).fillna(0)
    p["0"] = raw_time_series
    
    if type(raw_time_series) is pd.Series:
        supervised_data = pd.DataFrame(p)
        supervised_data = pd.Panel({"0": supervised_data})
        supervised_data = supervised_data.swapaxes(0, 1).swapaxes(1, 2)
    else:
        supervised_data = pd.Panel(p)
    return supervised_data

X = pd.DataFrame(dataframe["close"])
X = timeseries_to_supervised(X, lag=lag)
X = X.swapaxes(0, 1)

supervised_X = X.fillna(0)
supervised_X_values = supervised_X.values

supervised_X_values[635:].shape



y_test_category = to_categorical(y, num_classes=3)

y_test_category[635:].shape

supervised_X_values = supervised_X_values[635:]
y_test_category = y_test_category[635:]

batch_size=1000
features=1

def LSTM_Model(lstm_layers=None, dense_layers=None):
    model = Sequential()
    
    if lstm_layers:
        for i in range(lstm_layers):
            model.add(
                LSTM(128, batch_input_shape=(batch_size, time_series_step+1, features), stateful=True, 
                 return_sequences=True, 
                 activation="relu"))
        model.add(LSTM(32, activation="relu", stateful=True))
    else:
        model.add(
            LSTM(128, batch_input_shape=(batch_size, time_series_step+1, features), stateful=True, 
             return_sequences=True, 
             activation="relu"
            ))
        model.add(LSTM(32, activation="relu", stateful=True))
        
    if dense_layers:
        for i in range(dense_layers):
            model.add(Dense(128, activation="sigmoid"))
        model.add(Dense(3))
    else:
        # model.add(Dense(128))
        model.add(Dense(3))
    
    return model

adam = optimizers.adam(lr=1e-9, clipnorm=1.)

def trainLSTM_network():   
    model = LSTM_Model(6, 0)
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=[ 'accuracy'])
    
    num_epochs = 1
    for i in range(num_epochs):
        model.fit(
            np.nan_to_num(supervised_X_values),
            np.nan_to_num(y_test_category),
            epochs=1,
            batch_size=batch_size,
            verbose=1,
            shuffle=False,
            validation_split=0.2
        )
    return model

supervised_X_values.shape

y.max()

model = trainLSTM_network()





