import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
from pytz import timezone

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dropout
from keras.utils.np_utils import to_categorical
from keras import optimizers
import keras
import talib
from fractions import gcd
import time

import pywt
from statsmodels.robust import stand_mad
from sklearn.metrics import accuracy_score

fig_size = (12, 9)
plt.rcParams["figure.figsize"] = fig_size

def get_CU():
    import dovahkiin as dk
    dp = dk.DataParser()
    X = dp.get_data("cu")
    return X

def get_SP500():
    import pandas_datareader as pdr    
    SP500 = pdr.get_data_yahoo('^GSPC')
    return SP500

def get_X_data(time_interval):
    import dovahkiin as dk
    dp = dk.DataParser()
    X = dp.get_data("cu", split_threshold=time_interval)
    return X

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalise_window = []
        if (window[0]==0):
            normalised_window = [0 for p in window]
        else:
            normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(layers, with_regularizer=None, dropout=0.2):
    model = Sequential()
    
    total_lstm_layers = len(layers) - 3 # substract the first (input size), fisrt layer and last layer
    assert(total_lstm_layers > 0)
    
    print("> Using regularizer: {}".format(with_regularizer))

    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=layers[1],
        return_sequences=True,
        kernel_regularizer=with_regularizer,
        recurrent_regularizer=with_regularizer,
        bias_regularizer=with_regularizer,
        dropout=dropout,
        recurrent_dropout=dropout))
    
    for i in range(total_lstm_layers):
        model.add(LSTM(
            layers[i+2],
            return_sequences=(i<total_lstm_layers-1),
            kernel_regularizer=with_regularizer,
            recurrent_regularizer=with_regularizer,
            bias_regularizer=with_regularizer,
            dropout=dropout,
            recurrent_dropout=dropout))

    model.add(Dense(
        output_dim=layers[-1]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def timeseries_to_supervised(raw_time_series, lag):
    p = {}
    for i in range(1, lag+1):
        p["{}".format(i)] = raw_time_series.shift(i).fillna(0)
    p["0"] = raw_time_series
    
    supervised_data = pd.Panel(p)
    return supervised_data

def non_shuffling_train_test_split(X, y, test_size=0.2):
    i = int((1 - test_size) * X.shape[0]) + 1
    X_train, X_test = np.split(X, [i])
    y_train, y_test = np.split(y, [i])
    return X_train, X_test, y_train, y_test

def create_supervised_X(raw_time_series, lag):
    supervised_X = timeseries_to_supervised(raw_time_series, lag)
    swaped_supervised_X = supervised_X.swapaxes(0, 1)
    return swaped_supervised_X

def load_data(X, X_train_begin=2012, X_train_end=2015, normalise_window=True, look_back=4, UseLSTM=True):
    
    #Divide Data into Train and Test
    train_begin_time = "{}".format(X_train_begin)
    train_end_time = "{}".format(X_train_end)
    test_begin_time = "{}".format(X_train_end+1)
    
    X["HLC"] = (X.high + X.low + X.close) / 3

    train_id = len(X[:train_end_time])
    
    prices = X.HLC.reset_index(drop=True).values

    data_sequence = []
    
    for i in range(len(prices)):
        if (i - look_back) < 0:
            data_sequence.append([0 for i in range(look_back+1)])
        else:
            data_sequence.append(prices[i-look_back:i+1])
        
    assert(len(data_sequence) == len(X))
    
    if normalise_window:
        data_sequence = normalise_windows(data_sequence)
    
    data_sequence = np.array(data_sequence)
    
    # the first look_back period doesn't have X_train, y_train
    train_data = data_sequence[look_back+1:train_id,]
    
    np.random.shuffle(train_data)

    X_train = train_data[:, :-1]
    X_test = data_sequence[train_id:, :-1]
    
    print("Test:  {}".format(len(X_test)))
    print("Train: {}".format(len(X_train)))

    y_train = train_data[:, -1]
    y_test  = data_sequence[train_id:, -1]
    
    if UseLSTM:
        # LSTM expects a 3 dimensional vector. In our case, we have only one feature.
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  

    return [X_train, y_train, X_test, y_test]

def denormalize(X, predicted, look_back):
    prices = pd.DataFrame(X["HLC"]["2016":])
    base = X["HLC"].reset_index(drop=True)
    result = []
    for i in range(len(predicted)):
        result.append(base[-len(predicted)-look_back:].values[i] * (predicted[i] + 1))
    predicted_prices = np.array(result)
    prices["Predict"] = predicted_prices
    return prices

def directional_Accuracy(result):
    direction_prediction = (result.pct_change() <= 0)
    result = accuracy_score(direction_prediction["HLC"], direction_prediction["Predict"])
    return result

def WaveletPredictor(time_interval="1D", 
                     LSTM_Stateful=False,
                     regularizer=None,
                     network_structure=[1, 30, 100, 1],
                     look_back=30,
                     training_begin=2012,
                     training_testing_seperation_time=2015,
                     epochs=10,
                     batch_size=32):

    X = get_X_data(time_interval)
    X_train, y_train, X_test, y_test = load_data(X, 
                                                 X_train_begin=training_begin, 
                                                 normalise_window=True, 
                                                 look_back=look_back,
                                                 UseLSTM=True)    
    
    model = build_model(network_structure, with_regularizer=regularizer)
    
    print("> Model Summary: ")
    print(model.summary())

    history = model.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                nb_epoch=epochs,
                validation_split=0.05)
    
    predicted = predict_point_by_point(model, X_test)
    
    title = ""
    
    if LSTM_Stateful:
        title = "{} Stateful LSTM Predictor".format(time_interval)
    else:
        title = "{} Stateless LSTM Predictor".format(time_interval)
        
    if regularizer:
        title += " {}".format(regularizer)
    else:
        title += " without regularizer"
    
    plt.clf()
    print("> Training Results")
    plt.figure(1)
    plt.title("Training Results")
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["train loss", "validation loss"], loc='upper left')
    
    print("> Normalized Prediction vs Actual")
    plt.figure(2)
    plt.title("Normalized {}".format(title))
    plt.plot(y_test)
    plt.plot(predicted)
    plt.legend(["actual", "predicted"], loc='upper left')
    plt.show()

    print("> Prediction vs Actual")
    fig = plt.figure(3)
    result = denormalize(X, predicted, look_back)
    ax = result.plot()
    plt.title(title)
    acc = directional_Accuracy(result)
    ax.legend(["actual", "predicted"], loc='upper left')
    ax.annotate('Accuracy: {}'.format(acc), xy=(0.30, 0.95), xycoords='axes fraction')
    
    plt.show()
    return [model, history, y_test, predicted]

look_back = 32
model, history, actual, predicted = WaveletPredictor(epochs=40, network_structure=[1, look_back, 128, 1], look_back=look_back)



look_back = 32
model, history, actual, predicted = WaveletPredictor(
    time_interval="10T",
    epochs=20, 
    regularizer='l1', 
    network_structure=[1, look_back, 128, 1], 
    look_back=look_back,
    batch_size=512
)

look_back = 4
model, history, actual, predicted = WaveletPredictor(epochs=20, LSTM_Stateful=True, network_structure=[1, look_back, 128, 1], look_back=look_back)

look_back = 100
model, history, actual, predicted = WaveletPredictor(epochs=20, LSTM_Stateful=True, network_structure=[1, look_back, 128, 1], look_back=look_back)

