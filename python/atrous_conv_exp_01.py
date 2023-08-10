import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import keras.layers
from keras.layers import Conv1D, Activation, Dense, Input, Flatten, Add, Concatenate, Multiply
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers
from statsmodels.tsa.stattools import adfuller

import plotly as py
import plotly.graph_objs as go
import cufflinks as cf

cf.set_config_file(offline=True, world_readable=False, theme='ggplot')

df = pd.read_csv("power_data.txt")
df.columns = ['power']
df.index = pd.date_range('1/1/1997', periods=35039, freq='15Min')
date_range = pd.date_range('1/1/1997', periods=35039, freq='15Min')

print(len(df))
df.iplot(y='power')

df = df.resample('1H').mean()
print(len(df))
df.head()

def seasonal_decomposition(x, period):
    """Extracts the seasonal components of the signal x, according to period"""
    num_period = len(x) // period
    assert(num_period > 0)
    
    x_trunc = x[:num_period*period].reshape((num_period, period))
    x_season = np.mean(x_trunc, axis=0)
    x_season = np.concatenate((np.tile(x_season, num_period), x_season[:len(x) % period]))
    return x_season

def automatic_seasonality_remover(x, k_components=10, verbose=True):
    """Extracts the most likely seasonal component via FFT"""
    f_x = np.fft.rfft(x - np.mean(x))
    f_x = np.real(f_x * f_x.conj())
    periods = len(x) / f_x.argsort()[-k_components:][::-1]
    periods = np.rint(periods).astype(int)
    min_error = None
    best_period = 0
    best_season = np.zeros(len(x))
    
    for period in periods:
        if period == len(x): continue
        x_season = seasonal_decomposition(x, period)
        error = np.average((x - x_season)**2)
        if verbose:
            print("Testing period: {}. Error: {}".format(period, error))
        
        if min_error is None or error < min_error:
            min_error = error
            best_season = x_season
            best_period = period
    
    if verbose:
        print("Best fit period: {}".format(best_period))
    return best_season

df['season'] = automatic_seasonality_remover(df['power'].values)
df['deseason'] = df['power'] - df['season']

#df.iplot(online=False)

input_size = 512
num_filters = 16
num_output_filter = 12
in_channels = 1 #1 channel input. May want to consider quantizing input?
use_skip = True

def wavenet_model(look_ahead=16):
    def residual_block(x, nb_filters, dilation_rate):
        original_x = x
        tanh_out = Conv1D(nb_filters, 2, dilation_rate=dilation_rate, padding='causal', 
                          activation='tanh')(x) #, kernel_regularizer=regularizers.l2(0.05)
        sigm_out = Conv1D(nb_filters, 2, dilation_rate=dilation_rate, padding='causal', 
                          activation='sigmoid')(x) #, kernel_regularizer=regularizers.l2(0.05)

        x = Multiply()([tanh_out, sigm_out])

        res_x = Conv1D(nb_filters, 1, padding='same')(x)
        skip_x = Conv1D(nb_filters, 1, padding='same')(x)

        return res_x, skip_x

    inps = Input((input_size, in_channels))
    #Layer 1
    x = Conv1D(num_filters, 2, padding='causal', name='layer_1_causal_conv')(inps)
    skip_connections = []
    #We construct such that the final output's receptive field is equivalent to the input
    layers = int(np.log2(input_size//2) - 1)
    for layer in range(layers):
        x, skip_x = residual_block(x, num_filters, 2**(layer+1))
        skip_connections.append(skip_x)

    if use_skip:
        x = Add()(skip_connections)

    x = Activation('relu')(x)
    x = Conv1D(num_output_filter, 1, activation='relu')(x)
    x = Conv1D(num_output_filter//2, 1)(x)

    #We take last output for classification
    x = keras.layers.Lambda(lambda x: x[:, -1:, :], name='extraction_layer')(x)

    #Perform regression
    x = Dense(look_ahead * in_channels)(x)
    x = keras.layers.Reshape((look_ahead, in_channels))(x)
    model = Model(inputs=inps, outputs=x)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model

def reshape_and_replicate(x, look_back, look_ahead):
    """Reshapes and replicate data into format convenient for training
    x: Shape (t_steps, channels)
    look_back: Number of time-steps network depends on
    look_ahead: Number of time-steps network attempts to predict ahead
    
    Output:
    X_in: History with size (t_len - look_ahead - look_back, look_back, channels)
    X_out: Prediction target with size (t_len - look_ahead - look_back, look_ahead, channels)
    """
    t_len = x.shape[0]
    channels = x.shape[1]
    X_input = np.zeros((t_len - look_ahead - look_back + 1, look_back, channels))
    X_out = np.zeros((t_len - look_ahead - look_back + 1, look_ahead, channels))
    for i in range(len(X_input)):
        X_input[i, :] = x[i:i+look_back, :].reshape(1, look_back, channels)
        X_out[i, :] = x[i+look_back:i+look_back+look_ahead, :].reshape(1, look_ahead, channels)
        
    return X_input, X_out
    

def time_series_anomaly(x_raw, look_ahead=16, look_back=512, epochs=20):
    
    """Perform anomaly detection on time-series x
    x: np.array of dimension (, n_dim), 0-th index is time,
    and 1st index is for multi_dimension input"""
    single_dim = False
    if len(x_raw.shape) == 1:  # 1 dimensional time-series
        in_channels = 1
        x_raw = x_raw.reshape(-1, 1)
        single_dim = True
    elif len(x_raw.shape) == 2:  # Multi-dim time-series
        in_channels = x_raw.shape[1]
    else:
        raise(Exception("Unexpected input shape. Expected 1 or 2 dimension. Received: {}".format(len(x_raw.shape))))
    t_len = x_raw.shape[0]
    x_raw = x_raw.astype(np.float32)
    
    #Perform normalization
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    x = scaler.fit_transform(x_raw)
    
    model = wavenet_model(look_ahead=look_ahead)
    pred = np.zeros((t_len - look_back - look_ahead, look_ahead, in_channels))
    for i in range(t_len - look_back - look_ahead): #Number of prediction steps
        if i % 10 == 0:
            print("Iter %d of %d" % (i, t_len-look_back-look_ahead), end="\r")
        #Generate history and output prediction
        X_input, X_out = reshape_and_replicate(x[i:look_back+look_ahead+i, :], look_back, look_ahead)
        model.fit(X_input, X_out, verbose=0)
        
        #Now predict the steps ahead
        pred[i, :, :] = model.predict(x[i+look_ahead:i+look_ahead+look_back, :].reshape(1,look_back, in_channels))
        #Sanity check -- uncomment to check that reshaping below is correct
        #try:
        #    pred[i, :, :] = x[i+look_ahead+look_back:i+(look_ahead*2)+look_back, :].reshape(1,look_ahead, in_channels)
        #except:
        #    continue
    
    
    #Predict
    #pred = model.predict(X_input)
    pred = scaler.inverse_transform(pred)    
    
    #Correc the time-frames
    pred_full = np.full((t_len + look_ahead, look_ahead, in_channels), np.nan, dtype=np.float32)
    for i in range(look_ahead):  # Shift the time-frames forward
        pred_full[look_back+look_ahead+i:t_len+i, i, :] = pred[:, i, :]
    #Truncate the extra predictions
    pred_full = pred_full[:t_len , :, :]
    
    error = pred_full - x_raw.reshape((-1, 1, in_channels))
    pred_mean = np.nanmean(pred_full, axis=1)
    pred_dev = np.std(pred_full, axis=1)
    if single_dim:
        pred_full = pred_full.squeeze()
        error = error.squeeze()
        pred_mean = pred_mean.squeeze()
        
    output = {'pred_full': pred_full,
              'pred': pred_mean,
              'error': error,
              'conf': pred_dev,
              }
    return output

output = time_series_anomaly(df.power.values, look_ahead=8, look_back=input_size, epochs=1)

def plot_timeseries_with_bound(df, val_col, bound_col, true_col, xTitle='', yTitle='',
                       title='', filename='output.html'):
    cf.set_config_file(offline=True, world_readable=True, theme='ggplot')
#    cf_output = df.iplot(kind='scatter', title=title, xTitle=xTitle,
#                         online=False, asFigure=True)
    
    upper_bound = go.Scatter(
        name='Upper Bound',
        #x=df['Time'],
        y=df[val_col]+df[bound_col],
        mode='lines',
        marker=dict(color="444"),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty' )

    trace = go.Scatter(
        name='Measurement',
        #x=df['Time'],
        y=df[val_col],
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty' )
    
    lower_bound = go.Scatter(
        name='Lower Bound',
        #x=df['Time'],
        y=df[val_col]-df[bound_col],
        marker=dict(color="444"),
        line=dict(width=0),
        mode='lines' )
    
    true_line = go.Scatter(
        name='Actual',
        #x=df['Time'],
        y=df[true_col],
        mode='lines',
        line=dict(color='rgb(180, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='lines' )
    
    error = go.Scatter(
        name='Error',
        #x=df['Time'],
        y= (df[true_col] - df[val_col])/df[bound_col],
        mode='lines',
        line=dict(color='rgb(200, 100, 100)'),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='lines' ) 
    # Trace order can be important
    # with continuous error bars
    data = [lower_bound, trace, upper_bound, true_line, error]
    
    layout = go.Layout(
        yaxis=dict(title=yTitle),
        title=title,
        showlegend = True)
    fig = go.Figure(data=data, layout=layout)
    
    py.offline.iplot(fig, filename=filename) 

df['pred'] = output['pred']
df['error'] = df['power'] - df['pred']
df['conf'] = output['conf']
df['score'] = (df['error'] / df['conf'])*20


#plot_timeseries_with_bound(df, 'pred', 'conf', 'power', title="Prediction")
df[['power', 'pred', 'error', 'score']].iplot()

