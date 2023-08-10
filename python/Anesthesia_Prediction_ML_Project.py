import pandas as pd
from pandas import DataFrame
from pandas import concat
import numpy as np
from os import listdir

# Function to load in data
def load_data(casenum):
    # Create a pandas dataframe
    df = pd.DataFrame()
    
    # The desired features as mentioned above
    cols = [3, 6, 8, 13, 23, 24, 29, 36]
    
    # Convert the case number to a string and pad zeros if needed
    casenum = str(casenum).zfill(2)

    # List of files for a case number
    files = [f for f in listdir('Cases/case' + casenum + '/fulldata')]
    
    # Append files into one dataframe
    for f in files:
        # Read in file and append
        df = df.append(pd.read_csv('Cases/case' + casenum + '/fulldata/' + f,index_col=False, usecols=cols))
        
        # Replace NaN values with zero
        df = df.fillna(0)
    
    # Only take every 100th sample
    X = np.array(df)[::100,:]
    return X, df

# Desired case to analyze
casenum = 1

# Load in data
X, df = load_data(casenum)

print('The shape of full data frame: {0}'.format(df.shape))
print('The shape of downsampled data: {0}'.format(X.shape))

# Show first 5 rows of data frame before downsampling
df.head()

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Plot the inhaled and exhaled sevoflurane
t = np.arange(X.shape[0]) / 60 / 60  # Time in hours
plt.plot(t, X[:,5], label = 'inSEV')
plt.plot(t, X[:,4], label = 'etSEV')
plt.xlabel('Time (hours)')
plt.title('Case {0}'.format(casenum))
plt.legend()

from sklearn.preprocessing import MinMaxScaler 

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequency (t, t+1, ... t+n)
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

    
# Ensure all data is float
X = X.astype('float32')

# Normalize features
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(X)

# Frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[8,9,10,11,13,14,15]], axis = 1, inplace = True)

print('The new shape of the data is: {0}'.format(reframed.shape))

# First five rows of new data matrix
reframed.head()

# Get values from the dataframe
values = reframed.values

# Only first half hour (in seconds)
ntrain = 1800

# Split data into train and test sets
train = values[:ntrain, :]
test = values[ntrain:, :]

# Split data into input and output
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# Reshape input to be 3D [samples, timesteps, features]
train_X_3D = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X_3D = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X_3D.shape, train_y.shape, test_X_3D.shape, test_y.shape )

import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, LSTM

import keras.backend as K
K.clear_session()

# Design Network
model = Sequential()
model.add(LSTM(100, input_shape=(train_X_3D.shape[1], train_X_3D.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Print Model Summary
model.summary()

# Fit Network
history = model.fit(train_X_3D, train_y, epochs=50, batch_size=10, validation_data=(test_X_3D, test_y), verbose=2, shuffle=False)

# Plot History
plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()

# Evaluate Model
yhat = model.predict(test_X_3D)

# Match shape of test y to yhat
test_y = test_y.reshape((len(test_y), 1))

# Calculate normalized RSS
RSS_test_normal = np.mean((yhat-test_y)**2)/(np.std(test_y)**2)

print('Normalized RSS test: {0}'.format(RSS_test_normal))
print('R^2: {0}'.format(1 - RSS_test_normal))

# Time vector
t = np.arange(values.shape[0])[ntrain:] / 60 / 60 # in hours

# Plot the actual and predicted sevoflurane
plt.plot(t, test_y, label = 'Actual')
plt.plot(t, yhat, label = 'Predicted')
plt.xlabel('Test Time (Hours)')
plt.title('Case: {0}'.format(casenum))
plt.legend(loc = 'lower left')

cases = [3, 4, 5]
plt.figure(figsize=(18, 4))

for i, case in enumerate(cases):
    # Load in, scale, and shift the data
    X, df = load_data(case)
    X = X.astype('float32')
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(X)
    reframed = series_to_supervised(scaled, 1, 1)
    reframed.drop(reframed.columns[[8,9,10,11,13,14,15]], axis = 1, inplace = True)
    
    # Get values from the dataframe
    values = reframed.values

    # Only first half hour (in seconds)
    ntrain = 1800
    
    # Split data into train and test sets
    train = values[:ntrain, :]
    test = values[ntrain:, :]

    # Split data input and output
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    
    # Reshape input to be 3D [samples, timesteps, features]
    train_X_3D = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X_3D = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    
    # Clear Keras session
    K.clear_session()

    # Design Network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X_3D.shape[1], train_X_3D.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    # Fit Network
    history = model.fit(train_X_3D, train_y, epochs=50, batch_size=10, validation_data=(test_X_3D, test_y), verbose=0, shuffle=False)

    # Make a prediction
    yhat = model.predict(test_X_3D)

    # Match shape of test y to yhat
    test_y = test_y.reshape((len(test_y), 1))
    
    # Calculate normalized RSS
    RSS_test_normal = np.mean((yhat-test_y)**2)/(np.std(test_y)**2)
    print('Case {0}: RSS = {1}, R^2 = {2}'.format(case , RSS_test_normal, 1-RSS_test_normal ))
    
    # Time vector
    t = np.arange(values.shape[0])[ntrain:] / 60 / 60 # in hours
    
    # Plot actual and predicted sevoflurane
    plt.subplot(1,3,i+1)
    plt.plot(t, test_y,label='Actual')
    plt.plot(t, yhat,label='Predicted')
    plt.legend(loc = 'lower left')
    plt.xlabel('Test Time (Hours)')
    plt.title('Case: {0}'.format(case))

# Replace the train and test X matrices to be the original 2 dimensional arrays
train_X = train[:, :-1]
test_X = test[:, :-1]

from sklearn import linear_model

# Create LinearRegression class and fit model with training data
regr = linear_model.LinearRegression()
regr.fit(train_X,train_y)

# Predict output based on test data
yhat_linear = regr.predict(test_X)

# Match shape of yhat to test y
yhat_linear = yhat_linear.reshape((len(yhat_linear), 1))

# Time vector
t = np.arange(1, X.shape[0])[ntrain:] / 60 / 60 # in hours
    
# Plot the actual and predicted sevoflurane
plt.plot(t, test_y,label='Actual')
plt.plot(t, yhat_linear,label='Predicted')
plt.xlabel('Test Time (Hours)')
plt.title('Case: {0}'.format(casenum))
plt.legend(loc = 'lower left')

# Calculate normalized RSS
RSS_test_normal = np.mean((yhat_linear-test_y)**2)/(np.std(test_y)**2)

print('Normalized RSS test: {0}'.format(RSS_test_normal))
print('R^2: {0}'.format(1 - RSS_test_normal))

cases = [3, 4, 5]
plt.figure(figsize=(18, 4))

for i, case in enumerate(cases):
    # Load in, scale, and shift the data
    X, df = load_data(case)
    X = X.astype('float32')
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(X)
    reframed = series_to_supervised(scaled, 1, 1)
    reframed.drop(reframed.columns[[8,9,10,11,13,14,15]], axis = 1, inplace = True)
    
    # Get values from the dataframe
    values = reframed.values

    # Only first half hour (in seconds)
    ntrain = 1800

    # Split data into train and test sets
    train = values[:ntrain, :]
    test = values[ntrain:, :]

    # Split data input and output
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    
    # Create LinearRegression class and fit model with training data
    regr = linear_model.LinearRegression()
    regr.fit(train_X,train_y)
    
    # Predict output based on test data
    yhat_linear = regr.predict(test_X)

    # Calculate normalized RSS
    RSS_test_normal = np.mean((yhat_linear-test_y)**2)/(np.std(test_y)**2)
    print('Case {0}: RSS = {1}, R^2 = {2}'.format(case , RSS_test_normal, 1-RSS_test_normal ))
    
    # Time vector
    t = np.arange(1, X.shape[0])[ntrain:] / 60 / 60 # in hours
    
    # Plot the actual and predicted sevoflurane
    plt.subplot(1,3,i+1)
    plt.plot(t, test_y,label='Actual')
    plt.plot(t, yhat_linear,label='Predicted')
    plt.legend(loc = 'lower left')
    plt.xlabel('Test Time (Hours)')
    plt.title('Case: {0}'.format(case))

