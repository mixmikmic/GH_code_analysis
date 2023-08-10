# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
df_train = pd.read_csv('apple_stockprice_train.csv')
training_set = df_train.iloc[:, 1:2].values 

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) # normalised between 0 and 1
training_set_scaled = sc.fit_transform(training_set)

timesteps = 120 #in days, experiment with other days unit to tune the accuracy of the model 

X_train = []
y_train = []

for i in range(timesteps, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-timesteps:i, 0]) #(t):memorise the 60 (0 to 59) look-back stock prices
    y_train.append(training_set_scaled[i, 0]) #(t+1): the 61st stock price (60)
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) # batch_size, timesteps, input_dim

df_test = pd.read_csv('apple_stockprice_test.csv')
real_stock_price = df_test.iloc[:, 1:2].values
df_total = pd.concat((df_train['Open'], df_test['Open']), axis = 0) #vertical concatenation 
inputs = df_total[len(df_total) - len(df_test) - timesteps:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)  
X_test = []
for i in range(timesteps, len(inputs)):
    X_test.append(inputs[i-timesteps:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2)) 

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

from keras.callbacks import History #, EarlyStopping
histories = History()
regressor.fit(X_train, y_train, validation_split = 0.3, epochs = 100, batch_size = 32, callbacks = [histories])

plt.plot(histories.history['loss'])
plt.plot(histories.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test(validation)'], loc='upper right')
plt.show()

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'red', label = 'Real Apple Inc. Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Apple Inc. Price')
plt.title('Apple Inc. Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Apple Inc. Stock Price')
plt.legend()
plt.tight_layout()
plt.savefig('timesteps_{}.png'.format(timesteps))
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
rmse

from IPython.display import Image
Image(filename='timesteps_60.png') 

from IPython.display import Image
Image(filename='timesteps_120.png') 



