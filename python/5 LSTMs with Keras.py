from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

from statsmodels.tsa.arima_process import arma_generate_sample
xs = arma_generate_sample([1.0, -0.6, 1.0, -0.6], [1.0], 2001, 1.0, burnin=100) / 50
plt.plot(xs)

x = xs[:-1]; y = xs[1:]

def nmse(y1, y2):
    return np.linalg.norm(y1 - y2)**2 / np.linalg.norm(y2)**2

nmse(x, y)

xtrain = x.reshape(len(x), 1, 1)
ytrain = y.reshape(len(y), 1)

bs = 1
model = Sequential()
model.add(LSTM(10, batch_input_shape=(bs, 1, 1), stateful=True))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

for i in range(3):
    model.fit(xtrain, ytrain, epochs=1, batch_size=bs, shuffle=False)
    model.reset_states()

model.reset_states()

yhat = model.predict(xtrain, batch_size=bs)

start=1800; stop=2000
plt.figure(figsize=(15,10))
plt.plot(yhat[start:stop], '-', ytrain[start:stop], 'r-')

nmse(yhat, ytrain)



