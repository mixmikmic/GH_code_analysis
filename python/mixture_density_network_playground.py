import matplotlib.pyplot as plt
import numpy as np
import math

NSAMPLE = 1000
x_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
noise_data = np.float32(np.random.normal(size=(NSAMPLE,1)))
# y_data = np.float32(np.sin(0.75*x_data)*7.0+x_data*0.5+noise_data*1.0)

y_data = np.float32(np.sin(x_data)+0.2*noise_data)

plt.figure(figsize=(8, 8))
plot_out = plt.plot(x_data,y_data,'ro',alpha=0.3)
plt.show()

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(20, input_dim=1, activation='tanh'))
model.add(Dense(1, activation=None))

model.compile(loss='mean_squared_error', optimizer='sgd')

history = model.fit(x_data, y_data, validation_split=0.3, epochs=15000, verbose=1)
print ('model trained!')

x_test = np.float32(np.arange(-10.5,10.5,0.1))
x_test = x_test.reshape(x_test.size,1)
y_test = model.predict(x_test)

plt.figure(figsize=(8, 8))
plot_out = plt.plot(x_data,y_data,'ro',
                   x_test,y_test,'bo',alpha=0.3)
plt.show()

model_inv = Sequential()
model_inv.add(Dense(20, input_dim=1, activation='tanh'))
model_inv.add(Dense(1, activation=None))

model_inv.compile(loss='mean_squared_error', optimizer='sgd')

model.fit(y_data, x_data, epochs=1000, verbose=0)
print ('model trained!')

x_test = np.float32(np.arange(-10.5,10.5,0.1))
x_test = np.float32(np.sin(x_data)+0.2*noise_data)
x_test = x_test.reshape(x_test.size,1)
y_test = model.predict(x_test)

plt.figure(figsize=(8, 8))
plot_out = plt.plot(y_data,x_data,'ro',
                   x_test,y_test,'bo',alpha=0.3)
plt.show()



