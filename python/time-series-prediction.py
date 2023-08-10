get_ipython().magic('matplotlib inline')

import numpy as np
import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

dataframe = pandas.read_csv('international-airline-passengers.csv',
                          usecols=[1], engine='python')
type(dataframe)

plt.plot(dataset)
plt.show()

np.random.seed(7)
dataset = dataframe.values.astype(np.float32)
type(dataset)

def generate_data(dataset, look_back=1, test_split=0.25):
    x = []
    y = []
    for i in range(look_back, len(dataset) - 1):
        x.append(dataset[i - look_back:i, 0])
        y.append(dataset[i, 0])
    split_index = int(len(x) * (1. - test_split))
    return (np.array(x[:split_index]), np.array(y[:split_index])), (np.array(x[split_index:]), np.array(y[split_index:]))

look_back = 4
(X_train, Y_train), (X_test, Y_test) = generate_data(dataset, look_back)
for i in range(0, 5):
    print("%s -> %d" % (X_train[i], Y_train[i]))

print("Train shape: %s -> %s" % (X_train.shape, Y_train.shape))
print("Test shape: %s -> %s" % (X_test.shape, Y_test.shape))

model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', accuracy=['metrics'])
print(model.summary())

model.fit(X_train, Y_train, nb_epoch=200, batch_size=2, verbose=1)

score = model.evaluate(X_test, Y_test)
print("Score: %1.4f" % score)

trainedPrediction = model.predict(X_train)
testedPrediction = model.predict(X_test)
plt.plot(dataset)
plt.plot(trainedPrediction)
plt.plot(np.concatenate((np.zeros_like(X_train), testedPrediction)))
plt.show()



