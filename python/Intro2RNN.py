from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from io import StringIO
import requests
import numpy as np
import pandas as pd
import time, math

np.set_printoptions(precision=4)

ticker = 'AAPL'

r = requests.get("https://finance.google.com/finance/historical?q=" + ticker + "&startdate=01-Jan-2008&output=csv")
stock = pd.read_csv(StringIO(r.text))

stock.head()

stock.drop('Date', axis=1, inplace=True)

cols = stock.columns.tolist()
cols = cols[-1:] + cols[:-1]
stock = stock[cols]

stock.head()

# normalizaing data
scale = MinMaxScaler(feature_range=(0,1)) # or StandardScaler
#scale = StandardScaler()
price = MinMaxScaler(feature_range=(0,1))
price.fit(stock['Close'].reshape(-1,1))
stock = pd.DataFrame(scale.fit_transform(stock), columns=['Volume', 
                                                          'Open', 
                                                          'High', 
                                                          'Low', 
                                                          'Close'])

scalers = {}
prices = {}

def load_data(stock, seq_len, split):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix()
    sequence_length = seq_len + 1
    result = []
    
    
#    print (len(range(len(data) - sequence_length)))
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
#         scalers[index] = MinMaxScaler(feature_range=(0,1))
#         prices[index] = MinMaxScaler(feature_range=(0,1))
#         prices[index].fit_transform(data[index: index 
#                                          + sequence_length][:,-1].reshape(-1,1))
#         result.append(scalers[index].fit_transform(data[index: index 
#                                                         + sequence_length]))
    
    result = np.array(result)
    row = len(result) * split
#     print (result)#[:,:-1])
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]
    
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  
    
    return [x_train, y_train, x_test, y_test]

import matplotlib.pyplot as plt

plt.plot(stock['Close'])#[::-1].reset_index()['Close'])

plt.show()

window = 30 # Another hyperparameter

X_train, y_train, X_test, y_test = load_data(stock[::-1], window, 0.85)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)

def build_model(layers):
    model = Sequential()

    for x in range(0,1):
        model.add(LSTM(input_dim=layers[0], output_dim=layers[1], return_sequences=True))
        model.add(Dropout(0.2))

    model.add(LSTM(layers[2], return_sequences=False)) 
    model.add(Dropout(0.2))

    model.add(Dense(output_dim=layers[2]))
    model.add(Activation("relu"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

model = build_model([5, window, 1])

model.fit(X_train, y_train, batch_size=512, epochs=100, validation_split=0.1, verbose=1)

trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

pred = model.predict(X_test)

plt.figure(figsize=(15,4), dpi=100)

#for x in np.arange(len(X_train))
results = []
pred_results = []
counter = 0
for x in np.arange(2160,2507,1):
    results = results + [prices[x].inverse_transform(y_test[counter:counter + 30].reshape(-1,1))[0]]
    pred_results = pred_results + [prices[x].inverse_transform(pred[counter:counter + 30].reshape(-1,1))[0]]
    counter = counter+1
plt.plot(np.arange(2160, len(stock) - 31, 1), pred_results, color='red', label='predicted price')
#plt.plot(np.arange(2160, len(stock) - 31, 1), results, color='blue', label='actual closing price')
plt.plot(np.arange(2160, len(stock) - 31, 1), stock['Close'][::-1].reset_index()['Close'][-347:], label='actual price')

plt.xlabel('number of days where 01-Jan-2008 is 0 ')
plt.ylabel('price per stock')
plt.legend(loc='upper left')
plt.show()



