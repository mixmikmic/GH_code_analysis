get_ipython().magic('matplotlib inline')

# LSTM for international airline passengers problem with regression framing
import numpy as np
import matplotlib.pyplot as plt
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

# fix random seed for reproducibility
np.random.seed(7)

# load the dataset
dataframe = pandas.read_csv('~/Downloads/international-airline-passengers.csv', 
                            usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

print(np.min(dataset), np.max(dataset), np.mean(dataset), np.median(dataset))

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

print('trainX.shape=',trainX.shape)

print('X', 'Y')
for (x,y) in zip(trainX[:10], trainY[:10]):
    print(x, y)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print('trainX.shape=',trainX.shape)


print('X', 'Y')
for (x,y) in zip(trainX[:10], trainY[:10]):
    print(x, y)

def buildModel(input_dim=1):
    model = Sequential()
    model.add(LSTM(4, input_dim=input_dim))
    model.add(Dense(1))
    
    return model

# create and fit the LSTM network
model = buildModel(look_back)
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score:', scaler.inverse_transform(np.array([[trainScore]])))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score:', scaler.inverse_transform(np.array([[testScore]])))

def plotPrediction(look_back, model, trainX, testX, dataset, batch_size=32):
    # generate predictions for training
    trainPredict = model.predict(trainX, batch_size=batch_size)
    testPredict = model.predict(testX, batch_size=batch_size)
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(dataset)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
    

plotPrediction(look_back, model, trainX, testX, dataset)


trainPredict = model.predict(trainX)
print('X', 'targY', 'predY')
for (x,ty,py) in zip(scaler.inverse_transform(trainX[:10, 0, 0]), 
                     scaler.inverse_transform(trainY[:10]), 
                     scaler.inverse_transform(trainPredict[:10, 0])):
    print(x, ty, py)

np.random.seed(7)

look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print('X', 'Y')
for (x,y) in zip(trainX[:10], trainY[:10]):
    print(x, y)

# create and fit the LSTM network
model = buildModel(look_back)
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=0)

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score:', scaler.inverse_transform(np.array([[trainScore]])))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score:', scaler.inverse_transform(np.array([[testScore]])))

plotPrediction(look_back, model, trainX, testX, dataset)


#trainPredict = model.predict(trainX)
#print('X', 'targY', 'predY')
#for (x,ty,py) in zip(trainX[:10,0], trainY[:10], trainPredict[:10]):
#    print(x, ty,py)

np.random.seed(7)

look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

print('X', 'Y')
for (x,y) in zip(trainX[:10], trainY[:10]):
    print(x, y)

# create and fit the LSTM network
model = buildModel(1)
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=0)

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score:', scaler.inverse_transform(np.array([[trainScore]])))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score:', scaler.inverse_transform(np.array([[testScore]])))

plotPrediction(look_back, model, trainX, testX, dataset)

#trainPredict = model.predict(trainX)
#print('X', 'targY', 'predY')
#for (x,ty,py) in zip(trainX[:10,0], trainY[:10], trainPredict[:10]):
#    print(x, ty,py)

def buildStatefulModel(look_back=1, batch_size=1):
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), 
                   stateful=True))
    model.add(Dense(1))
    
    return model

np.random.seed(7)

# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

batch_size = 1

model = buildStatefulModel(look_back, batch_size)
model.compile(loss='mean_squared_error', optimizer='adam')
for i in range(100):
    model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, 
              verbose=0, shuffle=False)
    model.reset_states()

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, batch_size=batch_size, verbose=0)
print('Train Score:', scaler.inverse_transform(np.array([[trainScore]])))
testScore = model.evaluate(testX, testY, batch_size=batch_size, verbose=0)
print('Test Score:', scaler.inverse_transform(np.array([[testScore]])))

plotPrediction(look_back, model, trainX, testX, dataset, batch_size)

#trainPredict = model.predict(trainX)
#print('X', 'targY', 'predY')
#for (x,ty,py) in zip(trainX[:10,0], trainY[:10], trainPredict[:10]):
#    print(x, ty,py)

from keras.layers import Dropout

def buildStackedStatefulModel(look_back=1, batch_size=1):
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), 
                   stateful=True, return_sequences=True, activation='tanh', dropout_W=0., dropout_U=0.))
    model.add(LSTM(2, stateful=True, activation='tanh'))
    model.add(Dense(1, activation='relu'))
    
    return model

np.random.seed(8)

# reshape into X=t and Y=t+1
look_back = 4
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

batch_size = 1
epoch = 200

model = buildStackedStatefulModel(look_back, batch_size)
model.compile(loss='mean_squared_error', optimizer='adam')

for i in range(epoch):
    model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, 
              verbose=2 if i%(epoch / 10)==0 else 0, shuffle=False)
    
    model.reset_states()
    
    if i%(epoch / 10)==0:
        trainScore = model.evaluate(trainX, trainY, batch_size=batch_size, verbose=0)
        print('Train Score:', scaler.inverse_transform(np.array([[trainScore]])))
        testScore = model.evaluate(testX, testY, batch_size=batch_size, verbose=0)
        print('Test Score:', scaler.inverse_transform(np.array([[testScore]])))
        
    model.reset_states()
    
model.reset_states()
# Estimate model performance
trainScore = model.evaluate(trainX, trainY, batch_size=batch_size, verbose=0)
print('Train Score:', scaler.inverse_transform(np.array([[trainScore]])))
testScore = model.evaluate(testX, testY, batch_size=batch_size, verbose=0)
print('Test Score:', scaler.inverse_transform(np.array([[testScore]])))

model.reset_states()
plt.rcParams['figure.figsize'] = (20.0, 15.0)
plotPrediction(look_back, model, trainX, testX, dataset, batch_size)

#trainPredict = model.predict(trainX)
#print('X', 'targY', 'predY')
#for (x,ty,py) in zip(trainX[:10,0], trainY[:10], trainPredict[:10]):
#    print(x, ty,py)



