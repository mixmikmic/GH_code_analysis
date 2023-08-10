import numpy 
import matplotlib.pyplot as plt
import pandas 
from keras.models import Sequential
from keras.layers import Dense

# fix random seed for reproducibility
numpy.random.seed(7)

datapath = '/home/sherlock/Documents/testlib/international-airline-passengers.csv'

# load the dataset
dataframe = pandas.read_csv(datapath, usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

dataset[0:5]

dataset.shape

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

dataset[1:3,0]

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1,look_forward=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        b = dataset[(i+look_back):(i+look_back+look_forward),0]
        dataX.append(a)
        dataY.append(b)
    return numpy.array(dataX), numpy.array(dataY)

# reshape into X=t and Y=t+1
look_back = 4
look_forward =2
trainX, trainY = create_dataset(train, look_back,look_forward)
testX, testY = create_dataset(test, look_back,look_forward)

trainX.shape

# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(4))
model.add(Dense(2))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=200, batch_size=2, verbose=2)

import math
import scipy

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict.shape

trainPredict

trainPredict[:,0]

trainPredict1 =trainPredict[:,0]
trainPredict11 = trainPredict1.reshape(91,1)
trainPredict11.shape

trainPredict2 =trainPredict[:,1]
trainPredict12 = trainPredict2.reshape(91,1)
trainPredict12.shape

# shift train predictions for plotting
trainPredictPlot1 = numpy.empty_like(dataset)
trainPredictPlot1[:, :] = numpy.nan
trainPredictPlot1[look_back:len(trainPredict)+look_back, :] = trainPredict11

#shift train t+2 predictions for plotting
trainPredictPlot2 = numpy.empty_like(dataset)
trainPredictPlot2[:, :] = numpy.nan
trainPredictPlot2[look_back:len(trainPredict)+look_back, :] = trainPredict12



testPredict.shape

testPredict1 =testPredict[:,0]
testPredict11 = testPredict1.reshape(43,1)
testPredict2 =testPredict[:,1]
testPredict12 = testPredict2.reshape(43,1)

# shift test predictions for plotting
testPredictPlot1 = numpy.empty_like(dataset)
testPredictPlot1[:, :] = numpy.nan
testPredictPlot1[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict11

# shift test predictions for plotting
testPredictPlot2 = numpy.empty_like(dataset)
testPredictPlot2[:, :] = numpy.nan
testPredictPlot2[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict12


# plot baseline and predictions
plt.plot(dataset)
plt.plot(trainPredictPlot1)
plt.plot(trainPredictPlot2)
plt.plot(testPredictPlot1)
plt.plot(testPredictPlot2)
plt.show()



