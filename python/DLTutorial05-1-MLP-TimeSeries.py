get_ipython().system('ls -al ~/Downloads/')

get_ipython().system('cat ~/Downloads/international-airline-passengers.csv')

get_ipython().magic('matplotlib inline')

import pandas
import matplotlib.pyplot as plt
dataset = pandas.read_csv('~/Downloads/international-airline-passengers.csv', 
                          usecols=[1], engine='python', skipfooter=3)
plt.plot(dataset)
plt.show()

import numpy
import matplotlib.pyplot as plt
import pandas
from keras.models import Sequential
from keras.layers import Dense
# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = pandas.read_csv('~/Downloads/international-airline-passengers.csv', 
                            usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

print('dataset.shape = ', dataset.shape)
print('dataset[0] = ', dataset[0])

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

print('X', 'Y')
for (x,y) in zip(trainX[:10,0], trainY[:10]):
    print(x, y)

def buildModel(look_back):
    model = Sequential()
    model.add(Dense(8, input_dim=look_back, activation='relu'))
    model.add(Dense(1))

    return model

model = buildModel(look_back)
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=200, batch_size=2, verbose=2)

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score:', trainScore)
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score:', testScore)

def plotPrediction(look_back, model, trainX, testX, dataset):
    # generate predictions for training
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(dataset)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
    

plotPrediction(look_back, model, trainX, testX, dataset)

#trainPredict = model.predict(trainX)
#print('X', 'targY', 'predY')
#for (x,ty,py) in zip(trainX[:10,0], trainY[:10], trainPredict[:10]):
#    print(x, ty,py)

# reshape dataset
look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

print('X', 'Y')
for (x,y) in zip(trainX[:10], trainY[:10]):
    print(x, y)

model = buildModel(look_back)
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=200, batch_size=2, verbose=0)

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score:', trainScore)
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score:', testScore)

plotPrediction(look_back, model, trainX, testX, dataset)



