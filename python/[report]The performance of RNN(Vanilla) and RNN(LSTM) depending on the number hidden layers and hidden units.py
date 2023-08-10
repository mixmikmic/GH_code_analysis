import sys
# HY_python_NN absolute directory.
my_absdir = "/Users/hyungwonyang/Google_Drive/Python/HY_python_NN"
sys.path.append(my_absdir)

import numpy as np
import main.setvalues as set
import main.rnnnetworkmodels as net

# import data.
# data directory.
rnn_data = np.load(my_absdir+'/train_data/pg8800_lstm_char_data.npz')
train_input = rnn_data['train_input']
train_output = rnn_data['train_output']
test_input = rnn_data['test_input']
test_output = rnn_data['test_output']

# parameters
problem = 'classification' # classification, regression
rnnCell = 'rnn' # rnn, lstm, gru
trainEpoch = 20
learningRate = 0.001
learningRateDecay = 'off' # on, off
batchSize = 100
dropout = 'on'
hiddenLayers = [200]
timeStep = 20
costFunction = 'adam' # gradient, adam
validationCheck = 'on' # if validationCheck is on, then 20% of train data will be taken for validation.

rnn_values = set.RNNParam(inputData=train_input,
                           targetData=train_output,
                           timeStep=timeStep,
                           hiddenUnits=hiddenLayers
                           )

# Setting hidden layers: weightMatrix and biasMatrix
rnn_weightMatrix = rnn_values.genWeight()
rnn_biasMatrix = rnn_values.genBias()
rnn_input_x,rnn_input_y = rnn_values.genSymbol()

rnn_net = net.RNNModel(inputSymbol=rnn_input_x,
                        outputSymbol=rnn_input_y,
                        rnnCell=rnnCell,
                        problem=problem,
                        hiddenLayer=hiddenLayers,
                        trainEpoch=trainEpoch,
                        learningRate=learningRate,
                        learningRateDecay=learningRateDecay,
                        timeStep=timeStep,
                        batchSize=batchSize,
                        dropout=dropout,
                        validationCheck=validationCheck,
                        weightMatrix=rnn_weightMatrix,
                        biasMatrix=rnn_biasMatrix)

# Generate a RNN(vanilla) network.
rnn_net.genRNN()

# Train the RNN(vanilla) network.
# In this tutorial, we will run only 20 epochs.
rnn_net.trainRNN(train_input,train_output)

# Test the trained RNN(vanilla) network.
rnn_net.testRNN(test_input,test_output)

# Save the trained parameters.
vars = rnn_net.getVariables()
# Terminate the session.
rnn_net.closeRNN()

import numpy as np
import main.setvalues as set
import main.rnnnetworkmodels as net

# import data.
# data directory.
lstm_data = np.load(my_absdir+'/train_data/pg8800_lstm_char_data.npz')
train_input = lstm_data['train_input']
train_output = lstm_data['train_output']
test_input = lstm_data['test_input']
test_output = lstm_data['test_output']

# parameters
problem = 'classification' # classification, regression
rnnCell = 'lstm' # rnn, lstm, gru
trainEpoch = 20
learningRate = 0.001
learningRateDecay = 'off' # on, off
batchSize = 100
dropout = 'on'
hiddenLayers = [200]
timeStep = 20
costFunction = 'adam' # gradient, adam
validationCheck = 'on' # if validationCheck is on, then 20% of train data will be taken for validation.

lstm_values = set.RNNParam(inputData=train_input,
                           targetData=train_output,
                           timeStep=timeStep,
                           hiddenUnits=hiddenLayers
                           )

# Setting hidden layers: weightMatrix and biasMatrix
lstm_weightMatrix = lstm_values.genWeight()
lstm_biasMatrix = lstm_values.genBias()
lstm_input_x,lstm_input_y = lstm_values.genSymbol()

lstm_net = net.RNNModel(inputSymbol=lstm_input_x,
                        outputSymbol=lstm_input_y,
                        rnnCell=rnnCell,
                        problem=problem,
                        hiddenLayer=hiddenLayers,
                        trainEpoch=trainEpoch,
                        learningRate=learningRate,
                        learningRateDecay=learningRateDecay,
                        timeStep=timeStep,
                        batchSize=batchSize,
                        dropout=dropout,
                        validationCheck=validationCheck,
                        weightMatrix=lstm_weightMatrix,
                        biasMatrix=lstm_biasMatrix)

# Generate a RNN(lstm) network.
lstm_net.genRNN()

# Train the RNN(lstm) network.
# In this tutorial, we will run only 20 epochs.
lstm_net.trainRNN(train_input,train_output)

# Test the trained RNN(lstm) network.
lstm_net.testRNN(test_input,test_output)

# Save the trained parameters.
vars = lstm_net.getVariables()
# Terminate the session.
lstm_net.closeRNN()



