import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
np.random.seed(0)

# Number of samples
n = 100

# Choose n random numbers for x and y
x = np.random.rand(n)
y = np.random.rand(n)

# Create an array of [x,y] scaled:
# We scale the data because neural networks perform better when all inputs are
# in a similar value range
data = preprocessing.scale(np.stack([x,y], axis=1))

# Create z.  We reshape it to an array of 1-element arrays for pyBrain
target = (np.sin(x) + 2*y).reshape(n,1)

# Create train/test split
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=.25, random_state=1
)

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer

network = buildNetwork(2, 5, 1, hiddenclass=TanhLayer)

from pybrain.datasets.classification import SupervisedDataSet

# Create a dataset with 2 inputs and 1 output
ds_train = SupervisedDataSet(2,1)

# add our data to the dataset
ds_train.setField('input', data_train)
ds_train.setField('target', target_train)

# Do the same for the test set
ds_test = SupervisedDataSet(2,1)
ds_test.setField('input', data_test)
ds_test.setField('target', target_test)

from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation import Validator

# Create a trainer for the network and training dataset
trainer = BackpropTrainer(network, ds_train)

# Train for a number of epochs and report accuracy:
for i in range(10):
    # Train 10 epochs
    trainer.trainEpochs(10)
    
    # Report mean squared error for training and testing sets
    # `network.activateOnDataset` will return the predicted values for each input in the dataset passed to it.
    # Then `Validator.MSE` returns the mean squared error of the returned value with the actual value.
    print("Train MSE:", Validator.MSE(network.activateOnDataset(ds_train), target_train))
    print("Test MSE:", Validator.MSE(network.activateOnDataset(ds_test), target_test))

from sklearn.datasets import load_digits

# Load all the samples for all digits 0-9
digits = load_digits()

# Assign the matrices to a variable `data`
data = digits.data

# Assign the labels to a variable `target`
target = digits.target.reshape((len(digits.target), 1))

# Split the data into 75% train, 25% test
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=.25, random_state=0
)

from pybrain.structure import SoftmaxLayer

# Create a network with 64 inputs, 2 layers of 16 hidden units and 10 outputs (for each digit 0-9)
network = buildNetwork(data.shape[1], 16, 16, 10, hiddenclass=SoftmaxLayer)

from pybrain.datasets.classification import ClassificationDataSet

# Create a dataset with 64 inputs
ds_train = ClassificationDataSet(data_train.shape[1])

# Set the input data
ds_train.setField('input', data_train)
ds_train.setField('target', target_train)

# The convert to dummy variables
# For instance, this will represent 0 as (1,0,0,0,0,0,0,0,0,0)
# 1 as (0,1,0,0,0,0,0,0,0,0), 4 as (0,0,0,0,1,0,0,0,0,0) and so on.
ds_train._convertToOneOfMany()

# Do the same for the test set
ds_test = ClassificationDataSet(data_test.shape[1])
ds_test.setField('input', data_test)
ds_test.setField('target', target_test)
ds_test._convertToOneOfMany()

from sklearn.metrics import accuracy_score

trainer = BackpropTrainer(network, ds_train)

for i in range(10):
    trainer.trainEpochs(50)
    
    # We use `argmax(1)` to convert the results back from the 10 outputs to a single label
    print("Training Accuracy:", accuracy_score(target_train, network.activateOnDataset(ds_train).argmax(1)))
    print("Testing Accuracy:", accuracy_score(target_test, network.activateOnDataset(ds_test).argmax(1)))



