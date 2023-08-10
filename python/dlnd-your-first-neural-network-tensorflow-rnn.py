get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

seq_len = 5

data_path = 'Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)

rides.head()

rides[:24*10].plot(x='dteday', y='cnt')

dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
data.head()

quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std

# Save the last 21 days 
test_data = data[-21*24:]
data = data[:-21*24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

def load_data(data, seq_len):
    result = []
    
    sequence_length = seq_len + 1
    
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    return result

# Hold out the last 60 days of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]

train_x_rnn = load_data(train_features.values, seq_len)

train_y_rnn = load_data(targets.values, seq_len)

train_x_rnn[0]



class NeuralNetwork(object):
    
    def modelNL(self):
        
        self.x_tensor = tf.placeholder(tf.float32, [None, val_features.shape[1]], name='x')
        self.y_tensor = tf.placeholder(tf.float32, [None, 1], name='y')
        
        lstm = tf.contrib.rnn.BasicLSTMCell(val_features.shape[1])

        # Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        ### Run the data through the RNN layers
        # Run each sequence step through the RNN and collect the outputs
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state)
        final_state = state

        # Reshape output so it's a bunch of rows, one output row for each step for each batch
        seq_output = tf.concat(outputs, axis=1)
        output = tf.reshape(seq_output, [-1, lstm_size])
        
        output = tf.layers.dense(inputs=self.x_tensor, units=100)
        
        output = tf.layers.dense(inputs=output, units=100)
        
        output = tf.layers.dense(inputs=output, units=100, activation=tf.nn.sigmoid)
        
        pred = tf.layers.dense(inputs=output, units=1)
        
        self.pred = tf.identity(pred, name='pred')
        
        #self.cost = tf.reduce_mean(tf.pow(tf.subtract(self.pred, self.y_tensor), 2))
        self.cost = tf.reduce_mean(tf.square(self.pred - self.y_tensor))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        
        return tf.Graph()
    
    def __init__(self,learning_rate, timestep):
        self.timestep = timestep
        self.lr = learning_rate
        self.sess = tf.Session()
        self.modelNL()
        self.sess.run(tf.global_variables_initializer())
    
    def train(self, X, Y):
        feed = {self.x_tensor: X, self.y_tensor: Y}
        loss, _ = self.sess.run([self.cost, self.optimizer], feed_dict=feed)
        return loss
    
    def loss(self, X, Y):
        feed = {self.x_tensor: X, self.y_tensor: Y}
        loss = self.sess.run([self.cost], feed_dict=feed)
        return loss

    def run_raw(self, X):
        feed = {self.x_tensor: X}
        predictions = self.sess.run(self.pred, feed_dict=feed)
        return predictions
    
    def run(self, X):
        feed = {self.x_tensor: X.values}
        predictions = self.sess.run(self.pred, feed_dict=feed)
        return predictions

def MSE(y, Y):
    return np.mean((y-Y)**2)

import sys

### Set the hyperparameters here ###
epochs = 100
learning_rate = 0.0001
hidden_nodes = 5
output_nodes = 1

N_i = train_features.shape[1]
network = NeuralNetwork(learning_rate)

losses = {'train':[], 'validation':[]}
for e in range(epochs):
    loss = 0
    num = 0
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    for record, target in zip(train_features.ix[batch].values, train_targets.ix[batch]['cnt']):
        num += 1
        batch_loss = network.train([record], [[target]])
        loss += batch_loss
    
    
    print('Epoch {}/{} '.format(e+1, epochs), 'Training loss: {:.4f}'.format(batch_loss))
    
    # Printing out the training progress
    train_loss = network.loss(train_features, np.array(train_targets['cnt'].values).reshape(-1,1))
    val_loss = network.loss(val_features, np.array(val_targets['cnt'].values).reshape(-1,1))
    print("\rProgress: " + str(100 * e/float(epochs))[:4]                      + "% ... Training loss: " + str(train_loss)[:5]                      + " ... Validation loss: " + str(val_loss)[:5])
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

val_features.values[1]

plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()

fig, ax = plt.subplots(figsize=(8,4))

mean, std = scaled_features['cnt']
predictions = network.run(test_features)*std + mean
ax.plot(predictions, label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)

import unittest

inputs = [0.5, -0.2, 0.1]
targets = [0.4]
test_w_i_h = np.array([[0.1, 0.4, -0.3], 
                       [-0.2, 0.5, 0.2]])
test_w_h_o = np.array([[0.3, -0.1]])

class TestMethods(unittest.TestCase):
    
    ##########
    # Unit tests for data loading
    ##########
    
    def test_data_path(self):
        # Test that file path to dataset has been unaltered
        self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')
        
    def test_data_loaded(self):
        # Test that data frame loaded
        self.assertTrue(isinstance(rides, pd.DataFrame))
    
    ##########
    # Unit tests for network functionality
    ##########

    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        # Test that the activation function is a sigmoid
        self.assertTrue(np.all(network.activation_function(0.5) == 1/(1+np.exp(-0.5))))

    def test_train(self):
        # Test that weights are updated correctly on training
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()
        
        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output, 
                                    np.array([[ 0.37275328, -0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[ 0.10562014,  0.39775194, -0.29887597],
                                              [-0.20185996,  0.50074398,  0.19962801]])))

    def test_run(self):
        # Test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))

suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
unittest.TextTestRunner().run(suite)

