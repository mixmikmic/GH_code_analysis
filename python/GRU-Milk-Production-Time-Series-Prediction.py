# Import Dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Read in the Dataset
data = pd.read_csv('./dataset/monthly-milk-production.csv', index_col='Month')

# Analyze Dataset
data.head()

# Preprocess the Data
data.index = pd.to_datetime(data.index)

# Data after Preprocessing
data.head()

data.describe()

# Plot Dataset
data.plot()

data.info()

# Train Test Split
# Total Number of Rows: 168
# All except last 12: 168-12 = 156
train_data = data.head(156)
print(train_data)

test_data = data.tail(12)
print(test_data)

# Scale Data
scl = MinMaxScaler()
scaled_train = scl.fit_transform(train_data)
scaled_test = scl.transform(test_data)

print('Scaled Training Data: \n',scaled_train)

print('Scaled Test Data: \n',scaled_test)

# Batch Function to Input Data in Batches
def next_batch(training_data, batch_size,steps):
    # Grab a random starting point for each point of data
    # Generate some random values
    rand_start = np.random.randint(0,len(training_data)-steps)
    # Y Batch
    y_batch = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)
    return y_batch[:,:-1].reshape(-1,steps,1), y_batch[:,1:].reshape(-1,steps,1)

# RNN Model

# Define Constants
# Input : X
num_inputs = 1

# Number Time Steps
num_time_steps = 12

# Neurons in Hidden Layer
num_neurons = 100

# Number of Outputs Required
num_outputs = 1

# Learning Rate
lr = 0.001

# Number of Training Iterations
num_iters = 6000

# Batch Size
batch_size = 1

# Placeholders
# Features: X
# Shape = [Batch Size, Number of Time Steps i.e. No. of RNN Units, Number of Inputs]
#       = [1, 30, 1] i.e. 30 values as input into 30 RNN units.
X = tf.placeholder(tf.float32, shape=[None, num_time_steps, num_inputs])

# Labels
y = tf.placeholder(tf.float32, shape=[None, num_time_steps, num_outputs])

# Create RNN Cell Layer
# Since, we want only one output and not 100 using 100 neurons in hidden layer, we wrap the rnn model into output wrapper
gru_model = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.GRUCell(num_units=num_neurons, activation=tf.nn.relu), output_size=num_outputs)

# Get Output of RNN Cell
# Performs dynamic unrolling of RNN cells
outputs, states = tf.nn.dynamic_rnn(gru_model, X, dtype=tf.float32)

# Loss Function: MSE
loss = tf.reduce_mean(tf.square(outputs-y))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

init = tf.global_variables_initializer()

# Save Trained Model
save_model = tf.train.Saver()

# Run RNN in Session
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(num_iters):
        X_batch, y_batch = next_batch(scaled_train, batch_size, num_time_steps)
        sess.run(optimizer, feed_dict={X: X_batch,y:y_batch})
        
        if i%100 == 0:
            mse = loss.eval(feed_dict={X:X_batch, y:y_batch})
            print('STEP: {0}, ERROR: {1}'.format(i, mse))
    print('STEP: {0}, ERROR: {1}'.format(i+1, mse))
    save_model.save(sess,'./Trained-Model/gru_time_series_model')

# Making Predictions on Test Data: Generative Session
# Predict Time Series into the Future
with tf.Session() as sess:
    save_model.restore(sess,'./Trained-Model/gru_time_series_model')
    train_seed = list(scaled_train[-12:])
    
    for i in range(12):
        X_batch = np.array(train_seed[-num_time_steps:]).reshape(1,num_time_steps,1)
        y_pred = sess.run(outputs, feed_dict={X:X_batch})
        train_seed.append(y_pred[0,-1,0])

# Show Result of Predictions
train_seed

# Grab the last 12 generated values from train_seed and de-normalize them to get the actual milk production values
results = scl.inverse_transform(np.array(train_seed[12:]).reshape(12,1))

test_data['Generated Values'] = results

test_data

test_data.plot()

