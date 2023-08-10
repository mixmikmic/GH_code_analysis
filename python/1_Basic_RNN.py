get_ipython().magic('matplotlib inline')
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Global configuration parameters
n_epochs = 20
total_series_length = 50000
truncated_backprop_steps = 15
state_size = 4 
n_classes = 2
echo_step = 3 # Number of steps the input is shifted to the right
batch_size = 5
n_batches = total_series_length//batch_size//truncated_backprop_steps

eta = 0.01 # Learning rate

def generateData():
    """
    Generates training data. The input data is simply a vector of random
    numbers with n_classes classes. The target output is the input shifted 
    by "echo_steps" steps to the right.
    
    Returns:
        x: numpy array of shape (batch_size,-1) filled with random values
        in the range (n_classes)
        
        y: numpy array of shape (batch_size, -1), x shifted "echo_step" to 
        the right
    """

    x = np.array(np.random.choice(n_classes, total_series_length))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))

    return(x, y)

# Create placeholders for the input, target output and state of the network
X_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, truncated_backprop_steps])
y_placeholder = tf.placeholder(dtype=tf.int32, shape=[batch_size, truncated_backprop_steps])
init_state = tf.placeholder(tf.float32, [batch_size, state_size])

# Create variables for weights and biases. To save computation, the two
# weight matrices U and W are concatenated such that we have to perform only
# one matrix multiplication when calculating the state of the recurrent network.
U_W_concat = tf.Variable(np.random.randn(state_size+1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

V = tf.Variable(np.random.randn(state_size, n_classes), dtype=tf.float32)
c = tf.Variable(np.zeros((1,n_classes)), dtype=tf.float32)


# We split the batch data into adjacent time steps by unpacking the columns of
# the input batch into a list. This causes the network to be trained 
# simultaenously on multiple (namely "batch_size") parts of the input time 
# series. We account for this by setting the state of the network ("init_state")
# to have "batch_size" rows.

input_series = tf.unstack(X_placeholder, axis=1)
labels_series = tf.unstack(y_placeholder, axis=1)

# Forward pass
current_state = init_state
state_series = []

for i in input_series:
    inp = tf.reshape(i, [batch_size, 1])
    inp_state_concat = tf.concat([inp, current_state],1)
    new_state = tf.tanh(tf.matmul(inp_state_concat,U_W_concat)+b)
    
    state_series.append(new_state)
    current_state = new_state

# Softmax output layer, compute output for each state
logits = [tf.matmul(state, V)+c for state in state_series]
predictions = [tf.nn.softmax(logit) for logit in logits]

# Compute loss
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for label, logit in zip(labels_series, logits)]
total_loss = tf.reduce_mean(losses)

# Training step
train_step = tf.train.AdamOptimizer(learning_rate=eta).minimize(total_loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    
    training_losses = []
 
    for epoch in range(n_epochs):
        X_gen, y_gen = generateData()
        
        # The initial state of the network is set to zero
        _current_state = np.zeros((batch_size, state_size))
        
        print("")
        print("Epoch: ", epoch)
        
        for batch_number in range(n_batches):
            start_idx = batch_number * truncated_backprop_steps
            end_idx = start_idx + truncated_backprop_steps
            
            batch_x = X_gen[:, start_idx:end_idx]
            batch_y = y_gen[:, start_idx:end_idx]
            
            _total_loss, _train_step, _current_state = sess.run(
            [total_loss, train_step, current_state], 
            feed_dict={
                X_placeholder: batch_x,
                y_placeholder: batch_y,
                init_state: _current_state
            })
            
            training_losses.append(_total_loss)
            
            if batch_number%100 == 0:
                print("Step: ", batch_number, "Loss:", _total_loss)

plt.figure(figsize=(10,8));
plt.plot(training_losses);
plt.xlabel('Number of training iterations');
plt.ylabel('Error');









