from __future__ import absolute_import, division, print_function

"""
Imports
"""
import numpy as np
import tensorflow as tf
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import time, datetime

RANDOM_SEED = 1234
RANDOM_SEED = int(time.time())
num_input_class = 2

def gen_data(size=1000000):
    X = np.array(np.random.choice(num_input_class, size=(size,)))
    X = onehot(X, num_input_class)
#     print('X, gen\t', X.shape)
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i-3, 1] == 1:
            threshold += 0.5
        if X[i-8, 1] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)

# adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length, num_input_class], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i, :] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1), :]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)

def gen_epochs(num_epochs, batch_size, num_steps):
    for i in range(num_epochs):
        yield gen_batch(gen_data(), batch_size, num_steps)

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()
    

def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out


# Global config variables
# batch_size = 200

num_steps = 10 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 200
num_epochs = 1000
learning_rate = 0.01

# num_unit = 6
input_dim = 2
output_dim = 2

from CTRNN import CTRNNModel

def shape_printer(obj, prefix):
    try:
        print(prefix, obj.shape)
    except AttributeError:
        print(prefix, type(obj))
        for o in obj:
            shape_printer(o, prefix + '\t')

def time_str():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-(%H-%M-%S)')

def train_network(model, num_epochs, batch_size=32, num_steps=200, verbose=True, save=False):
    tf.set_random_seed(RANDOM_SEED)
    training_losses = []
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        sess.run(tf.global_variables_initializer())

        print('\tBegin training loop')
        steps = 0
        for idx, epoch in enumerate(gen_epochs(num_epochs, batch_size, num_steps)):
            print('epoch', idx)
            training_loss = 0
            state_tuple = model.zero_state_tuple(batch_size=batch_size)
#             print('state_tuple', type(state_tuple[0]), state_tuple[0].get_shape(), 
#                   state_tuple[1][0].get_shape(), state_tuple[1][0].get_shape())
            for X, Y in epoch:
                steps += 1
#                 print('Just before sess.run')
#                 print('state_tuple', type(state_tuple))
#                 shape_printer(state_tuple, 'tl')
                feed_dict = {model.x:X, model.y:Y, model.init_tuple:state_tuple}
                training_loss_, _, state_tuple, summary = sess.run([
                        model.total_loss, 
                        model.train_op,
                        model.state_tuple,
                        model.TBsummaries
                    ], 
                        feed_dict=feed_dict)
#                 print('Just after sess.run')
                training_loss += training_loss_
                summary_writer.add_summary(summary, steps)

                if steps % 100 == 0 and steps > 0:
                    if verbose:
                        print('Average loss at step', steps,
                             'for last 100 steps: ', training_loss/100.)
                        tau = sess.run(model.tau)
                        print('tau', tau)
                    training_losses.append(training_loss/100.)
                    training_loss = 0                    
#                     break
#                 break
    return training_losses
        

# import CTRNN
logdir = 'logdir/' + time_str()
print('logdir:', logdir)
tf.reset_default_graph()

### WIRING TEST
## Success criteria
# 1) 3l-highCrap and 2l should perform equally well
# 2) 3l-IOCrap should perform very poorly 
## Results: see below

# # 3l-highCrap
# logdir = 'logdir/' + '3l-highCrap'
# taus = [tf.Variable(5, name='tau', dtype=tf.float32, trainable=True), 
#         tf.Variable(5, name='tau', dtype=tf.float32),
#         tf.Variable(50000, name='tau', dtype=tf.float32, trainable=False)]
# num_units = [7, 8, 9]


# 3l-IOCrap
logdir = 'logdir/' + '3l-IOCrap-long'
taus = [tf.Variable(50000, name='tau', dtype=tf.float32, trainable=False), 
        tf.Variable(5, name='tau', dtype=tf.float32),
        tf.Variable(5, name='tau', dtype=tf.float32, trainable=True)]
num_units = [7, 8, 9]


# # 2l
# logdir = 'logdir/' + '2l'
# taus = [tf.Variable(5, name='tau', dtype=tf.float32, trainable=True), 
#         tf.Variable(5, name='tau', dtype=tf.float32)]
# num_units = [7, 8]


# Just playing arround :) 
# taus = [tf.Variable(1, name='tau', dtype=tf.float32, trainable=True), 
#         tf.Variable(2, name='tau', dtype=tf.float32),
#         tf.Variable(2, name='tau', dtype=tf.float32),
#         tf.Variable(2, name='tau', dtype=tf.float32, trainable=True)]
# num_units = [2,2,2,2] 


print('Creating the model\n')
model = CTRNNModel(num_units=num_units, tau=taus, num_steps=num_steps, input_dim=input_dim, output_dim=output_dim, learning_rate=learning_rate)
print('\nTraining:')
loss = train_network(model, num_epochs=num_epochs, batch_size=batch_size, num_steps=num_steps)


print('Terminated!!')

print('No learning:\t', 0.66)
print('3 dep learning:\t', 0.52)
print('8 dep learning:\t', 0.45)
plt.plot(loss)















