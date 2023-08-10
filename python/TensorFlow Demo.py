## initializing necessary libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

input_value_one = tf.constant(42)
input_value_two = tf.constant(12)

multiply_inputs = tf.multiply(input_value_one, input_value_two)

sess = tf.Session()

output_value = sess.run(multiply_inputs)

print(output_value)

## Importing working data
dataset = pd.read_csv('test_data//minerals_verge_market.csv')
working_data = dataset.loc[dataset['type_name'] == 'Tritanium', ['record_date', 'avg_price']]
working_data = working_data.join(
    working_data['avg_price'].shift(1),
    rsuffix = '_prior'
)
working_data = working_data.loc[np.all(pd.notnull(working_data), axis = 1), :]
print(working_data)
feature_cols = ['avg_price_prior']
label_cols = ['avg_price']

input_ph = tf.placeholder(shape = (None, len(feature_cols)), dtype = tf.float32)

feed = {
    input_ph: working_data[feature_cols].values
}

weights = tf.constant(0.5, shape = (1, 1))
graph = tf.matmul(input_ph, weights)
result = sess.run(graph, feed_dict = feed)
print(result)

## Creating the weights and biases for the model, weights randomly initialized, biases initialized to zero
weights = tf.Variable(tf.random_normal(shape = (len(feature_cols), len(label_cols))))
bias = tf.Variable(tf.zeros(shape = (len(label_cols),)))

## Don't need to add a new placeholder, since we already have one that will work in input_ph
label_ph = tf.placeholder(shape = (None, len(label_cols)), dtype = tf.float32)

## Creating the linear regression model, and it's adjoining loss function
lin_reg = tf.add(tf.matmul(input_ph, weights), bias)
loss = tf.reduce_sum(tf.square(tf.subtract(label_ph, lin_reg)))

## Creating the optimizer, and setting its learning rate
learn_rate = 1e-3
optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)

## Creating the feed dictionary for the features & labels
feed = {
    input_ph: working_data[feature_cols].values,
    label_ph: working_data[label_cols].values
}

## Initializing variables prior to running model
sess.run(tf.global_variables_initializer())

print(sess.run([weights, bias, loss], feed_dict = feed))

sess.run(optimizer, feed_dict = feed)
print(sess.run([weights, bias, loss], feed_dict = feed))

loss_record = []
iters = 100

for i in range(iters):
    _, iter_loss = sess.run([optimizer, loss], feed_dict = feed)
    loss_record.append(iter_loss)
    
plt.plot(loss_record)

print(sess.run([weights, bias])) ## Note, no feed dictionary!

working_data.insert(len(working_data.columns), 'avg_price_predict', sess.run(lin_reg, feed_dict = feed))
print(working_data)

iters = 10000

for i in range(iters):
    _, iter_loss = sess.run([optimizer, loss], feed_dict = feed)
    loss_record.append(iter_loss)
    
plt.plot(loss_record)

working_data['avg_price_predict'] = sess.run(lin_reg, feed_dict = feed)
print(working_data)

def tf_sig(inp):
    return tf.divide(1., tf.add(1., tf.exp(tf.negative(inp))))

sig = tf.divide(1., tf.add(1., tf.exp(tf.negative(lin_reg))))
sess.run(sig, feed_dict = feed)

nn_layer_1_weights = tf.Variable(tf.random_normal(shape = (len(feature_cols), 4)))
nn_layer_1_bias = tf.Variable(tf.zeros(shape = (4,)))

nn_layer_2_weights = tf.Variable(tf.random_normal(shape = (4, 3)))
nn_layer_2_bias = tf.Variable(tf.zeros(shape = (3,)))

nn_layer_3_weights = tf.Variable(tf.random_normal(shape = (3, 1)))
nn_layer_3_bias = tf.Variable(tf.zeros(shape = (1,)))

nn_graph_layer_1 = tf_sig(tf.add(tf.matmul(input_ph, nn_layer_1_weights), nn_layer_1_bias))
nn_graph_layer_2 = tf.add(tf.matmul(nn_graph_layer_1, nn_layer_2_weights), nn_layer_2_bias)
nn_graph_layer_3 = tf.add(tf.matmul(nn_graph_layer_2, nn_layer_3_weights), nn_layer_3_bias)

sess.run(tf.global_variables_initializer())

sess.run(nn_graph_layer_3, feed_dict = feed)

loss = tf.reduce_sum(tf.square(tf.subtract(label_ph, nn_graph_layer_3)))

learn_rate = 1e-3
optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)

feed = {
    input_ph: working_data[feature_cols].values,
    label_ph: working_data[label_cols].values
}

loss_record = []
iters = 1000
sess.run(tf.global_variables_initializer())

for i in range(iters):
    _, iter_loss = sess.run([optimizer, loss], feed_dict = feed)
    loss_record.append(iter_loss)
    
plt.plot(loss_record)

sess.run(nn_graph_layer_3, feed_dict = feed)

