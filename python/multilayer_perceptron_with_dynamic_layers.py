from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import data
from utils.dataset import create_handwritten_dataset

train_data, train_labels, test_data, test_labels, label_map = create_handwritten_dataset(
        "/home/ashok/Data/Datasets/devanagari-character-dataset/nhcd/numerals", test_ratio=0.2)

n_classes = len(label_map)
image_size = (28, 28)
image_channel = 1
n_train_samples = len(train_labels)
n_test_samples = len(test_labels)

print("Classes: {}, Label map: {}".format(n_classes, label_map))
print("Train samples: {}, Test samples: {}".format(n_train_samples, n_test_samples))

# Parameters
learning_rate = 0.001
training_epochs = 200
batch_size = 100
display_step = 1

# Network Parameters
hidden_layer_neurons=[1024,512]
n_input = image_size[0]*image_size[1]*image_channel# 28*25*1=784

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input], name="x")
y = tf.placeholder(tf.float32, [None, n_classes], name="y")

# Create model
def layer(inputs, n_neurons, activation="linear", name=None):
    """
    activation: [relu, sigmoid, tenh, linear]
    """
    with tf.variable_scope(name):
        n_inputs = int(inputs.get_shape()[1])
        w = tf.Variable(tf.random_normal([n_inputs, n_neurons]), name=name+"_w")
        b = tf.Variable(tf.random_normal([n_neurons]), name=name+"_b")
        output = tf.add(tf.matmul(inputs, w), b)
        
        if activation =="relu":
            ouput = tf.nn.relu(output)
        elif activation == "sigmoid":
            ouput = tf.nn.sigmoid(output)
        elif activation == "tenh":
            ouput = tf.nn.tenh(output)
        elif activation == "linear":
            pass
        else:
            raise("Unknown activation, {}".format(activation))
            
        return output
    
def multilayer_perceptron(inputs, hidden_layer_neurons, n_classes):
    if hidden_layer_neurons is None or len(hidden_layer_neurons) ==0: #Perceptron
        raise("Please provide hidden layers. hidden_layer_neurons = [N,O,P,...]")
    net = inputs
    for i in range(len(hidden_layer_neurons)):
        net = layer(net, hidden_layer_neurons[i], activation="relu", name="hidden_layer"+str(i+1))
    net = layer(net, n_classes, "linear", "output_layer")
    return net
    
# Construct model
pred = multilayer_perceptron(x, hidden_layer_neurons,  n_classes)

# Predictions
pred_probas = tf.nn.softmax(pred,name="pred_prob")
pred_classes = tf.argmax(pred, axis=1, name="pred_class")

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
acc =  tf.reduce_mean(tf.cast(correct, tf.float32))

# Summary
tf.summary.scalar("cost", cost)
tf.summary.scalar("accuracy", acc)
summary_op = tf.summary.merge_all()

# Initializing the variables
init = tf.global_variables_initializer()

# Start training

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.InteractiveSession(config=config)
sess.run(init)
summary_writer = tf.summary.FileWriter('/tmp/mlp', sess.graph)

# Training cycle
global_step = 0
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_train_samples/batch_size)
    if n_train_samples % batch_size != 0: # samller last batch
        total_batch += 1

    # Loop over all batches
    for i in range(total_batch):
        start = i*batch_size
        end = start+batch_size
        if end > n_train_samples:
            end = n_train_samples-1
        batch_x = train_data[start:end]
        batch_y = train_labels[start:end]
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c, summary = sess.run([optimizer, cost, summary_op], feed_dict={x: batch_x,
                                                      y: batch_y})
        # Compute average loss
        avg_cost += c / total_batch
        summary_writer.add_summary(summary, global_step)
        global_step += 1
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch: {:04d}, cost = {:.9f}".format(epoch+1, avg_cost))
        
print("Optimization Finished!")

# Testing
train_acc = sess.run(acc, feed_dict={x:train_data, y: train_labels})
test_acc = sess.run(acc, feed_dict={x:test_data, y: test_labels})

print("Train Accuracy: {:.2f}%".format(train_acc*100))
print("Test Accuracy: {:.2f}%".format(test_acc*100))
summary_writer.close()

# Inference
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import imread, imshow, imshow_array, imresize, normalize_array, im2bw, pil2array, rgb2gray

image = imread('/home/ashok/Projects/ml-for-all-github/data/five.png')

imshow(image)

if image.size != image_size:
    image = imresize(image, image_size)
    
image = rgb2gray(image)
image  = pil2array(image)
image = normalize_array(image)
image = np.reshape(image, (image_size[0]*image_size[1]*image_channel))
image  = np.reshape(np.asarray(image), image_size[0]*image_size[1]*image_channel)

output  = sess.run(pred_probas, feed_dict={x:[image]})
output_label = np.argmax(output)

print('Output label: {}, score: {:.2f}%'.format(output_label, output[0][output_label]*100))



