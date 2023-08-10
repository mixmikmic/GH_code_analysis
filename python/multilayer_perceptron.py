from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import os
import sys
tf.set_random_seed(1234)
np.random.seed(1234)
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
batch_size = 128
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # NHCD data input (img shape: 28*28)

# tf Graph input
x = tf.placeholder("float", [None, n_input], name="x")
y = tf.placeholder("float", [None, n_classes], name="y")

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
   
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Predictions
pred_probas = tf.nn.softmax(pred)
pred_classes = tf.argmax(pred, axis=1)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
acc =  tf.reduce_mean(tf.cast(correct, tf.float32))
    
# Initializing the variables
init = tf.global_variables_initializer()

# Start training

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.InteractiveSession(config=config)

sess.run(init)

# Training cycle
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
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                      y: batch_y})
        # Compute average loss
        avg_cost += c / total_batch
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch: {:04d}, cost = {:.9f}".format(epoch+1, avg_cost))
        
print("Optimization Finished!")

# Testing
train_acc = sess.run(acc, feed_dict={x:train_data, y: train_labels})
test_acc = sess.run(acc, feed_dict={x:test_data, y: test_labels})

print("Train Accuracy: {:.2f}%".format(train_acc*100))
print("Test Accuracy: {:.2f}%".format(test_acc*100))

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



