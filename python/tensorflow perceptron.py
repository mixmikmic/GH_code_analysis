import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

hidden_weights = 256

w = tf.Variable(tf.random_normal([n_input, hidden_weights]))
b = tf.Variable(tf.random_normal([hidden_weights]))

w2 = tf.Variable(tf.random_normal([hidden_weights, hidden_weights]))
w3 = tf.Variable(tf.random_normal([hidden_weights, n_classes]))

input_layer = tf.add(tf.matmul(x, w), b)
hidden = tf.matmul(input_layer, w2)
perceptron = tf.matmul(hidden, w3)

# Loss & Optimizer

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=perceptron, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# Initialize the variables

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
        
            avg_cost += c / total_batch
            
        if epoch % display_step == 0:
            print("Epoch:", "%04d" % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            
    print("Finished")
    
    
    correct_prediction = tf.equal(tf.argmax(perceptron, 1), tf.argmax(y, 1))
    
    # Accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

classification_x, classification_y = make_classification(1000, n_features=100, n_informative=30)
classification_y = np.expand_dims(classification_y, axis=1)

print(classification_x.shape, classification_y.shape)

train_x, test_x, train_y, test_y = train_test_split(classification_x, classification_y, test_size=0.33)

n_input = classification_x.shape[1]
n_classes = classification_y.shape[1]

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

w = tf.Variable(tf.random_normal([n_input, n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))

perceptron = tf.add(tf.matmul(x, w), b)

perceptron_loss = tf.reduce_mean(tf.maximum(0., -y * tf.add(tf.matmul(x, w), b)))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(perceptron_loss)

init = tf.global_variables_initializer()

training_epochs = 100
learning_rate = 0.001
display_step = 10

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(train_x.shape[0]/batch_size)
        
        for i in range(total_batch):
            start = i * batch_size
            end = start + batch_size
            batch_x = train_x[start:end]
            batch_y = train_y[start:end]

            _, c = sess.run([optimizer, perceptron_loss], feed_dict={x: batch_x, y: batch_y})
        
            avg_cost += c / total_batch
            
        if epoch % display_step == 0:
            print("Epoch:", "%04d" % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            
    print("Finished")
    
    
    correct_prediction = tf.equal(tf.argmax(perceptron, 1), tf.argmax(y, 1))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: test_x, y: test_y}))

margin = 0.1

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

w = tf.Variable(tf.random_normal([n_input, n_classes]))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(train_x.shape[0]/batch_size)
        
        for i in range(total_batch):
            start = i * batch_size
            end = start + batch_size
            batch_x = train_x[start:end]
            batch_y = train_y[start:end]
            
            loss = tf.reduce_mean(tf.maximum(0., -y * tf.matmul(x, w), b))
            is_mistake = tf.less_equal(y * tf.matmul(x, w), margin)
            eta = (margin - (y[is_mistake] * tf.matmul(x[is_mistake], w))) / (tf.matmul(x[is_mistake], tf.transpose(x[is_mistake])) + 1)
            update = tf.assign(w, eta * tf.matmul(x[is_mistake], y[is_mistake]))
    
            _, c = sess.run([update, loss])
            
            avg_cost += c / total_batch
            
        if epoch % display_step == 0:
            print("Epoch:", "%04d" % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            
    print("Finished")
    
    
    correct_prediction = tf.equal(tf.argmax(perceptron, 1), tf.argmax(y, 1))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: test_x, y: test_y}))



