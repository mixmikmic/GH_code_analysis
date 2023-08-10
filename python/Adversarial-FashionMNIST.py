import itertools 
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import fast_gradient as fg
import saliency_map as sm
import deepfool as df

# We define the characteristics of the input image 
height = 28
width = 28
channels = 1
input_shape = (width, height, channels)
n_inputs = height * width
n_classes = 10

data = input_data.read_data_sets('data/fashion', one_hot=True)

# Define train and test sets
X_train = data.train.images
X_test = data.test.images
y_train = data.train.labels.astype("int")
y_test = data.test.labels.astype("int")

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

X_train = X_train.reshape(-1, width, height, channels)
X_test = X_test.reshape(-1, width, height, channels)

def get_start_end(ind, batch_size, n_sample):
    start = ind*batch_size
    end = min(n_sample, start+batch_size)
    return start, end

# Create dictionary of target classes
label_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot'
}

# Select samples to view
samples_train = [47, 23]
samples_test = [14, 18]

# Plot samples
print("Train samples:")
for s in samples_train:
    sample = data.train.images[s].reshape(28,28)
    sample_label = data.train.labels[s]
    print("y = {label_index} {label_vec} ({label})".format(label_index=np.argmax(sample_label, axis=0), label_vec=sample_label, label=label_dict[np.argmax(sample_label, axis=0)]))
    plt.imshow(sample, cmap='Greys')
    plt.show()
print("Test samples:")
for s in samples_test:
    sample = data.test.images[s].reshape(28,28)
    sample_label = data.test.labels[s]
    print("y = {label_index} {label_vec} ({label})".format(label_index=np.argmax(sample_label, axis=0), label_vec=sample_label, label=label_dict[np.argmax(sample_label, axis=0)]))
    plt.imshow(sample, cmap='Greys')
    plt.show()

# Set to True if you want to use a GPU for computing
UseGPU = False

if UseGPU:
    config = tf.ConfigProto(device_count = {'GPU': 1})
else:
    config = tf.ConfigProto(device_count = {'GPU': 0})

def model(x, logits=False, training=False):
    conv1 = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3], padding='SAME', name='conv1', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2, name='pool1')
    
    conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=[3, 3], padding='SAME', name='conv2', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2, name='pool2')
    
    flat = tf.reshape(pool2, [-1, 7*7*64], name='flatten')
    
    dense1 = tf.layers.dense(flat, units=128, activation=tf.nn.relu, name='dense1')
    dropout1 = tf.layers.dropout(dense1, rate=0.25, training=training, name='dropout1')
    dense2 = tf.layers.dense(dropout1, units=64, activation=tf.nn.relu, name='dense2')
    dropout2 = tf.layers.dropout(dense2, rate=0.30, training=training, name='dropout2')
    
    logits_ = tf.layers.dense(dropout2, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')
    
    if logits:
        return y, logits_
    return y

tf.reset_default_graph()
class Dummy:
    pass
env = Dummy()

with tf.variable_scope('model'):
    env.x = tf.placeholder(tf.float32, (None, width, height, channels), name='x')
    env.y = tf.placeholder(tf.int32, (None, n_classes), name='y')
    env.training = tf.placeholder(bool, (), name='mode')

    env.ybar, logits = model(env.x, logits=True, training=env.training)

    truez = tf.argmax(env.y, axis=1)
    predictedz = tf.argmax(env.ybar, axis=1)
    
    count = tf.cast(tf.equal(truez, predictedz), tf.float32)
    env.acc = tf.reduce_mean(count, name='acc')
    xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y, logits=logits)
    env.loss = tf.reduce_mean(xent, name='loss')
    
    with tf.name_scope("eval"):
        accuracy = tf.reduce_mean(tf.cast(count, tf.float32))
        
    with tf.name_scope("save"):
        saver = tf.train.Saver()
    
env.optimizer = tf.train.AdamOptimizer().minimize(env.loss)

with tf.variable_scope('model', reuse=True):
    env.adv_fgm = fg.fgm(model, env.x)
    env.adv_jsma_t0 = sm.jsma(model, env.x, 0)
    env.adv_jsma_t1 = sm.jsma(model, env.x, 1)
    env.adv_jsma_t2 = sm.jsma(model, env.x, 2)
    env.adv_jsma_t3 = sm.jsma(model, env.x, 3)
    env.adv_jsma_t4 = sm.jsma(model, env.x, 4)
    env.adv_jsma_t5 = sm.jsma(model, env.x, 5)
    env.adv_jsma_t6 = sm.jsma(model, env.x, 6)
    env.adv_jsma_t7 = sm.jsma(model, env.x, 7)
    env.adv_jsma_t8 = sm.jsma(model, env.x, 8)
    env.adv_jsma_t9 = sm.jsma(model, env.x, 9)
    env.adv_df = df.deepfool(model, env.x)

LoadModel = True
SaveModel = False
n_epochs = 30
BATCH_SIZE = 100

n_train = X_train.shape[0]
n_batch = int(np.ceil(n_train/BATCH_SIZE))

sess = tf.Session(config=config)
with sess.as_default():
    if LoadModel:
        saver.restore(sess, "./fashion_mnist_cnn.model")
        print("Model loaded!")
    else:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for epoch in range(n_epochs):
            acc_train = 0.0
            for iteration in range(n_batch):
                start, end = get_start_end(iteration, BATCH_SIZE, n_train)
                sess.run(env.optimizer, feed_dict={env.x: X_train[start:end], env.y: y_train[start:end], env.training: True})
                acc_train += accuracy.eval(feed_dict={env.x: X_train[start:end], env.y: y_train[start:end], env.training: False})

            acc_train = (acc_train / float(n_batch))

            acc_test = accuracy.eval(feed_dict={env.x: X_test, env.y: y_test, env.training: False})
            print("[", epoch+1, "] Train accuracy:", acc_train, "Test accuracy:", acc_test)

if SaveModel:
    save_path = saver.save(sess, "./fashion_mnist_cnn.model")
    print("Model saved!")

with sess.as_default():
    adv_test_fgm = sess.run(env.adv_fgm, feed_dict={env.x: X_test, env.y: y_test, env.training: False})
    adv_test_jsma = sess.run(env.adv_jsma_t0, feed_dict={env.x: X_test, env.y: y_test, env.training: False})
    adv_test_df = sess.run(env.adv_df, feed_dict={env.x: X_test, env.y: y_test, env.training: False})

def SELECT_JSMA(target_class):
    if (target_class == 1): return env.adv_jsma_t1
    elif (target_class == 2): return env.adv_jsma_t2
    elif (target_class == 3): return env.adv_jsma_t3
    elif (target_class == 4): return env.adv_jsma_t4
    elif (target_class == 5): return env.adv_jsma_t5
    elif (target_class == 6): return env.adv_jsma_t6
    elif (target_class == 7): return env.adv_jsma_t7
    elif (target_class == 8): return env.adv_jsma_t8
    elif (target_class == 9): return env.adv_jsma_t9
    else: return env.adv_jsma_t0

# We select the image we want to test with:
target_test = 32
target_image = X_test[target_test-1:target_test]
target_label = y_test[target_test-1:target_test]

# Predict with normal test image:
with sess.as_default():
    best = sess.run(predictedz, feed_dict={env.x: target_image, env.training: False})
    print("y = {label_index} {label_vec} ({label})".format(label_index=np.argmax(target_label[0], axis=0), label_vec=target_label[0], label=label_dict[np.argmax(target_label[0], axis=0)]))
    plt.imshow(target_image[0].reshape(28,28), cmap='Greys')
    plt.show()
    print("Predicted = "+str(best[0])+" ("+str(label_dict[best[0]])+")")

# Try to predict with untargeted FGM attack
with sess.as_default():
    res = sess.run(env.adv_fgm, feed_dict={env.x: target_image})
    best = sess.run(predictedz, feed_dict={env.x: res, env.training: False})
    print("y = {label_index} {label_vec} ({label})".format(label_index=np.argmax(target_label[0], axis=0), label_vec=target_label[0], label=label_dict[np.argmax(target_label[0], axis=0)]))
    plt.imshow(res.reshape(28,28), cmap='Greys')
    plt.show()
    print("Predicted = "+str(best[0])+" ("+str(label_dict[best[0]])+")")

# Try to predict with targeted JSMA attack

# We can change target_class to direct the JSMA attack to that class
target_class = 0

with sess.as_default():
    res = sess.run(SELECT_JSMA(target_class), feed_dict={env.x: target_image})
    best = sess.run(predictedz, feed_dict={env.x: res, env.training: False})
    print("y = {label_index} {label_vec} ({label})".format(label_index=np.argmax(target_label[0], axis=0), label_vec=target_label[0], label=label_dict[np.argmax(target_label[0], axis=0)]))
    plt.imshow(res.reshape(28,28), cmap='Greys')
    plt.show()
    print("Predicted = "+str(best[0])+" ("+str(label_dict[best[0]])+")")

# Try to predict with untargeted DeepFool attack
with sess.as_default():
    res = sess.run(env.adv_df, feed_dict={env.x: target_image})
    best = sess.run(predictedz, feed_dict={env.x: res, env.training: False})
    print("y = {label_index} {label_vec} ({label})".format(label_index=np.argmax(target_label[0], axis=0), label_vec=target_label[0], label=label_dict[np.argmax(target_label[0], axis=0)]))
    plt.imshow(res.reshape(28,28), cmap='Greys')
    plt.show()
    print("Predicted = "+str(best[0])+" ("+str(label_dict[best[0]])+")")

with sess.as_default():
    acc = accuracy.eval(feed_dict={env.x: X_test, env.y: y_test, env.training: False})
    acc_adv_fgm = accuracy.eval(feed_dict={env.x: adv_test_fgm, env.y: y_test, env.training: False})
    acc_adv_jsma = accuracy.eval(feed_dict={env.x: adv_test_jsma, env.y: y_test, env.training: False})
    acc_adv_df = accuracy.eval(feed_dict={env.x: adv_test_df, env.y: y_test, env.training: False})
    
    print("Test accuracy:", acc)
    print("Test accuracy for adversarial FGM examples:", acc_adv_fgm)
    print("Test accuracy for adversarial JSMA examples:", acc_adv_jsma)
    print("Test accuracy for adversarial DeepFool examples:", acc_adv_df)



