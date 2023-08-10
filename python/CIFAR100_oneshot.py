import numpy as np
import sys
import tensorflow as tf
import time

import load_cifar
import ops

from sklearn.decomposition import PCA

get_ipython().magic('matplotlib inline')
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt

(train_images, test_images), (train_labels, test_labels) = load_cifar.Batches(
    dataset="cifar100",
    include_labels=True)
images = np.concatenate([train_images, test_images])
labels = np.concatenate([train_labels, test_labels])

figsize(16, 8)
plt.imshow(
    np.concatenate(
        [np.concatenate(list(images[i * 20:(i + 1) * 20]), axis=1)
         for i in xrange(10)]))
plt.axis("off")
plt.show()

categories = np.array(list(set(labels)))
np.random.shuffle(categories)
test_categories = categories[:10]
train_categories = categories[10:]

indexer = np.array([label in train_categories for label in labels])
train_ix = np.arange(len(labels))[indexer]
test_ix = np.arange(len(labels))[~indexer]

try:
    del(sess)
    print "deleted session"
except Exception as e:
    print "no existing session to delete"
sess = tf.InteractiveSession()

class ClassificationNetwork(object):
    
    def __init__(self, session):
        self.sess = session
        self.train_xents = []
        self.train_accuracies = []
        
        with tf.variable_scope("ClassificationNetwork") as scope:
            # Input starts at 32x32
            self.inputs = tf.placeholder(tf.float32, [None, 32, 32, 3], name="Inputs")
            self.conv1 = ops.ConvLayer(self.inputs, 5, 16, tf.nn.elu, "Conv1")
            
            # Downsample to 16x16
            self.pool1 = ops.MaxPool2x2(self.conv1, name="Pool1")
            self.conv2 = ops.ConvLayer(self.pool1, 3, 32, tf.nn.elu, "Conv2")
            self.conv3 = ops.ConvLayer(self.conv2, 3, 32, tf.nn.elu, "Conv3")
            
            # Downsample to 8x8
            self.pool2 = ops.MaxPool2x2(self.conv3, name="Pool2")
            self.conv4 = ops.ConvLayer(self.pool2, 3, 64, tf.nn.elu, "Conv4")
            self.conv5 = ops.ConvLayer(self.conv4, 3, 64, tf.nn.elu, "Conv4")
            
            # Downsample to 4 x 4
            self.pool3 = ops.MaxPool2x2(self.conv5, name="Pool3")
            self.conv6 = ops.ConvLayer(self.pool3, 3, 128, tf.nn.elu, "Conv6")
            self.conv7 = ops.ConvLayer(self.conv6, 3, 128, tf.nn.elu, "Conv7")
            
            # Softmax regression
            self.flat = tf.reshape(
                self.conv7,
                [-1, int(np.prod(self.conv7.get_shape()[1:]))],
                name="Flatten")
            self.drop_rate = tf.placeholder(tf.float32, [])
            self.drop = tf.nn.dropout(self.flat, self.drop_rate, name="Dropout")
            self.full_W, self.full_b, self.outputs = ops.HiddenLayer(
                self.drop,
                [int(self.drop.get_shape()[1]), 100],
                nonlin=tf.nn.softmax, scope="FullyConnected")
            
            # Error and Optimization
            self.targets = tf.placeholder(tf.float32, [None, 100], name="Targets")
            self.xent = -tf.reduce_mean(self.targets * tf.log(self.outputs), name="Xent")
            self.learning_rate = tf.placeholder(tf.float32, [])
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.xent)
            self.correct = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.targets, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
            
            # Initialize
            self.sess.run(tf.initialize_all_variables())

    def TrainBatch(self, images, labels, learning_rate):
        targets = np.zeros([images.shape[0], 100])
        targets[np.arange(images.shape[0]), labels] = 1.0
        _, xent, acc = self.sess.run(
            [self.train_step, self.xent, self.accuracy],
            feed_dict={
                self.learning_rate: learning_rate,
                self.drop_rate: 0.5,
                self.inputs: images,
                self.targets: targets})
        return xent, acc 
    
    def TrainEpoch(self, images, labels, batch_size, learning_rate):
        n_batches = int(images.shape[0] / batch_size)
        indices = np.arange(images.shape[0])
        np.random.shuffle(indices)
        for batch in np.arange(n_batches):
            start_offset = batch * batch_size
            stop_offset = start_offset + batch_size
            batch_indices = indices[start_offset:stop_offset]
            xent, acc = self.TrainBatch(
                images[batch_indices],
                labels[batch_indices],
                learning_rate)
            self.train_xents.append(xent)
            self.train_accuracies.append(acc)
            sys.stdout.write("{:03.1f}%\t".format(acc * 100))
            if np.isnan(xent):
                raise RuntimeError("NaN Entropy")
        

clf = ClassificationNetwork(sess)

clf.TrainEpoch(images[train_ix], labels[train_ix], 128, 0.001)

clf.TrainEpoch(images[train_ix], labels[train_ix], 128, 0.0005)

clf.TrainEpoch(images[train_ix], labels[train_ix], 128, 0.0001)

plt.plot(np.arange(len(clf.train_xents)), clf.train_xents)
plt.title("Training Cross Entropy by Batch")
plt.show()

plt.plot(np.arange(len(clf.train_accuracies)), clf.train_accuracies)
plt.title("Training Accuracy by Batch")
plt.show()

embeddings = clf.flat.eval(feed_dict={clf.inputs: images[test_ix]})

pca = PCA(n_components=2)
pca.fit(embeddings)

pca_embeddings = pca.transform(embeddings)

figsize(16, 8)

COLORS = [
    "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941",
    "#006FA6", "#A30059", "#FFDBE5", "#7A4900", "#0000A6"]

for c, label in enumerate(test_categories):
    indexer = labels[test_ix] == label
    plt.scatter(*pca_embeddings[indexer].T, alpha=0.2, color=COLORS[c])
plt.title("First Two Principal Components of 10 Categories Not Seen During Training")
plt.show()

from scipy.spatial import distance

dist = distance.squareform(distance.pdist(embeddings))

test_labels = labels[test_ix]
pairs = []
for x in np.arange(1, embeddings.shape[0]):
    for y in np.arange(x):
        pairs.append([dist[x, y], test_labels[x] == test_labels[y]])
pairs = np.array(pairs)

pairs[pairs[:, 1] > 0][:, 0].mean()



