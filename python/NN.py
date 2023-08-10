import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

dataset = pd.read_csv('creditcard.csv')
print 'total rows: ' + str(dataset.shape[0])
print 'total columns: ' + str(dataset.shape[1])
dataset = dataset.iloc[np.random.permutation(len(dataset))]
dataset.head()

ds = dataset.iloc[:, 1:]
correlation = ds.corr()
plt.figure(figsize = (10, 10))
sns.heatmap(correlation, vmax = 1, square = True, annot = False)
plt.savefig('pearson.pdf')
plt.show()

from sklearn.cross_validation import train_test_split
dataset = dataset.iloc[:, 1:]
x_data = dataset.iloc[:, :-2].values.astype(np.float32)
y_data = dataset.iloc[:, -1]

onehot = np.zeros((y_data.shape[0], np.unique(y_data).shape[0]))
for i in xrange(y_data.shape[0]):
    onehot[i, y_data[i]] = 1.0
    
x_train, x_test, y_train, y_test, _, y_test_label = train_test_split(x_data, onehot, y_data, test_size = 0.2)

size_layer_first = 512
size_layer_second = 128
learning_rate = 0.001
beta = 0.00005

X = tf.placeholder("float", [None, x_train.shape[1]])
Y = tf.placeholder("float", [None, np.unique(y_data).shape[0]])
layer1 = tf.Variable(tf.random_normal([x_train.shape[1], size_layer_first]))
layer2 = tf.Variable(tf.random_normal([size_layer_first, size_layer_first]))
layer3 = tf.Variable(tf.random_normal([size_layer_first, size_layer_second]))
layer4 = tf.Variable(tf.random_normal([size_layer_second, np.unique(y_data).shape[0]]))

bias1 = tf.Variable(tf.random_normal([size_layer_first], stddev = 0.1))
bias2 = tf.Variable(tf.random_normal([size_layer_first], stddev = 0.1))
bias3 = tf.Variable(tf.random_normal([size_layer_second], stddev = 0.1))
bias4 = tf.Variable(tf.random_normal([np.unique(y_data).shape[0]], stddev = 0.1))

hidden1 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(X, layer1) + bias1), 0.3)
hidden2 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(hidden1, layer2) + bias2), 0.3)
hidden3 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(hidden2, layer3) + bias3), 0.3)
hidden4 = tf.matmul(hidden3, layer4) + bias4

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = hidden4))
loss += tf.nn.l2_loss(layer1) * beta + tf.nn.l2_loss(layer2) * beta + tf.nn.l2_loss(layer3) * beta + tf.nn.l2_loss(layer4) * beta

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

correct_pred = tf.equal(tf.argmax(hidden4, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
import time

BATCH_SIZE = 128

EPOCH, LOSS, ACC = [], [], []
for i in xrange(10):
    last = time.time()
    EPOCH.append(i)
    TOTAL_LOSS, ACCURACY = 0, 0
    for n in xrange(0, (x_train.shape[0] // BATCH_SIZE) * BATCH_SIZE, BATCH_SIZE):
        cost, _ = sess.run([loss, optimizer], feed_dict = {X : x_train[n: n + BATCH_SIZE, :], Y : y_train[n: n + BATCH_SIZE, :]})
        ACCURACY += sess.run(accuracy, feed_dict = {X : x_train[n: n + BATCH_SIZE, :], Y : y_train[n: n + BATCH_SIZE, :]})
        TOTAL_LOSS += cost
    
    TOTAL_LOSS /= (x_train.shape[0] // BATCH_SIZE)
    ACCURACY /= (x_train.shape[0] // BATCH_SIZE)
    LOSS.append(TOTAL_LOSS); ACC.append(ACCURACY)
    print 'epoch: ' + str(i + 1) + ', loss: ' + str(TOTAL_LOSS) + ', accuracy: ' + str(ACCURACY) + ', s / epoch: ' + str(time.time() - last)

from sklearn import metrics
testing_acc, logits = sess.run([accuracy, tf.cast(tf.argmax(hidden4, 1), tf.int32)], feed_dict = {X : x_test, Y : y_test})
print 'testing accuracy: ' + str(testing_acc)
print(metrics.classification_report(y_test_label, logits, target_names = ['non', 'fraud']))

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.subplot(1, 2, 1)
plt.plot(EPOCH, LOSS)
plt.xlabel('epoch'); plt.ylabel('loss')
plt.subplot(1, 2, 2)
plt.plot(EPOCH, ACC)
plt.xlabel('epoch'); plt.ylabel('accuracy')
plt.show()

