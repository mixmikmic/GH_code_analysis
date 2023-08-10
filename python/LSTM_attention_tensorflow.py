import pickle
import numpy as np
import tensorflow as tf

from data import X_train, X_test, y_train, y_test
from data import fit_vectorizer, fit_onehot
from data import batch_flow, test_batch_flow

tf.set_random_seed(0)

embedding_size = 128
PAD = ' ' # 句子不到max_len长度时的占位符
max_len = max(len(x) for x in X_train)
print('单个训练样本最大长度：{}'.format(max_len))

vectorizer = fit_vectorizer(X_train, embedding_size, max_len, PAD)
onehot = fit_onehot(y_train)

n_epoch = 10
num_units = 128
batch_size = 256
time_steps = max_len
input_size = embedding_size
target_size = len(onehot.feature_indices_)
print('time_steps', time_steps)
print('input_size', input_size)
print('target_size', target_size)

test_batch_flow(X_train, y_train, batch_size, vectorizer, onehot, max_len, PAD)

X = tf.placeholder(tf.float32, [time_steps, batch_size, input_size], name='X')
y = tf.placeholder(tf.float32, [batch_size, target_size], name='y')
weight = tf.Variable(tf.random_normal([time_steps * num_units, target_size]), name='weight')
bias = tf.Variable(tf.zeros([target_size]), name='bias')

with tf.variable_scope("dynamic_scope", reuse=None):
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
    # 16 是 attention size 
    cell = tf.contrib.rnn.AttentionCellWrapper(cell, 16, state_is_tuple=True)
    outputs, states = tf.nn.dynamic_rnn(
        cell,
        inputs=X,
        time_major=True, dtype=tf.float32
    )
    outputs = tf.reshape(outputs, [batch_size, -1])
    pred = tf.nn.softmax(tf.add(tf.matmul(outputs, weight), bias))
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    cost = tf.reduce_mean(
        -tf.reduce_sum(y * tf.log(pred),
        reduction_indices=1)
    )

train_step = tf.train.AdamOptimizer().minimize(cost)

# 初始化所有变量
init = tf.initialize_all_variables()

# disable GPU，关闭GPU支持
config = tf.ConfigProto(
    device_count = {'GPU': 0}
)

with tf.Session(config=config) as sess:
    sess.run(init)
    for epoch in range(n_epoch + 1):
        costs = []
        accs = []
        for X_sample, y_sample in batch_flow(X_train, y_train, batch_size, vectorizer, onehot, max_len, PAD):
            feeds = {X: X_sample.reshape([time_steps, batch_size, input_size]), y: y_sample}
            sess.run(train_step, feeds)
            c, acc = sess.run([cost, accuracy], feeds)
            costs.append(c)
            accs.append(acc)
        print('epoch {} cost: {:.4f} acc: {:.4f}'.format(
            epoch, np.mean(costs), np.mean(acc)
        ))
    # train
    costs = []
    accs = []
    for X_sample, y_sample in batch_flow(X_train, y_train, batch_size, vectorizer, onehot, max_len, PAD):
        feeds = {X: X_sample.reshape([time_steps, batch_size, input_size]), y: y_sample}
        c, acc = sess.run([cost, accuracy], feeds)
        costs.append(c)
        accs.append(acc)
    print('train cost: {:.4f} acc: {:.4f}'.format(np.mean(costs), np.mean(acc)))
    # test
    costs = []
    accs = []
    for X_sample, y_sample in batch_flow(X_test, y_test, batch_size, vectorizer, onehot, max_len, PAD):
        feeds = {X: X_sample.reshape([time_steps, batch_size, input_size]), y: y_sample}
        c, acc = sess.run([cost, accuracy], feeds)
        costs.append(c)
        accs.append(acc)
    print('test cost: {:.4f} acc: {:.4f}'.format(np.mean(costs), np.mean(acc)))



