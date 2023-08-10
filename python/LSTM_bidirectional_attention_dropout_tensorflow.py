import pickle
import numpy as np
import tensorflow as tf

from data import X_train, X_test, y_train, y_test
from data import fit_vectorizer, fit_onehot
from data import batch_flow, test_batch_flow

tf.set_random_seed(0)

embedding_size = 64
PAD = ' ' # 句子不到max_len长度时的占位符
max_len = max(len(x) for x in X_train)
print('单个训练样本最大长度：{}'.format(max_len))

vectorizer = fit_vectorizer(X_train, embedding_size, max_len, PAD)
onehot = fit_onehot(y_train)

n_epoch = 10
num_units = 64
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
# 这里的weight相别其他实现大小*2了，因为双向LSTM的输出是双倍的，Forward一次，Backward一次
weight = tf.Variable(tf.random_normal([time_steps * num_units * 2, target_size]), name='weight')
bias = tf.Variable(tf.zeros([target_size]), name='bias')

with tf.variable_scope("dynamic_scope", reuse=None):
    X_ = tf.reshape(X, [-1, input_size])
    # 分割为 time_steps 个数组，每个数组大小 batch_size * input_size
    X_ = tf.split(0, time_steps, X_)
    
    fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
    fw_cell = tf.contrib.rnn.AttentionCellWrapper(fw_cell, 8, state_is_tuple=True)
    fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=0.4)
    
    bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
    bw_cell = tf.contrib.rnn.AttentionCellWrapper(bw_cell, 8, state_is_tuple=True)
    bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=0.5)
    
    outputs, _, _ = tf.nn.bidirectional_rnn(
        fw_cell, bw_cell, X_, dtype=tf.float32
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



