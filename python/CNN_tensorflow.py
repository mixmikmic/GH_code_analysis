import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from data import X_train, X_test, y_train, y_test

tf.set_random_seed(0)

embedding_size = 128
PAD = ' ' # 句子不到max_len长度时的占位符
max_len = max(len(x) for x in X_train)
print('单个训练样本最大长度：{}'.format(max_len))

# 载入我的自定义库 @qhduan
import sys
import importlib
sys.path.insert(0, '../../')
import tfkit
importlib.reload(tfkit)

wl = tfkit.WordLabel()
X_train_vec = wl.fit_transform(X_train, max_len=max_len)
X_test_vec = wl.transform(X_test, max_len=max_len)

print(wl.max_features, X_train_vec.shape, X_test_vec.shape)

oh = OneHotEncoder(sparse=False)
y_train = oh.fit_transform(y_train.reshape([-1, 1]))
y_test = oh.transform(y_test.reshape([-1, 1]))

learning_rate = 0.003
n_epoch = 10
batch_size = 128
time_steps = max_len
input_size = embedding_size
target_size = 2
print('time_steps', time_steps)
print('input_size', input_size)
print('target_size', target_size)

X = tf.placeholder(tf.float32, [batch_size, max_len], name='X')
y = tf.placeholder(tf.float32, [batch_size, target_size], name='X')

model = X
model = tfkit.embedding(model, wl.max_features, embedding_size, max_len, name='embedding')
model = tf.reshape(model, [batch_size, max_len, embedding_size, 1])
model = tfkit.conv(model, 512, (14, 1), name='conv_1', activation='relu', padding='VALID')
model = tfkit.flatten(model, 'flatten')
model = tfkit.full_connect(model, target_size, name='fc_2')

train_step, cost = tfkit.train_softmax(
    model, y,
    opt=tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
)

measures = [
    cost,
    tfkit.accuracy(model, y, softmax=True),
]

# 初始化所有变量
init = tf.global_variables_initializer()

# 本来是要关，不过CNN不用GPU真的好慢……
# disable GPU，关闭GPU支持
config = tf.ConfigProto(
#     device_count = {'GPU': 0}
)

with tf.Session(config=config) as sess:
    sess.run(init)
    total = int((len(X_train) - 1) / batch_size + 1)
    for epoch in range(n_epoch):
        print('epoch: {}'.format(epoch))
        record = []
        for X_sample, y_sample in tqdm(tfkit.batch_flow(X_train_vec, y_train, batch_size), total=total, file=sys.stdout):
            feeds = {X: X_sample, y: y_sample}
            sess.run(train_step, feeds)
            record.append(sess.run(measures, feeds))
        print('train: loss: {:.4f}, acc: {:.4f}'.format(
            np.mean([x[0] for x in record]),
            np.mean([x[1] for x in record])
        ))
        record = []
        for X_sample, y_sample in tfkit.batch_flow(X_test_vec, y_test, batch_size):
            feeds = {X: X_sample, y: y_sample}
            record.append(sess.run(measures, feeds))
        print('test: loss: {:.4f}, acc: {:.4f}'.format(
            np.mean([x[0] for x in record]),
            np.mean([x[1] for x in record])
        ))



