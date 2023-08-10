# 图表库
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
# 深度学习
import tensorflow as tf
# 矩阵
import numpy as np
# sklearn的各种辅助小工具
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# 设置随机种子，这样每次得到的随机数结果都一样了
tf.set_random_seed(0)

# 读取数据集
X, y = load_iris(True)

# 把目标变量改变形状，相当于把一个一维向量转化为一个1xn维矩阵（当然还是向量）
y = y.reshape([len(y), 1])

# one hot编码器，例如数据的分类数是3，可以吧把 0 编码为[0 0 1]，1 编码为 [0 1 0]， 2 编码为[1 0 0]
enc = OneHotEncoder()

y = enc.fit_transform(y).toarray()

# 分割测试集与训练集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# 训练集有120条数据，测试集30条数据，输入有4个变量，输出有3个变量（多分类）
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 输入变量数
input_size = 4
# 输出变量数
target_size = 3

# input的占位
X = tf.placeholder(tf.float32, [None, input_size])
y = tf.placeholder(tf.float32, [None, target_size])

# 要训练的参数
W = tf.Variable(tf.random_normal([input_size, target_size]))
b = tf.Variable(tf.random_normal([target_size]))

# 输出结果
pred = tf.nn.softmax(tf.add(tf.matmul(X, W), b))

# 定义损失函数，这个是标准softmax损失
cost = tf.reduce_mean(
    -tf.reduce_sum(y * tf.log(pred),
    reduction_indices=1)
)

# 学习率
learning_rate = 0.01
# 迭代次数
n_epoch = 1200

# 梯度下降算子
opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化所有变量
init = tf.initialize_all_variables()

# disable GPU，关闭GPU支持
config = tf.ConfigProto(
    device_count = {'GPU': 0}
)

# 保存历史损失
costs = []
with tf.Session(config=config) as sess:
    sess.run(init)
    # 开始迭代
    for epoch in range(n_epoch + 1):
        # 反向传播，梯度下降
        sess.run(opt, feed_dict={X: X_train, y: y_train})
        # 计算损失
        c = sess.run(cost, feed_dict={X: X_train, y: y_train})
        # 记录损失
        costs.append(c)
        if epoch % 50 == 0:
            print('Epoch: {}, cost: {}'.format(epoch, c))
    # 计算训练集与测试集结果
    pred_train = sess.run(pred, feed_dict={X: X_train, y: y_train})
    pred_test = sess.run(pred, feed_dict={X: X_test, y: y_test})

# 训练集准确率
acc = accuracy_score(y_train.argmax(axis=1), pred_train.argmax(axis=1))
print('train accuracy: {}'.format(acc))

# 测试集准确率
acc = accuracy_score(y_test.argmax(axis=1), pred_test.argmax(axis=1))
print('test accuracy: {}'.format(acc))



