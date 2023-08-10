import io
import re
import math
import zipfile
from collections import OrderedDict

from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from IPython.display import display
from scipy.misc import imresize
from sklearn.model_selection import train_test_split

print(tf.__version__)

tf.set_random_seed(0)

def extract_images_bytes(path='../CatDog/train.zip', scale=0.5):
    X_data, y_data = [], []
    # load origin
    z = zipfile.ZipFile(path, 'r')
    for file in z.filelist:
        m = re.match('.*(cat|dog).*', file.filename)
        if m:
            category = m.groups()[0]
            if category != 'cat': continue
            label = re.sub('[^/]+/', '', file.filename)
            img = Image.open(io.BytesIO((z.open(file.filename).read())))
            img_label = np.array(img)
            img_gray = np.array(img.convert('L'))
            img_gray_ = np.zeros([128, 128, 3])
            img_gray_[:, :, 0] = img_gray
            img_gray_[:, :, 1] = img_gray
            img_gray_[:, :, 2] = img_gray
            # 计算后的图
            X_data.append(img_gray_)
            y_data.append(img_label)
            if len(X_data) >= 128: break
    return np.array(X_data), np.array(y_data)

X_data, y_data = extract_images_bytes()
print(X_data.shape, y_data.shape)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.5, random_state=0)

mean_of_train = np.mean(X_train)
std_of_train = np.std(X_train)
print(mean_of_train, std_of_train)

X_train = (X_train - mean_of_train) / std_of_train
X_test = (X_test - mean_of_train) / std_of_train
y_train = (y_train - mean_of_train) / std_of_train
y_test = (y_test - mean_of_train) / std_of_train

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(np.mean(X_test), np.std(X_test), np.mean(y_train), np.std(y_train))

def rgb2yuv(rgb):
    """
    Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
    """
    rgb2yuv_filter = tf.constant(
        [[[[0.299, -0.169, 0.499],
           [0.587, -0.331, -0.418],
            [0.114, 0.499, -0.0813]]]])
    rgb2yuv_bias = tf.constant([0., 0.5, 0.5])

    temp = tf.nn.conv2d(rgb, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, rgb2yuv_bias)
    return temp

def yuv2rgb(yuv):
    """
    Convert YUV image into RGB https://en.wikipedia.org/wiki/YUV
    """
    yuv = tf.mul(yuv, 255)
    yuv2rgb_filter = tf.constant(
        [[[[1., 1., 1.],
           [0., -0.34413999, 1.77199996],
            [1.40199995, -0.71414, 0.]]]])
    yuv2rgb_bias = tf.constant([-179.45599365, 135.45983887, -226.81599426])
    temp = tf.nn.conv2d(yuv, yuv2rgb_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, yuv2rgb_bias)
    temp = tf.maximum(temp, tf.zeros(temp.get_shape(), dtype=tf.float32))
    temp = tf.minimum(temp, tf.mul(
        tf.ones(temp.get_shape(), dtype=tf.float32), 255))
    temp = tf.div(temp, 255)
    return temp

batch_size = 64
learning_rate = 0.0001
leakiness = 0.0

tf.set_random_seed(0)

X = tf.placeholder(tf.float32, [batch_size, X_train.shape[1], X_train.shape[2], 3], name='X')
y = tf.placeholder(tf.float32, [batch_size, y_train.shape[1], y_train.shape[2], 3], name='y')
print(X.get_shape(), y.get_shape())

def conv(input_layer, output_size, pitch_shape, name, strides=[1, 1, 1, 1], padding='VALID'):
    with tf.variable_scope(name):
        shape = [
            pitch_shape[0],
            pitch_shape[1],
            int(input_layer.get_shape()[-1]),
            output_size
        ]
        kernel = tf.Variable(tf.random_normal(shape, stddev=np.sqrt(2.0 / (shape[0] + shape[1] + shape[3]))))
        bias = tf.Variable(tf.zeros([shape[-1]]))
        conv = tf.nn.conv2d(input_layer, kernel, strides=strides, padding=padding)
        conv = tf.nn.bias_add(conv, bias)
        print(name, conv.get_shape())
        return conv

def relu(x, leakiness=0.0):
    """Relu, with optional leaky support.
    borrow from https://github.com/tensorflow/models/blob/master/resnet/resnet_main.py
    """
    return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

model = rgb2yuv(X)
print(model.get_shape())

# 输入图片填充 0 ，使得经过三次卷积之后回复128x128大小
model = tf.pad(model, [[0, 0], [7, 7], [7, 7], [0, 0]])
print(model.get_shape())

model = relu(conv(model, 64, (9, 9), 'conv_1', padding='VALID'), leakiness=leakiness)

model = relu(conv(model, 32, (3, 3), 'conv_2', padding='VALID'), leakiness=leakiness)

model = conv(model, 3, (5, 5), 'conv_3', padding='VALID')

pred = rgb2yuv(model)
print(pred.get_shape())

cost = tf.reduce_mean(tf.square(y - pred))

opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

params = tf.trainable_variables()

gradients = tf.gradients(cost, params)

clipped_gradients, norm = tf.clip_by_global_norm(
    gradients,
    5.0
)

train_step = opt.apply_gradients(zip(clipped_gradients, params))

init = tf.global_variables_initializer()

def batch_flow(inputs, targets, batch_size):
    """流动数据流"""
    flowed = 0
    total = len(inputs)
    while True:
        X_ret = []
        y_ret = []
        for i in range(total):
            X_ret.append(inputs[i])
            y_ret.append(targets[i])
            if len(X_ret) == batch_size:
                flowed += batch_size
                X, y = np.array(X_ret), np.array(y_ret)
                yield X, y
                X_ret = []
                y_ret = []
            if flowed >= total:
                break
        if flowed >= total:
            break

for batch_x, batch_y in batch_flow(X_train, y_train, batch_size):
    print(batch_x.shape, batch_y.shape)
    break

n_epoch = 100000

tf.set_random_seed(0)
with tf.Session() as sess:
    sess.run(init)
    total = None
    for epoch in tqdm(list(range(n_epoch))):
        costs = []
        for batch_x, batch_y in batch_flow(X_train, y_train, batch_size):
            _, c = sess.run([train_step, cost], feed_dict={X: batch_x, y: batch_y})
            costs.append(c)
        if total is None:
            total = len(costs)
        if epoch > 0 and epoch % 200 == 0:
            print('epoch: {}, loss: {:.4f}'.format(epoch, np.mean(costs)))
        if epoch > 0 and epoch % 1000 == 0:
            print('calculate train accuracy')
            costs = []
            train_result = []
            for batch_x, batch_y in batch_flow(X_train, y_train, batch_size):
                c, p = sess.run([cost, pred], feed_dict={X: batch_x, y: batch_y})
                costs.append(c)
                train_result += list(p)
            print('test loss: {:.4f}'.format(np.mean(costs)))
            print('calculate test accuracy')
            costs = []
            test_result = []
            for batch_x, batch_y in batch_flow(X_test, y_test, batch_size):
                c, p = sess.run([cost, pred], feed_dict={X: batch_x, y: batch_y})
                costs.append(c)
                test_result += list(p)
            print('test loss: {:.4f}'.format(np.mean(costs)))
    print('Done')

def disp(n, mean, std, X, y, result):
    x = imresize(np.uint8(X[n] * std + mean), (128, 128))
    y = np.uint8(y[n] * std + mean)
    pred = np.uint8(result[n] * std + mean)
    display(
        Image.fromarray(x),
        Image.fromarray(y),
        Image.fromarray(pred)
    )

disp(0, mean_of_train, std_of_train, X_train, y_train, train_result)

disp(0, mean_of_train, std_of_train, X_test, y_test, test_result)

disp(1, mean_of_train, std_of_train, X_train, y_train, train_result)

disp(2, mean_of_train, std_of_train, X_train, y_train, train_result)

disp(1, mean_of_train, std_of_train, X_test, y_test, test_result)

disp(2, mean_of_train, std_of_train, X_test, y_test, test_result)



