# copy helper code from: https://github.com/dmlc/mxnet-notebooks/blob/master/python/tutorials/mnist.ipynb

import numpy as np
import os
import urllib
import gzip
import struct

def download_data(url, force_download=True): 
    fname = url.split("/")[-1]
    if force_download or not os.path.exists(fname):
        urllib.urlretrieve(url, fname)
    return fname

def read_data(label_url, image_url):
    with gzip.open(download_data(label_url)) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_data(image_url), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)

path='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'

(train_lbl, train_img) = read_data(
    path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz')
(val_lbl, val_img) = read_data(
    path+'t10k-labels-idx1-ubyte.gz', path+'t10k-images-idx3-ubyte.gz')

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(train_img[i], cmap='Greys_r')
    plt.axis('off')
    
plt.show()

train_img[0].shape

# Transform the data

import mxnet as mx

# 4D (batch_size, num_channels, width, height) ==> (bsize, 1, 28, 28)

def to_4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

batch_size = 200

train_iter = mx.io.NDArrayIter(to_4d(train_img), train_lbl, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(to_4d(val_img), val_lbl, batch_size)

train_iter.__dict__

# Model

data = mx.sym.Variable('data')
#mxnet.symbol.Convolution(data=None, weight=None, bias=None, kernel=_Null, stride=_Null, dilate=_Null, pad=_Null, num_filter=_Null, num_group=_Null, workspace=_Null, no_bias=_Null, cudnn_tune=_Null, cudnn_off=_Null, layout=_Null, name=None, attr=None, out=None, **kwargs)¶

# conv 1 layer
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=30)
act1 = mx.sym.Activation(data=conv1, act_type='relu')
pool1 = mx.sym.Pooling(data=act1, pool_type="max", kernel=(2,2))

# conv 2 layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
act2 = mx.sym.Activation(data=conv2, act_type='relu')
pool2 = mx.sym.Pooling(data=act2, pool_type="max", kernel=(2,2))

# conv 3 layer
conv3 = mx.sym.Convolution(data=pool2, kernel=(5,5), num_filter=50)
act3 = mx.sym.Activation(data=conv3, act_type='relu')
pool3 = mx.sym.Pooling(data=act3, pool_type="max", kernel=(2,2))


# Fully connected layer
flatten = mx.sym.Flatten(pool3)
fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=500)
act3 = mx.sym.Activation(data=fc1, act_type='relu')

fc2 = mx.sym.FullyConnected(data=act3, num_hidden=10)

lenet = mx.sym.SoftmaxOutput(data=fc2, name="softmax")

# Output layer

shape = {"data": (batch_size, 1, 28, 28)}
mx.viz.plot_network(lenet, shape=shape)

# Train the model

import logging
logging.getLogger().setLevel(logging.DEBUG)

ctx = [mx.gpu(i) for i in range(2)]
num_epoch = 10

net = mx.mod.Module(symbol=lenet, context=ctx)
net.bind(data_shapes=[train_iter.provide_data[0]], label_shapes=[train_iter.provide_label[0]])

net.fit(train_iter,
       val_iter,
       optimizer="sgd",
       optimizer_params={'learning_rate' : 0.1},
       eval_metric='acc',
       batch_end_callback=mx.callback.Speedometer(batch_size, 200),
       num_epoch=num_epoch
       )


# prediction function

idx = 30; 
plt.imshow(val_img[idx], cmap='Greys_r')
plt.axis('off')
plt.show()

preds = net.predict(val_iter)
prob = preds[idx].asnumpy()

print "predicted: %d , probability: %f " % (prob.argmax(), max(prob))



