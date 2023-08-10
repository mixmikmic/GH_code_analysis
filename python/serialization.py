from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
ctx = mx.cpu()
# ctx = mx.gpu()

X = nd.ones((100, 100))
Y = nd.zeros((100, 100))
import os
os.makedirs('checkpoints', exist_ok=True)
filename = "checkpoints/test1.params"
nd.save(filename, [X, Y])

A, B = nd.load(filename)
print(A)
print(B)

mydict = {"X": X, "Y": Y}
filename = "checkpoints/test2.params"
nd.save(filename, mydict)

C = nd.load(filename)
print(C)

num_hidden = 256
num_outputs = 1
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_outputs))

net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=ctx)
net(nd.ones((1, 100), ctx=ctx))

filename = "checkpoints/testnet.params"
net.save_params(filename)
net2 = gluon.nn.Sequential()
with net2.name_scope():
    net2.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net2.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net2.add(gluon.nn.Dense(num_outputs))
net2.load_params(filename, ctx=ctx)
net2(nd.ones((1, 100), ctx=ctx))

