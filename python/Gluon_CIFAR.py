# Parameters
EPOCHS = 10
N_CLASSES=10
BATCHSIZE = 64
LR = 0.01
MOMENTUM = 0.9
GPU = True

import os
from os import path
import sys
import numpy as np
import math
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
from utils import cifar_for_library, yield_mb
from nb_logging import NotebookLogger, output_to, error_to
import codecs

ctx = mx.gpu()

sys.__stdout__ = codecs.getwriter("utf-8")(sys.__stdout__)

nb_teminal_logger = NotebookLogger(sys.stdout.session, sys.stdout.pub_thread, sys.stdout.name, sys.__stdout__)

rst_out = output_to(nb_teminal_logger)
rst_err = error_to(nb_teminal_logger)

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("MXNet: ", mx.__version__)
print("Numpy: ", np.__version__)

data_path = path.join(os.getenv('AZ_BATCHAI_INPUT_DATASET'), 'cifar-10-batches-py')

def SymbolModule():
    sym = gluon.nn.Sequential()
    with sym.name_scope():
        sym.add(gluon.nn.Conv2D(channels=50, kernel_size=3, padding=1, activation='relu'))
        sym.add(gluon.nn.Conv2D(channels=50, kernel_size=3, padding=1))
        sym.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        sym.add(gluon.nn.Activation('relu'))
        # Equiv to gluon.nn.LeakyReLU(0)
        sym.add(gluon.nn.Dropout(0.25))
        sym.add(gluon.nn.Conv2D(channels=100, kernel_size=3, padding=1, activation='relu'))
        sym.add(gluon.nn.Conv2D(channels=100, kernel_size=3, padding=1))
        sym.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        sym.add(gluon.nn.Activation('relu'))
        sym.add(gluon.nn.Dropout(0.25))
        sym.add(gluon.nn.Flatten())
        sym.add(gluon.nn.Dense(512, activation='relu'))
        sym.add(gluon.nn.Dropout(0.25))
        sym.add(gluon.nn.Dense(N_CLASSES))
    return sym

def init_model(m):
    trainer = gluon.Trainer(m.collect_params(), 'sgd',
                            {'learning_rate': LR, 'momentum':MOMENTUM})
    criterion = gluon.loss.SoftmaxCrossEntropyLoss()
    return trainer, criterion

get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = cifar_for_library(data_path, channel_first=True)\n\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')

get_ipython().run_cell_magic('time', '', 'sym = SymbolModule()\nsym.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)')

get_ipython().run_cell_magic('time', '', 'trainer, criterion = init_model(sym)')

get_ipython().run_cell_magic('time', '', "# Sets training = True \nfor j in range(EPOCHS):\n    train_loss = 0.0\n    for data, target in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        # Get samples\n        data = nd.array(data).as_in_context(ctx)\n        target = nd.array(target).as_in_context(ctx)\n        with autograd.record():\n            # Forwards\n            output = sym(data)\n            # Loss\n            loss = criterion(output, target)\n        # Back-prop\n        loss.backward()\n        trainer.step(data.shape[0])\n        train_loss += nd.sum(loss).asscalar()\n    # Log\n    print('Epoch %3d: loss: %5.4f'%(j, train_loss/len(x_train)))")

get_ipython().run_cell_magic('time', '', '# Test model\nn_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = y_test[:n_samples]\nc = 0\nfor data, target in yield_mb(x_test, y_test, BATCHSIZE):\n    # Get samples\n    data = nd.array(data).as_in_context(ctx)\n    # Forwards\n    output = sym(data)\n    pred = nd.argmax(output, axis=1)\n    # Collect results\n    y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = pred.asnumpy()\n    c += 1')

print("Accuracy: ", float(sum(y_guess == y_truth))/len(y_guess))

