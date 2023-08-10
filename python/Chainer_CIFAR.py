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
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import cuda
from utils import cifar_for_library, yield_mb
from nb_logging import NotebookLogger, output_to, error_to
import codecs

sys.__stdout__ = codecs.getwriter("utf-8")(sys.__stdout__.detach())

nb_teminal_logger = NotebookLogger(sys.stdout.session, sys.stdout.pub_thread, sys.stdout.name, sys.__stdout__)

rst_out = output_to(nb_teminal_logger)
rst_err = error_to(nb_teminal_logger)

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Chainer: ", chainer.__version__)
print("Numpy: ", np.__version__)

data_path = path.join(os.getenv('AZ_BATCHAI_INPUT_DATASET'), 'cifar-10-batches-py')

class SymbolModule(chainer.Chain):
    def __init__(self):
        super(SymbolModule, self).__init__(
            conv1=L.Convolution2D(3, 50, ksize=(3,3), pad=(1,1)),
            conv2=L.Convolution2D(50, 50, ksize=(3,3), pad=(1,1)),      
            conv3=L.Convolution2D(50, 100, ksize=(3,3), pad=(1,1)),  
            conv4=L.Convolution2D(100, 100, ksize=(3,3), pad=(1,1)),  
            # feature map size is 8*8 by pooling
            fc1=L.Linear(100*8*8, 512),
            fc2=L.Linear(512, N_CLASSES),
        )
    
    def __call__(self, x):
        h = F.relu(self.conv2(F.relu(self.conv1(x))))
        h = F.max_pooling_2d(h, ksize=(2,2), stride=(2,2))
        h = F.dropout(h, 0.25)
        
        h = F.relu(self.conv4(F.relu(self.conv3(h))))
        h = F.max_pooling_2d(h, ksize=(2,2), stride=(2,2))
        h = F.dropout(h, 0.25)       
        
        h = F.dropout(F.relu(self.fc1(h)), 0.5)
        return self.fc2(h)

def init_model(m):
    optimizer = optimizers.MomentumSGD(lr=LR, momentum=MOMENTUM)
    optimizer.setup(m)
    return optimizer

def to_chainer(array, **kwargs):
    return chainer.Variable(cuda.to_gpu(array), **kwargs)

get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = cifar_for_library(data_path, channel_first=True)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')

get_ipython().run_cell_magic('time', '', '# Create symbol\nsym = SymbolModule()\nif GPU:\n    chainer.cuda.get_device(0).use()  # Make a specified GPU current\n    sym.to_gpu()  # Copy the model to the GPU')

get_ipython().run_cell_magic('time', '', 'optimizer = init_model(sym)')

get_ipython().run_cell_magic('time', '', 'for j in range(EPOCHS):\n    for data, target in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        # Get samples\n        optimizer.update(L.Classifier(sym), to_chainer(data), to_chainer(target))\n    # Log\n    print(j)')

get_ipython().run_cell_magic('time', '', "n_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = y_test[:n_samples]\nc = 0\n\nwith chainer.using_config('train', False):\n    for data, target in yield_mb(x_test, y_test, BATCHSIZE):\n        # Forwards\n        pred = chainer.cuda.to_cpu(sym(to_chainer(data)).data.argmax(-1))\n        # Collect results\n        y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = pred\n        c += 1")

print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))

