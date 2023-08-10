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
import mxnet as mx
from utils import cifar_for_library, yield_mb
from nb_logging import NotebookLogger, output_to, error_to
import codecs

sys.__stdout__ = codecs.getwriter("utf-8")(sys.__stdout__)

nb_teminal_logger = NotebookLogger(sys.stdout.session, sys.stdout.pub_thread, sys.stdout.name, sys.__stdout__)

rst_out = output_to(nb_teminal_logger)
rst_err = error_to(nb_teminal_logger)

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("MXNet: ", mx.__version__)

data_path = path.join(os.getenv('AZ_BATCHAI_INPUT_DATASET'), 'cifar-10-batches-py')

def create_symbol():
    data = mx.symbol.Variable('data')
    # size = [(old-size - kernel + 2*padding)/stride]+1
    # if kernel = 3, pad with 1 either side
    conv1 = mx.symbol.Convolution(data=data, num_filter=50, pad=(1,1), kernel=(3,3))
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    conv2 = mx.symbol.Convolution(data=relu1, num_filter=50, pad=(1,1), kernel=(3,3))
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool1 = mx.symbol.Pooling(data=relu2, pool_type="max", kernel=(2,2), stride=(2,2))
    drop1 = mx.symbol.Dropout(data=pool1, p=0.25)
    
    conv3 = mx.symbol.Convolution(data=drop1, num_filter=100, pad=(1,1), kernel=(3,3))
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    conv4 = mx.symbol.Convolution(data=relu3, num_filter=100, pad=(1,1), kernel=(3,3))
    relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu4, pool_type="max", kernel=(2,2), stride=(2,2))
    drop2 = mx.symbol.Dropout(data=pool2, p=0.25)
           
    flat1 = mx.symbol.Flatten(data=drop2)
    fc1 = mx.symbol.FullyConnected(data=flat1, num_hidden=512)
    relu7 = mx.symbol.Activation(data=fc1, act_type="relu")
    drop4 = mx.symbol.Dropout(data=relu7, p=0.5)
    fc2 = mx.symbol.FullyConnected(data=drop4, num_hidden=N_CLASSES) 
    
    input_y = mx.symbol.Variable('softmax_label')  
    m = mx.symbol.SoftmaxOutput(data=fc2, label=input_y, name="softmax")
    return m

def init_model(m):
    if GPU:
        ctx = [mx.gpu(0)]
    else:
        ctx = mx.cpu()
    
    mod = mx.mod.Module(context=ctx, symbol=m)
    mod.bind(data_shapes=[('data', (BATCHSIZE, 3, 32, 32))],
             label_shapes=[('softmax_label', (BATCHSIZE,))])

    # Glorot-uniform initializer
    mod.init_params(initializer=mx.init.Xavier(rnd_type='uniform'))
    mod.init_optimizer(optimizer='sgd', 
                       optimizer_params=(('learning_rate', LR), ('momentum', MOMENTUM), ))
    return mod

get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = cifar_for_library(data_path, channel_first=True)\n\n# Load data-iterator\n#train_iter = mx.io.NDArrayIter(x_train, y_train, BATCHSIZE, shuffle=True)\n# Use custom iterator instead of mx.io.NDArrayIter() for consistency\n# Wrap as DataBatch class\nwrapper_db = lambda args: mx.io.DataBatch(data=[mx.nd.array(args[0])], label=[mx.nd.array(args[1])])\n\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')

get_ipython().run_cell_magic('time', '', '# Load symbol\nsym = create_symbol()')

get_ipython().run_cell_magic('time', '', '# Initialise model\nmodel = init_model(sym)')

get_ipython().run_cell_magic('time', '', "# Train and log accuracy\nmetric = mx.metric.create('acc')\nfor j in range(EPOCHS):\n    #train_iter.reset()\n    metric.reset()\n    #for batch in train_iter:\n    for batch in map(wrapper_db, yield_mb(x_train, y_train, BATCHSIZE, shuffle=True)):\n        model.forward(batch, is_train=True) \n        model.update_metric(metric, batch.label)\n        model.backward()              \n        model.update()\n    print('Epoch %d, Training %s' % (j, metric.get()))")

get_ipython().run_cell_magic('time', '', 'y_guess = model.predict(mx.io.NDArrayIter(x_test, batch_size=BATCHSIZE, shuffle=False))\ny_guess = np.argmax(y_guess.asnumpy(), axis=-1)')

print("Accuracy: ", float(sum(y_guess == y_test))/len(y_guess))

