import os
os.chdir('..')

import sys
sys.path.insert(0, './python')
import caffe

from pylab import *
get_ipython().run_line_magic('matplotlib', 'inline')

# Download and prepare data
get_ipython().system('data/mnist/get_mnist.sh')
get_ipython().system('examples/mnist/create_mnist.sh')

from caffe import layers as L
from caffe import params as P

def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()
    
with open('examples/mnist/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet('examples/mnist/mnist_train_lmdb', 64)))
    
with open('examples/mnist/lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet('examples/mnist/mnist_test_lmdb', 100)))

get_ipython().system('cat examples/mnist/lenet_auto_train.prototxt')

get_ipython().system('cat examples/mnist/lenet_auto_solver.prototxt')

caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('examples/mnist/lenet_auto_solver.prototxt')

# each output is (batch size, feature dim, spatial dim)
[(k, v.data.shape) for k, v in solver.net.blobs.items()]

# just print the weight sizes (not biases)
[(k, v[0].data.shape) for k, v in solver.net.params.items()]

solver.net.forward()  # train net
solver.test_nets[0].forward()  # test net (there can be more than one)

# we use a little trick to tile the first eight images
imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray')
print solver.net.blobs['label'].data[:8]

imshow(solver.test_nets[0].blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray')
print solver.test_nets[0].blobs['label'].data[:8]

solver.step(1)

imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(4, 5, 5, 5)
       .transpose(0, 2, 1, 3).reshape(4*5, 5*5), cmap='gray')

get_ipython().run_cell_magic('time', '', "niter = 200\ntest_interval = 25\n# losses will also be stored in the log\ntrain_loss = zeros(niter)\ntest_acc = zeros(int(np.ceil(niter / test_interval)))\noutput = zeros((niter, 8, 10))\n\n# the main solver loop\nfor it in range(niter):\n    solver.step(1)  # SGD by Caffe\n    \n    # store the train loss\n    train_loss[it] = solver.net.blobs['loss'].data\n    \n    # store the output on the first test batch\n    # (start the forward pass at conv1 to avoid loading new data)\n    solver.test_nets[0].forward(start='conv1')\n    output[it] = solver.test_nets[0].blobs['ip2'].data[:8]\n    \n    # run a full test every so often\n    # (Caffe can also do this for us and write to a log, but we show here\n    #  how to do it directly in Python, where more complicated things are easier.)\n    if it % test_interval == 0:\n        print 'Iteration', it, 'testing...'\n        correct = 0\n        for test_it in range(100):\n            solver.test_nets[0].forward()\n            correct += sum(solver.test_nets[0].blobs['ip2'].data.argmax(1)\n                           == solver.test_nets[0].blobs['label'].data)\n        test_acc[it // test_interval] = correct / 1e4")

_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')

for i in range(8):
    figure(figsize=(2, 2))
    imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    figure(figsize=(10, 2))
    imshow(output[:50, i].T, interpolation='nearest', cmap='gray')
    xlabel('iteration')
    ylabel('label')

for i in range(8):
    figure(figsize=(2, 2))
    imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    figure(figsize=(10, 2))
    imshow(exp(output[:50, i].T) / exp(output[:50, i].T).sum(0), interpolation='nearest', cmap='gray')
    xlabel('iteration')
    ylabel('label')

