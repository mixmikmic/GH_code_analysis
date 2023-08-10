from chainer.datasets import mnist

# Download the MNIST data if you haven't downloaded it yet
train, test = mnist.get_mnist(withlabel=True, ndim=1)

# set matplotlib so that we can see our drawing inside this notebook
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# Display an example from the MNIST dataset.
# `x` contains the input image array and `t` contains that target class
# label as an integer.
x, t = train[0]
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.show()
print('label:', t)

from chainer import iterators

# Choose the minibatch size.
batchsize = 128

train_iter = iterators.SerialIterator(train, batchsize)
test_iter = iterators.SerialIterator(test, batchsize,
                                     repeat=False, shuffle=False)

import chainer
import chainer.links as L
import chainer.functions as F

class MLP(chainer.Chain):

    def __init__(self, n_mid_units=100, n_out=10):
        # register layers with parameters by super initializer
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1=L.Linear(None, n_mid_units)
            self.l2=L.Linear(None, n_mid_units)
            self.l3=L.Linear(None, n_out)

    def __call__(self, x):
        # describe the forward pass, given x (input data)
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

gpu_id = 0  # change to -1 if not using GPU

model = MLP()
if gpu_id >= 0:
    model.to_gpu(gpu_id)

print('The shape of the bias of the first layer, l1, in the model、', model.l1.b.shape)
print('The values of the bias of the first layer in the model after initialization、', model.l1.b.data)

from chainer import optimizers

# Choose an optimizer algorithm
optimizer = optimizers.SGD(lr=0.01)
# Give the optimizer a reference to the model so that it
# can locate the model's parameters.
optimizer.setup(model)

import numpy as np
from chainer.dataset import concat_examples
from chainer.cuda import to_cpu

max_epoch = 10

while train_iter.epoch < max_epoch:
    
    # ---------- The first iteration of training loop ----------
    train_batch = train_iter.next()
    image_train, target_train = concat_examples(train_batch, gpu_id)
    
    # calculate the prediction of the model
    prediction_train = model(image_train)

    # calculation of loss function, softmax_cross_entropy
    loss = F.softmax_cross_entropy(prediction_train, target_train)

    # calculate the gradients in the model
    model.cleargrads()
    loss.backward()

    # update the parameters of the model
    optimizer.update()
    # --------------- until here One loop ----------------
    
    # Check if the generalization of the model is improving 
    # by measuring the accuracy of prediction after every epoch

    if train_iter.is_new_epoch:  # after finishing the first epoch

        # display the result of the loss function
        print('epoch:{:02d} train_loss:{:.04f} '.format(
            train_iter.epoch, float(to_cpu(loss.data))), end='')

        test_losses = []
        test_accuracies = []
        while True:
            test_batch = test_iter.next()
            image_test, target_test = concat_examples(test_batch, gpu_id)

            # forward the test data
            prediction_test = model(image_test)

            # calculate the loss function
            loss_test = F.softmax_cross_entropy(prediction_test, target_test)
            test_losses.append(to_cpu(loss_test.data))

            # calculate the accuracy
            accuracy = F.accuracy(prediction_test, target_test)
            accuracy.to_cpu()
            test_accuracies.append(accuracy.data)
            
            if test_iter.is_new_epoch:
                test_iter.epoch = 0
                test_iter.current_position = 0
                test_iter.is_new_epoch = False
                test_iter._pushed_position = None
                break

        print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(
            np.mean(test_losses), np.mean(test_accuracies)))

from chainer import serializers

serializers.save_npz('my_mnist.model', model)

# check if the model is saved.
get_ipython().magic('ls -la my_mnist.model')

# Create the infrence (evaluation) model as the previous model
infer_model = MLP()

# Load the saved parameters into the parameters of the new inference model to overwrite 
serializers.load_npz('my_mnist.model', infer_model)

# Send the model to utilize GPU by to_GPU
if gpu_id >= 0:
    infer_model.to_gpu(gpu_id)

# Get a test image and label
x, t = test[0]
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.show()
print('label:', t)

from chainer.cuda import to_gpu

# change the shape to minibutch. 
# In this example, the size of minibatch is 1. 
# Inference using any mini-batch size can be performed.

print(x.shape, end=' -> ')
x = x[None, ...]
print(x.shape)

# to calculate by GPU, send the data to GPU, too. 
if gpu_id >= 0:
    x = to_gpu(x, 0)

# forward calculation of the model by sending X
y = infer_model(x)

# The result is given as Variable, then we can take a look at the contents by the attribute, .data. 
y = y.data

# send the gpu result to cpu
y = to_cpu(y)

# The most probable number by looking at the argmax
pred_label = y.argmax(axis=1)

print('predicted label:', pred_label[0])

