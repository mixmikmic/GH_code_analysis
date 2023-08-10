import mxnet as mx
import mxnet.gluon as gluon
from mxnet import nd, autograd
import matplotlib.pyplot as plt
import numpy as np
import mxnet.autograd as ag
import math
mx.random.seed(1)

loss = gluon.loss.L1Loss()

# getting data ready
output = nd.arange(-5,5,0.01)
output.attach_grad() # we need the gradient
thelabel = nd.zeros_like(output) 
with ag.record():    # start recording
    theloss = loss(output, thelabel)
theloss.backward()   # and compute the gradient

plt.plot(output.asnumpy(), theloss.asnumpy(), label='L1 loss')
plt.plot(output.asnumpy(), output.grad.asnumpy(), label='Gradient of L1 loss')
plt.legend()
plt.show()

loss = gluon.loss.L2Loss()

with ag.record():    # start recording
    theloss = loss(output, thelabel)
theloss.backward()   # and compute the gradient

plt.plot(output.asnumpy(), theloss.asnumpy(), label='L2 loss')
plt.plot(output.asnumpy(), output.grad.asnumpy(), label='Gradient of L2 loss')
plt.legend()
plt.show()

loss = gluon.loss.Huber(rho=0.5)

with ag.record():    # start recording
    theloss = loss(output, thelabel)
theloss.backward()   # and compute the gradient

plt.plot(output.asnumpy(), theloss.asnumpy(), label='Huber loss 0.5')
plt.plot(output.asnumpy(), output.grad.asnumpy(), label='Gradient of Huber loss 0.5')

# and now for the same loss function with rho=1.0, the default
loss = gluon.loss.Huber()

with ag.record():    # start recording
    theloss = loss(output, thelabel)
theloss.backward()   # and compute the gradient

plt.plot(output.asnumpy(), theloss.asnumpy(), label='Huber loss 1')
plt.plot(output.asnumpy(), output.grad.asnumpy(), label='Gradient of Huber loss 1')

plt.legend()
plt.show()

loss = gluon.loss.Quantile(tau=0.2)
with ag.record():    # start recording
    theloss = loss(output, thelabel)
theloss.backward()   # and compute the gradient

plt.plot(output.asnumpy(), theloss.asnumpy(), label='Quantile loss 0.2')
plt.plot(output.asnumpy(), output.grad.asnumpy(), label='Gradient of Quantile loss 0.2')

# and now for the same loss function with tau = 0.6 
loss = gluon.loss.Quantile(tau=0.6)

with ag.record():    # start recording
    theloss = loss(output, thelabel)
theloss.backward()   # and compute the gradient

plt.plot(output.asnumpy(), theloss.asnumpy(), label='Quantile loss 0.6')
plt.plot(output.asnumpy(), output.grad.asnumpy(), label='Gradient of Quantile loss 0.6')

plt.legend()
plt.show()

loss = gluon.loss.EpsilonInsensitive(epsilon=0.5)
with ag.record():    # start recording
    theloss = loss(output, thelabel)
theloss.backward()   # and compute the gradient

plt.plot(output.asnumpy(), theloss.asnumpy(), label='Epsilon-insensitive loss 0.5')
plt.plot(output.asnumpy(), output.grad.asnumpy(), label='Gradient of Epsilon-insensitive loss 0.5')
plt.legend()
plt.show()

loss = gluon.loss.LogCosh()
with ag.record():    # start recording
    theloss = loss(output, thelabel)
theloss.backward()   # and compute the gradient

plt.plot(output.asnumpy(), theloss.asnumpy(), label='LogCosh loss')
plt.plot(output.asnumpy(), output.grad.asnumpy(), label='Gradient of LogCosh loss')
plt.legend()
plt.show()

loss = gluon.loss.Poisson()
with ag.record():    # start recording
    theloss = loss(output, 10 * nd.ones_like(output))
theloss.backward()   # and compute the gradient

plt.plot(output.asnumpy(), theloss.asnumpy(), label='Poisson loss')
plt.plot(output.asnumpy(), output.grad.asnumpy(), label='Gradient of Poisson loss')
plt.legend()
plt.show()

# this implements an L2 norm triplet loss
# max(margin + |f1 - f2|^2 - |f1-f3|^2, 0) per observation
def TripletLoss(f1, f2, f3):
    margin = 1
    loss = nd.sum((f1-f2)**2 - (f1-f3)**2, axis=1) + 1
    loss = nd.maximum(loss, nd.zeros_like(loss))
    return loss

loss = TripletLoss
#with ag.record():    # start recording
#    theloss = loss(output, nd.ones_like(output))
#theloss.backward()   # and compute the gradient
#plt.plot(output.asnumpy(), theloss.asnumpy(), label='Huber Loss')
#plt.plot(output.asnumpy(), output.grad.asnumpy(), label='Gradient')
#plt.legend()
#plt.show()

f1 = nd.random_normal(shape=(5,10))
f2 = nd.random_normal(shape=(5,10))
f3 = nd.random_normal(shape=(5,10))

theloss = loss(f1, f2, f3)
print(theloss)

loss = gluon.loss.Logistic()

# getting data ready
thelabel = nd.ones_like(output)
with ag.record():    # start recording
    theloss = loss(output, thelabel)
theloss.backward()   # and compute the gradient

plt.plot(output.asnumpy(), theloss.asnumpy(), label='Logistic loss')
plt.plot(output.asnumpy(), output.grad.asnumpy(), label='Gradient of logistic loss')
# now compute the loss for y=-1
with ag.record():    # start recording
    theloss = loss(output, -thelabel)
theloss.backward()   # and compute the gradient
plt.plot(output.asnumpy(), theloss.asnumpy(), label='Logistic loss for y=-1')
plt.plot(output.asnumpy(), output.grad.asnumpy(), label='Gradient of loss for y=-1')
plt.legend()
plt.show()

loss = gluon.loss.SoftMargin()

with ag.record():    # start recording
    theloss = loss(output, thelabel)
theloss.backward()   # and compute the gradient

plt.plot(output.asnumpy(), theloss.asnumpy(), label='Soft margin')
plt.plot(output.asnumpy(), output.grad.asnumpy(), label='Gradient')
# now compute the loss for y=-1
theloss = loss(output, -thelabel)
plt.plot(output.asnumpy(), theloss.asnumpy(), label='Soft margin for y=-1')
plt.legend()
plt.show()

loss = gluon.loss.Exponential()

# getting data ready
thelabel = nd.ones_like(output)
with ag.record():    # start recording
    theloss = loss(output, thelabel)
theloss.backward()   # and compute the gradient

plt.plot(output.asnumpy(), theloss.asnumpy(), label='Logistic loss')
plt.plot(output.asnumpy(), output.grad.asnumpy(), label='Gradient of logistic loss')
plt.legend()
plt.show()

loss = gluon.loss.Langford()

with ag.record():    # start recording
    theloss = loss(output, nd.ones_like(output))
theloss.backward()   # and compute the gradient

plt.plot(output.asnumpy(), theloss.asnumpy(), label='VW style loss')
plt.plot(output.asnumpy(), output.grad.asnumpy(), label='Gradient')
# now compute the loss for y=-1
theloss = loss(output, -thelabel)
plt.plot(output.asnumpy(), theloss.asnumpy(), label='VW style loss for y=-1')
plt.legend()
plt.show()

loss = gluon.loss.SoftmaxCrossEntropyLoss()

f = nd.random_normal(shape=(1,10))
y = nd.array([4]) #class 4 is true

print('Softmax loss is {}.'.format(loss(f,y).asscalar()))

# now compute this by hand

p = nd.exp(f)
p = p / nd.sum(p)
print('Class 4 has negative log-likelihood {}.'.format(-nd.log(p[0,4]).asscalar()))

# plain vanilla loss 
loss = gluon.loss.MaxMargin()

# some classes (4 class problem)
label = nd.array([1,3,2])
output = nd.random_normal(shape=(3,4))

print('Function values for 3 problems {}'.format(output))
theloss = loss(output, label)
print('Loss function values {}'.format(theloss))
print('Instantiated loss matrix {}'.format(loss._delta))

# now make things more interesting by changing the loss matrix 
delta = nd.array(loss._delta) #call copy constructor
delta[0,3] = 4
delta[1,3] = 4
delta[2,3] = 4
loss = gluon.loss.MaxMargin(delta)
print('Instantiated loss matrix {}'.format(loss._delta))
print('Function values for 3 problems {}'.format(output))
theloss = loss(output, label)
print('Loss function values {}'.format(theloss))

loss = gluon.loss.KLDivLoss()

# generate some random probability distribution
f = nd.random_normal(shape=(1,10))
p = nd.exp(f)
p = p / nd.sum(p)

# generate some target distribution
y = nd.random_normal(shape=(1,10))
y = nd.exp(y)
y = y / nd.sum(y)

z = nd.zeros_like(y)
z[0,3] = 1

# distance between our estimate p and the 'true' distribution y
print(loss(nd.log(p), y))
# distance to itself - should be zero
print(loss(nd.log(p), p))
# equivalent of logistic loss with class 3 up to normalization over domain, i.e. 1/10
# note that this is VERY DIFFERENT from information theory but a traditional choice
# in deep learning
print(loss(nd.log(p), z))
print(-nd.log(p[0,3]))

# we broke the output data previously
output = nd.arange(-5,5,0.01)
output.attach_grad() # we need the gradient

loss = gluon.loss.DualKL()

lossp = loss(output, -nd.ones_like(output))
lossq = loss(output, nd.ones_like(output))

plt.plot(output.asnumpy(), lossp.asnumpy(), label='Loss for p')
plt.plot(output.asnumpy(), lossq.asnumpy(), label='Loss for q')
plt.legend()
plt.show()

loss = gluon.loss.RelativeNovelty(rho=3)

lossp = loss(output, -nd.ones_like(output))
lossq = loss(output, nd.ones_like(output))

plt.plot(output.asnumpy(), lossp.asnumpy(), label='Loss for p')
plt.plot(output.asnumpy(), lossq.asnumpy(), label='Loss for q')
plt.legend()
plt.show()

loss = gluon.loss.TripletLoss(margin=2)

# make some data. f1 and f2 are similar, f3 is (hopefully) far away
theshape = (5,3)
f1 = nd.normal(shape=theshape)
f2 = nd.normal(shape=theshape)
f3 = nd.normal(shape=theshape) * 5.0

# with the right pair of distances
theloss = loss(f1, f2, f3)
print(theloss)
# these are likely far away in the wrong way, since we blew f3 out of proportions
theloss = loss(f1, f3, f2)
print(theloss)

