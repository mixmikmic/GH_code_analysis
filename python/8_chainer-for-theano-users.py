import numpy as np

import theano
import theano.tensor as T

import chainer
import chainer.functions as F
import chainer.links as L

class TheanoConvolutionLayer(object):
    
    def __init__(self, input, filter_shape, image_shape):
        # Prepare initial values of the parameter W
        spatial_dim = np.prod(filter_shape[2:])
        fan_in = filter_shape[1] * spatial_dim
        fan_out = filter_shape[0] * spatial_dim
        scale = np.sqrt(3. / fan_in)
        
        # Create the parameter W
        W_init = np.random.uniform(-scale, scale, filter_shape)
        self.W = theano.shared(W_init.astype(np.float32), borrow=True)

        # Create the paramter b
        b_init = np.zeros((filter_shape[0],))
        self.b = theano.shared(b_init.astype(np.float32), borrow=True)

        # Describe the convolution operation
        conv_out = T.nnet.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape)
        
        # Add a bias
        self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        
        # Store paramters
        self.params = [self.W, self.b]

batchsize = 32
input_shape = (batchsize, 1, 28, 28)
filter_shape = (6, 1, 5, 5)

# Create a tensor that represents a minibatch
x = T.fmatrix('x')
input = x.reshape(input_shape)

conv = TheanoConvolutionLayer(input, filter_shape, input_shape)
f = theano.function([input], conv.output)

x_data = np.random.rand(32, 1, 28, 28).astype(np.float32)

y = f(x_data)

print(y.shape, type(y))

class ChainerConvolutionLayer(chainer.Link):
    
    def __init__(self, filter_shape):
        super().__init__()
        with self.init_scope():
            # Specify the way of initialize
            W_init = chainer.initializers.LeCunUniform()
            b_init = chainer.initializers.Zero()
        
            # Create a parameter object
            self.W = chainer.Parameter(W_init, filter_shape)          
            self.b = chainer.Parameter(b_init, filter_shape[0])
            
    def __call__(self, x):
        return F.convolution_2d(x, self.W, self.b)

chainer_conv = ChainerConvolutionLayer(filter_shape)

y = chainer_conv(x_data)

print(y.shape, type(y), type(y.array))

conv_link = L.Convolution2D(in_channels=1, out_channels=6, ksize=(5, 5))

y = conv_link(x_data)

print(y.shape, type(y), type(y.array))

x = T.fmatrix().reshape((32, 1, 28, 28))
W = T.fmatrix().reshape((6, 1, 5, 5))
b = T.fvector().reshape((6,))
conv_out = T.nnet.conv2d(x, W) + b.dimshuffle('x', 0, 'x', 'x')

f = L.TheanoFunction(inputs=[x, W, b], outputs=[conv_out])

class MyNetworkWithTheanoConvolution(chainer.Chain):
    
    def __init__(self, theano_conv):
        super().__init__()
        self.theano_conv = theano_conv
        W_init = chainer.initializers.LeCunUniform()
        b_init = chainer.initializers.Zero()
        with self.init_scope():
            self.W = chainer.Parameter(W_init, (6, 1, 5, 5))
            self.b = chainer.Parameter(b_init, (6,))
            self.l1 = L.Linear(None, 100)
            self.l2 = L.Linear(100, 10)
        
    def __call__(self, x):
        h = self.theano_conv(x, self.W, self.b)
        h = F.relu(h)
        h = self.l1(h)
        h = F.relu(h)
        return self.l2(h)

# Instantiate a model object
model = MyNetworkWithTheanoConvolution(f)

# And give an array/Variable to get the network output
y = model(x_data)

print(y.shape)

t = np.random.randint(0, 10, size=(32,)).astype(np.int32)
loss = F.softmax_cross_entropy(y, t)

model.cleargrads()
loss.backward()

W_gradient = model.W.grad_var.array
b_gradient = model.b.grad_var.array

print(W_gradient.shape, type(W_gradient))
print(b_gradient.shape, type(b_gradient))

