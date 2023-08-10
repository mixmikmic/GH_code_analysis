import torch

# Make a Tensor
x = torch.Tensor(4,6) 
print x 
y = torch.rand(4,6)
print y
a = torch.ones(2,4)
print a

# Get size
print 'size:', y.size()

# Change with Numpy (Same memory so changine one changes the other)
#   Tensor to np
b = a.numpy()
print b

#   np to Tensor
import numpy as np
f = np.ones(4)
print f
g = torch.from_numpy(f)
print f

t = torch.ones(4,6)

print 'row 0', t[0,:]
print 'col 2', t[:, 2]

# Operations on Tensors (Docs: http://pytorch.org/docs/master/torch.html)

print torch.add(x, y) # One of the ways to add

print x.add_(y) # Another one of the ways to add

import torch
from torch.autograd import Variable

input_tensor = torch.ones(1)

# Autograd has a `Variable` class which wraps around a `Tensor`
# (requires_grad=True to build the operations map)
inputs = Variable(input_tensor, requires_grad=True) 
print 'input Variable: ', inputs

# Can manipulate Variables
outputs = x**2 + 20
print 'output Variable: ', outputs

# Variable.data will return the `Tensor`
print 'Tensor in output Variable: ', outputs.data

# Variable.grad_fn holds history of all functions and variables 
# involved in reaching to this Variable
print 'Gradient fn to be used to compute gradient: ', outputs.grad_fn

# Backpropagation
# Call variable.backward to move back from output
# Backward takes in a loss etc (refer to NN section)

