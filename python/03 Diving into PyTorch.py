# Importing PyTorch and Numpy
import torch
import numpy as np
from torch.autograd import Variable

# Declaring Variable in Pytorch
a = Variable(torch.Tensor([19]))
b = torch.Tensor([97])

# Types
print(a)
print(b)
print(type(a))
print(type(b))

# Variables and Operations
x = Variable(torch.Tensor([1]), requires_grad=True)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)

# 2 * 1 + 3
y = w * x + b

y

# Compute the gradients
y.backward()

print(x.grad)    # x.grad = 2 
print(w.grad)    # w.grad = 1 
print(b.grad)    # b.grad = 1 

# Tensors of MxN Dimensions.

# Creates a Tensor (A Matrix) of size 5x3.
t = torch.Tensor(5, 3)

print(t)
print(t.size())

# Operations on Tensors

# Creating Tensors
p = torch.Tensor(4,4)
q = torch.Tensor(4,4)
ones = torch.ones(4,4)

print(p, y, ones)

print("Addition:{}".format(p + q))
print("Subtraction:{}".format(p - ones))
print("Multiplication:{}".format(p * ones))
print("Division:{}".format(q / ones))

# Creating a basic Nueral Network in PyTorch

x = Variable(torch.randn(5, 3))
y = Variable(torch.randn(5, 2))

# Importing NN
import torch.nn as nn

linear = nn.Linear(3, 2)
print(linear)

# Type of Linear
print(type(linear))

print ('Weights: ', linear.weight)
print ('Bias: ', linear.bias)

pred = linear(x)
print(pred)

