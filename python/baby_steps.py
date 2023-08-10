import torch
import torchvision

# See if gpu computation is possible
torch.cuda.is_available()

# Make a simple random tensor with two rows and three columns
d = torch.Tensor(2, 3)
d

# Verify that the tensor in fact is not on gpu
d.is_cuda

d.abs()

d

d.size()

# Number of elements
torch.numel(d)

torch.linspace(0,5, steps=11)

torch.logspace(0, 5, steps=11)

torch.eye(4)

e = torch.randn(2,3)
e

e.t()

e.clamp(0, e.max())

e.add(1)

e.cos()

e.div(2)

e.mul(3)

e.mean(), e.std()

e.max(), e.min()

f = torch.Tensor(2,3)
torch.ones(2,3, out=f)

f

g = torch.randn(4, 3)
g

h = torch.randn(4, 3)
h

# Cross Product
torch.cross(g, h)

i = torch.rand(4)
i

torch.diag(i)

j = torch.rand(4, 3)
j

## This is the trace operator, which adds up all the diagonal elements
j.trace()

k = torch.randn(4)
k

l = torch.randn(4)
l

## Dot Product
torch.dot(k, l)

m = torch.randn(4, 4)
m

m.eig()

n = torch.randn(4, 3)
o = torch.randn(3, 6)

## Matrix multiplication
torch.mm(n, o)

## Singular Value Decomposition 
n.svd()



