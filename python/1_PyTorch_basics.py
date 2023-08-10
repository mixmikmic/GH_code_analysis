import torch
import numpy as np
from __future__ import print_function

x = torch.Tensor(5, 3)
print(x)

x.zero_()

torch.Tensor([[1, 2, 3],  # rank 2 tensor
              [4, 5, 6],
              [7, 8, 9]])

x.size()

x = torch.rand(5, 3)
print(x)

npy = np.random.rand(5, 3)
y = torch.from_numpy(npy)
print(y)

z = x + y

x.type(), y.type()

z = x + y.float()
print(z)

torch.add(x, y.float())

x

x.add_(1)

x

x[:2, :2]

x * y.float()

torch.exp(x)

torch.transpose(x, 0, 1)

#transposing, indexing, slicing, mathematical operations, linear algebra, random numbers

torch.trace(x)

x.numpy()

torch.cuda.is_available()

if torch.cuda.is_available():
    x_gpu = x.cuda()
    print(x_gpu)

from torch.autograd import Variable

x = torch.rand(5, 3)
print(x)

x = Variable(torch.rand(5, 3))
print(x)

x.data

x.creator

x.grad

x = Variable(torch.Tensor([2]), requires_grad=False)
w = Variable(torch.Tensor([3]), requires_grad=True)
b = Variable(torch.Tensor([1]), requires_grad=True)

z = x * w
y = z + b
y

z.creator

y.creator

w.grad

y.backward()

w.grad

a = Variable(torch.Tensor([2]), requires_grad=True)

y = 3*a*a + 2*a + 1

y.backward()

a.grad



