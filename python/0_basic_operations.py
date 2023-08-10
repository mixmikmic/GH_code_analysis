import torch
import numpy as np

# Let's create a 2x2 matrix filled with 2s
A = torch.Tensor(2, 3) # creates a tensor of shape (2, 3)
A.fill_(2.) # fills the tensor with 2s. In PyTorch, operations postfixed by `_` are in place

# Or, simply...
B = 2. * torch.ones(3, 2)

print A, B

print A + A, A + 3

A_T =  A.t() # or A.t_() for in place transposition
AB = A.mm(B) # computes A.B (matrix multiplication), equivalent to A @ B in Python 3.5+
A_h = A * A # computes the element-wise matrix multiplication (Hadamard product)
print A_T, AB, A_h

# Applying a function element-wise to a Tensor
f =  lambda x: x * x
fA = f(A)

# Or, simply
A.apply_(lambda x: x * x)

A = np.ones((2, 3))
A = torch.from_numpy(A) # Casting an array from NumPy to PyTorch...
A = A.numpy() # ... and back to NumPy

