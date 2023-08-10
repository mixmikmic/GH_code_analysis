import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2],[3,4]])            # build a tensor
variable = Variable(tensor, requires_grad=True)      # build a variable, usually for compute gradients

print(tensor)       # [torch.FloatTensor of size 2x2]
print(variable)     # [torch.FloatTensor of size 2x2]

t_out = torch.mean(tensor*tensor)       # x^2
v_out = torch.mean(variable*variable)   # x^2
print(t_out)
print(v_out)

v_out.backward()    # backpropagation from v_out

variable.grad

variable # this is data in variable format

variable.data # this is data in tensor format

variable.data.numpy() # numpy format

type(v_out)

type(v_out.data)



