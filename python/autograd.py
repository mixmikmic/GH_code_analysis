get_ipython().run_line_magic('reset', '-f')
import torch
from torch.autograd import Variable
x=Variable(torch.ones(1,1),requires_grad=True)
y= x +3
target = 6
# Lets minimize a simple squared loss
loss = (y-target)**2
print("Loss",loss) 

# Lets look at the differential
loss.backward()
print("Gradient",x.grad)

# Step a little towards the minimum
x=x-0.01*x.grad
x=x.detach()

print("New Parameter Value",x)

y=x+3
loss=(y-target)**2
print("New Loss",loss)



