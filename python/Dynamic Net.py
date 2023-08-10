import random
import torch
from torch.autograd import Variable

class DynamicNet(torch.nn.Module):
    def __init__(self, In_size, H_size, Out_size):
        super(DynamicNet, self).__init__()
        self.input_layer = torch.nn.Linear(In_size, H_size)
        self.middle_layer = torch.nn.Linear(H_size, H_size)
        self.out_layer = torch.nn.Linear(H_size, Out_size)
        
    def forward(self, x):
        h = self.input_layer(x).clamp(min = 0)
        for _ in range(random.randint(0,10)):
            h = self.middle_layer(h).clamp(min = 0)
        out = self.out_layer(h)
        return out
    
M, In_size, H_size, Out_size = 10000, 5, 4, 2

x = Variable(torch.rand(M, In_size), requires_grad = False)  # Row is taking different example
y = Variable(torch.rand(M, Out_size), requires_grad = False) # that's how it's defined in package, 
                                                            # so operation will be col major

model = DynamicNet(In_size, H_size, Out_size)

loss = torch.nn.MSELoss(size_average = False)

learning_rate = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
for t in range(1000):
    out = model(x)
    loss_out = loss(out, y)
    if t%100 == 1:
        print(t, loss_out.data[0])
    optimizer.zero_grad()
    
    loss_out.backward()
    
    optimizer.step()

