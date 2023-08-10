import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())

input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)

target = Variable(torch.arange(1, 11))
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(out, target)
print(loss)

print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0])

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward(retain_graph=True)

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

get_ipython().run_cell_magic('time', '', 'import torch.optim as optim\n\nnet.zero_grad()\noptimizer = optim.SGD(net.parameters(), lr = 0.01)\noptimizer.zero_grad()\nfor i in range(10):\n    output = net(input)\n    loss = criterion(output, target)\n    loss.backward()\n    optimizer.step()\n    print(output, target)')

