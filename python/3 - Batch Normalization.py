get_ipython().magic('matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import torch                                                                                                                                                                                                       
import torchvision
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

data_points = int(1e4)
noise_factor = 700

X = Variable(torch.linspace(9403, 18293, data_points)) # (data_points)
X = torch.unsqueeze(X, 1) # (data_points, 1)

y = Variable(torch.linspace(455, 2320, data_points))
y = torch.unsqueeze(y, 1)

noise = noise_factor * Variable(torch.randn(data_points))
y = y.add(noise)

plt.scatter(X.data.numpy(), y.data.numpy(), s=1)
plt.show()

ds = data.TensorDataset(X.data, y.data)
data_loader = data.DataLoader(ds, batch_size=500,
                              shuffle=True,
                              num_workers=4)

class LinReg(nn.Module):
    def __init__(self, in_size, out_size):
        super(LinReg, self).__init__()
        self.lin = nn.Linear(in_size, out_size)
        self.bn = nn.BatchNorm1d(in_size)
        
    def forward(self, X):
        out = self.lin(X)
        out = self.bn(out)
        return out
    
model = LinReg(1,1)
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5e-3)

get_ipython().run_cell_magic('time', '', '\nn_epochs = 5\nlosses = []\n\nfor epoch in range(n_epochs):\n    for (X_batch, y_batch) in data_loader:\n        X_batch = Variable(X_batch)\n        y_batch = Variable(y_batch)\n        \n        y_pred = model(X_batch)\n        loss = loss_func(y_pred, y_batch)\n        losses.append(loss.data[0])\n        \n        optimizer.zero_grad()\n        loss.backward()\n        optimizer.step()')

# Sets model to evaluation mode
model.eval()

y_pred = model(X)
plt.scatter(X.data.numpy(), y.data.numpy(), s=1)
plt.plot(X.data.numpy(), y_pred.data.numpy(), 'r')
plt.show()

plt.title("Learning Curve")
plt.xlabel("Batch")
plt.ylabel("MSE Loss")
plt.scatter(np.linspace(1, len(losses), len(losses)), losses, s=3)
plt.show()

for i, module in enumerate(model.modules()):
    print(i, module)

list(model.lin.parameters())

model.state_dict().keys()



