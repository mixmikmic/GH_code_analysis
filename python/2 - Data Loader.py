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

data_points = int(1e6)
noise_factor = 30

X = Variable(torch.linspace(-1, 1, data_points)) # (data_points)
X = torch.unsqueeze(X, 1) # (data_points, 1)

y = Variable(torch.linspace(0, 100, data_points))
y = torch.unsqueeze(y, 1)

noise = noise_factor * Variable(torch.randn(data_points))
y = y.add(noise)

ds = data.TensorDataset(X.data, y.data)
data_loader = data.DataLoader(ds, batch_size=500,
                              shuffle=True,
                              num_workers=4)

model = nn.Linear(1,1)
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5e-2)

get_ipython().run_cell_magic('time', '', '\nn_epochs = 1\nlosses = []\n\nfor epoch in range(n_epochs):\n    for (X_batch, y_batch) in data_loader:\n        X_batch = Variable(X_batch)\n        y_batch = Variable(y_batch)\n        \n        y_pred = model(X_batch)\n        loss = loss_func(y_pred, y_batch)\n        losses.append(loss.data[0])\n        \n        optimizer.zero_grad()\n        loss.backward()\n        optimizer.step()')

y_pred = model(X)
plt.scatter(X.data.numpy(), y.data.numpy(), s=1)
plt.plot(X.data.numpy(), y_pred.data.numpy(), 'r')
plt.show()

plt.title("Learning Curve")
plt.xlabel("Batch")
plt.ylabel("MSE Loss")
plt.scatter(np.linspace(1, len(losses), len(losses)), losses, s=3)
plt.show()



