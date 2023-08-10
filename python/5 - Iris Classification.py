get_ipython().magic('matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import torch                                                                                                                                                                                                       
import torchvision
import torch.nn as nn
import torch.utils.data as data
from sklearn import datasets
from torch.autograd import Variable

iris = datasets.load_iris()
X = torch.from_numpy(iris.data.astype('float32'))

y = torch.from_numpy(iris.target)

ds = data.TensorDataset(X, y)
data_loader = data.DataLoader(ds, batch_size=50,
                              shuffle=True,
                              num_workers=4)

class LogReg(nn.Module):
    def __init__(self, in_size, n_classes):
        super(LogReg, self).__init__()
        self.lin = nn.Linear(in_size, n_classes)
        self.bn = nn.BatchNorm1d(n_classes)

        
    def forward(self, X):
        out = self.lin(X)
        out = self.bn(out)
        return out
    
model = LogReg(X.size(1),3)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

get_ipython().run_cell_magic('time', '', '\nn_epochs = 1000\naccuracies = []\n\nfor epoch in range(n_epochs):\n    if not (epoch % (n_epochs / 10)):\n        print("Epoch {}".format(epoch))\n        \n    for (X_batch, y_batch) in data_loader:\n        X_batch = Variable(X_batch)\n        y_batch = Variable(y_batch)\n        \n        y_pred = model(X_batch)\n        loss = loss_func(y_pred, y_batch)\n        \n        optimizer.zero_grad()\n        loss.backward()\n        optimizer.step()\n        \n        _, y_pred = torch.max(y_pred, 1)\n        acc = torch.sum((y_pred == y_batch).data) / y_pred.size(0)\n        accuracies.append(acc)')

plt.title("Learning Curve")
plt.xlabel("Batch")
plt.ylabel("Accuracy")
plt.scatter(np.linspace(1, len(accuracies), len(accuracies)), accuracies, s=3)
plt.show()

