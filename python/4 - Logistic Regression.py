get_ipython().magic('matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn import datasets
from torch.autograd import Variable

cluster_points = 300

X = np.zeros((3*cluster_points, 2))
y = np.zeros(3*cluster_points)

mean = [(40, 50),
        (27, 45),
        (38, 46)]

cov = [[[1,0], [0, 1]],
       [[1, 0], [0, 1]],
       [[1, 0], [0, 1]]]

colors = ['r', 'g', 'b']

for i, (m, c, color) in enumerate(zip(mean, cov, colors)):
    cluster = np.random.multivariate_normal(m, c, cluster_points)
    plt.scatter(cluster[:, 0], cluster[:, 1], s=3, c=color)
    X[(i*cluster_points):((i+1)*cluster_points), :] = cluster
    y[(i*cluster_points):((i+1)*cluster_points)] = i

plt.show()

X = torch.from_numpy(X.astype('float32'))
y = torch.from_numpy(y.astype('int'))

ds = data.TensorDataset(X, y)
data_loader = data.DataLoader(ds, batch_size=128,
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
    
model = LogReg(X.size(1), 3)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

get_ipython().run_cell_magic('time', '', '\nn_epochs = 100\naccuracies = []\n\nfor epoch in range(n_epochs):\n    if not (epoch % (n_epochs / 10)):\n        print("Epoch {}".format(epoch))\n        \n    for (X_batch, y_batch) in data_loader:\n        X_batch = Variable(X_batch)\n        y_batch = Variable(y_batch)\n        \n        y_pred = model(X_batch)\n        loss = loss_func(y_pred, y_batch)\n        \n        optimizer.zero_grad()\n        loss.backward()\n        optimizer.step()\n        \n        _, y_pred = torch.max(y_pred, 1)\n        acc = torch.sum((y_pred == y_batch).data) / y_pred.size(0)\n        accuracies.append(acc)')

plt.title("Learning Curve")
plt.xlabel("Batch")
plt.ylabel("Accuracy")
plt.scatter(np.linspace(1, len(accuracies), len(accuracies)), accuracies, s=1)
plt.show()

model.eval()
y_pred = model(Variable(X))
_, y_pred = torch.max(y_pred, 1)

for x, label in zip(X.numpy(), y_pred.view(-1).data.numpy()):
    plt.scatter(x[0], x[1], s=3, color=colors[label])
    
plt.show()

model.eval()
y_pred = model(Variable(X))
_, y_pred = torch.max(y_pred, 1)

acc = torch.sum((y_pred == Variable(y)).data) / y_pred.size(0)
print(acc)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=10000, max_iter=n_epochs) #higher C, lesser l2
clf.fit(X.numpy(), y.numpy())
acc = clf.score(X.numpy(), y.numpy())
print(acc)

y_pred = clf.predict(X.numpy())

for x, label in zip(X.numpy(), y_pred):
    plt.scatter(x[0], x[1], s=3, color=colors[label])
    
plt.show()

