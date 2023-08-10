import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
n = 50
x = np.random.randn(n)

x

y = x * (np.random.randn(n))
colors = np.random.randn(n)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.scatter(x, y, c = colors, alpha = 0.5)
plt.show()

x_values = list()
for i in range(0,11):
    x_values.append(i)

x_values

x_train = np.array(x_values, dtype = np.float32)
print(x_train)
print(type(x_train))
print(x_train.shape)
x_train = x_train.reshape(-1,1)
print(x_train.shape)

y_values = [2*i + 1 for i in x_values]
y_values

y_train = np.array(y_values, dtype = np.float32)
print(y_train)
print(type(y_train))
print(y_train.shape)
y_train = y_train.reshape(-1,1)
print(y_train.shape)

import torch
import torch.nn as nn
from torch.autograd import Variable

class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.linear(x)
        return out

input_dim, output_dim = 1, 1

model = LinearRegression(input_dim, output_dim)

model.train()

criterion = nn.MSELoss()

learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

epochs = 100

for epoch in range(epochs):
    epoch = epoch + 1
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))
    
    optimizer.zero_grad()
    
    ouputs = model(inputs)
    
    loss = criterion(ouputs, labels)
    
    loss.backward()
    
    optimizer.step()
    
    print("Epoch: {}, Loss: {}".format(epoch, loss.data[0]))
    

predicted = model(Variable(torch.from_numpy(x_train))).data.cpu().numpy()

predicted

# y = 2x+1
y_train

plt.clf()

predicted = model(Variable(torch.from_numpy(x_train))).data.cpu().numpy()

plt.plot(x_train, y_train, 'go', label = "Original Data", alpha = 0.5)

plt.plot(x_train, predicted, "--", label = "Predicted Data", alpha = 0.5)

plt.legend(loc = "best")

plt.show()

