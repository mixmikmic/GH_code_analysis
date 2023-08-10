get_ipython().magic('matplotlib inline')
from torch.autograd import Variable, Function

import collections
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import torch
import torch.nn.functional as F

# Number of features
num_features = 5

# Function to generate Dummy data
# Returns X (batch_size, num_features) and y (batch_size)
def get_batch(batch_size=20):
    X = torch.randn((batch_size, num_features))
    X = torch.normal(torch.zeros(batch_size, num_features), torch.zeros(batch_size, num_features).fill_(0.5))
    y = torch.mm(X, torch.randn((num_features, 1)))
    
    # Add noise
    y += torch.normal(torch.zeros(batch_size), torch.zeros(batch_size).fill_(0.1))
    y += torch.Tensor((batch_size)).uniform_(-1, 1)
    
    return Variable(X), Variable(y)

# Linear function
fc = torch.nn.Linear(num_features, 1)

# MSE loss function
mse_loss = torch.nn.MSELoss()

# SGD/Adam optimizer
optimizer = torch.optim.Adam(fc.parameters(), lr=0.05)

# Get the training data
X_train, y_train = get_batch()

for batch_idx in range(20):
    
    # Reset gradients
    fc.zero_grad()
    
    # Forward pass
    output = fc(X_train)
    
    # Compute the loss using the predicted output and y_train
    # loss = F.smooth_l1_loss(output, y_train)
    loss = mse_loss(output, y_train)
    
    # Backward pass
    loss.backward()
    
    # Apply gradients
    optimizer.step()

    # Stop criterion
    if loss.data[0] < 1e-3:
        break
        
    print('Loss: {:.6f} Batch: {}'.format(loss.data[0], batch_idx))

# print weights for debugging
print('Weights: {}'.format(fc.weight))
print('Biases: {}'.format(fc.bias))

# Scikit-learn - for comparison!
from sklearn import linear_model
lr = linear_model.LinearRegression()
lr.fit(X_train.data.numpy(), y_train.data.numpy())
y_lr = lr.predict(X_train.data.numpy())

# Plot the actual against the predicted Y
plt.figure(figsize=(10, 6))
plt.plot(y_train.data.numpy().reshape(y_train.size()[0]), label='actual')
plt.plot(output.data.numpy().reshape(output.size()[0]), label='predicted')
plt.plot(y_lr, label='sklearn')
plt.legend()



