# Import libs
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import random
import time

# Setup globals
batch_size = 1
in_features = 10
hidden = 20
out_features = 1

# Sequential API example
# Create model
model = nn.Sequential(
    nn.Linear(in_features, hidden),
    nn.ReLU(),
    nn.Linear(hidden, out_features)
)
print(model)

# Create dummy input
x = Variable(torch.randn(batch_size, in_features))
# Run forward pass
output = model(x)
print(output)

# Functional API example
# Create model
class CustomNet(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        """
        Create three linear layers
        """
        super(CustomNet, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, out_features)

    def forward(self, x):
        """
        Draw a random number from [0, 10]. 
        If it's 0, skip the second layer. Otherwise loop it!
        """
        x = F.relu(self.linear1(x))
        while random.randint(0, 10) != 0: 
        #while x.norm() > 2:
            print('2nd layer used')
            x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

custom_model = CustomNet(in_features, hidden, out_features)
print(custom_model)

# Run forward pass with same dummy variable
output = custom_model(x)
print(output)

# ConvNet example

# Debug example
# Create Convnet
class ConvNet(nn.Module):
    def __init__(self, in_channels, hidden, out_features):
        """
        Create ConvNet with two parallel convolutions
        """
        super(ConvNet, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=10,
                                 kernel_size=3,
                                 padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=10,
                                 kernel_size=3,
                                 padding=1)
        self.conv2 = nn.Conv2d(in_channels=20,
                               out_channels=1,
                               kernel_size=3,
                               padding=1)
        self.linear1 = nn.Linear(hidden, out_features)

    def forward(self, x):
        """
        Pass input through both ConvLayers and stack them afterwards
        """
        x1 = F.relu(self.conv1_1(x))
        x2 = F.relu(self.conv1_2(x))
        x = torch.cat((x1, x2), dim=1)
        x = self.conv2(x)
        print('x size (after conv2): {}'.format(x.shape))
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x
    
conv_model = ConvNet(in_channels=3, hidden=576, out_features=out_features)
# Create dummy input
x_conv = Variable(torch.randn(batch_size, 3, 24, 24))

# Run forward pass
output = conv_model(x_conv)
print(output)

## Dataset / DataLoader example
# Create a random Dataset
class RandomDataset(Dataset):
    def __init__(self, nb_samples, consume_time=False):
        self.data = torch.randn(nb_samples, in_features)
        self.target = torch.randn(nb_samples, out_features)
        self.consume_time=consume_time

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        # Transform data
        x = x + torch.FloatTensor(x.shape).normal_() * 1e-2
        
        if self.consume_time:
            # Do some time consuming operation
            for i in xrange(5000000):
                j = i + 1

        return x, y

    def __len__(self):
        return len(self.data)

# Training loop
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
def train(loader):
    for batch_idx, (data, target) in enumerate(loader):
        # Wrap data and target into a Variable
        data, target = Variable(data), Variable(target)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Weight update
        optimizer.step()

        print('Batch {}\tLoss {}'.format(batch_idx, loss.data.numpy()[0]))

# Create Dataset
data = RandomDataset(nb_samples=30)
# Create DataLoader
loader = DataLoader(dataset=data,
                    batch_size=batch_size,
                    num_workers=0,
                    shuffle=True)

# Start training
t0 = time.time()
train(loader)
time_fast = time.time() - t0
print('Training finished in {:.2f} seconds'.format(time_fast))

# Create time consuming Dataset
data_slow = RandomDataset(nb_samples=30, consume_time=True)
loader_slow = DataLoader(dataset=data_slow,
                         batch_size=batch_size,
                         num_workers=0,
                         shuffle=True)
# Start training
t0 = time.time()
train(loader_slow)
time_slow = time.time() - t0
print('Training finished in {:.2f} seconds'.format(time_slow))

loader_slow_multi_proc = DataLoader(dataset=data_slow,
                                    batch_size=batch_size,
                                    num_workers=4,
                                    shuffle=True)
# Start training
t0 = time.time()
train(loader_slow_multi_proc)
time_multi_proc = time.time() - t0
print('Training finished in {:.2f} seconds'.format(time_multi_proc))

