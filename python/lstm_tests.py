import numpy as np
import scipy as sp
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


get_ipython().run_line_magic('matplotlib', 'inline')

import torch
from torch.autograd import Variable
from torch import Tensor
import torch.nn as nn
from torch.nn.utils import rnn
import torch.nn.functional as F
import torch.optim as optim

#torch.manual_seed(1)

use_cuda = torch.cuda.is_available()
print("Is CUDA available? %s." % (use_cuda))

# Counting number of parameters
# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

ltc = pd.read_csv("~/Crypto/Datasets/cryptocurrencypricehistory/litecoin_price.csv")

ltc.columns

ltc["Open"].plot()

high_prices = ltc.loc[:,'High'].as_matrix()
low_prices = ltc.loc[:,'Low'].as_matrix()
mid_prices = (high_prices+low_prices)/2.0

plt.plot(mid_prices)

train_data = mid_prices[:1000]
test_data = mid_prices[1000:]

scaler = MinMaxScaler()
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)

train_data.shape

plt.plot(train_data)

# Train the Scaler with training data and smooth data
smoothing_window_size = 100
dtrain = np.zeros(train_data.shape)

for di in range(0,1000,smoothing_window_size):
    dtrain[di:di+smoothing_window_size,:] = scaler.fit_transform(train_data[di:di+smoothing_window_size,:])

dtest = scaler.fit_transform(test_data)

dtrain.shape

plt.plot(dtrain)

plt.plot(dtest)

# Now perform exponential moving average smoothing
# So the data will have a smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1
for ti in range(1000):
  EMA = gamma*dtrain[ti] + (1-gamma)*EMA
  dtrain[ti] = EMA

# Used for visualization and test purposes
all_mid_data = np.concatenate([dtrain,dtest],axis=0)

plt.plot(all_mid_data)


class SimpleLSTMSequence(nn.Module):
    
    # Options that can be used for the recurrent module, this makes the class more general without any change 
    # as pytorch has a nice unified interface for the 3 classes
    module_dict = {
                   "RNN": nn.RNN,
                   "LSTM": nn.LSTM,
                   "GRU": nn.GRU,
                   }
    
    def __init__(self, in_size, hid_size, out_size, layers, future=0, bias=True, dropout=0, bidirectional=False, cell="LSTM"):
        super(SimpleLSTMSequence, self).__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.out_size = out_size
        self.layers = layers
        self.future = future
        self.CELL_TYPE = SimpleLSTMSequence.module_dict[cell]
        self.lstm = self.CELL_TYPE(input_size=in_size, hidden_size=hid_size, num_layers=layers, bias=bias, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.linear = nn.Linear(hid_size, out_size)

    def forward(self, data, future=5, hidden=None):
        outputs = []
        out, hidden = self.lstm(data, hidden) if hidden is not None else self.lstm(data)
        out = self.linear(out)
        outputs += [out]
        for i in range(future):# if we should predict the future
           output, hidden = self.lstm(output, hidden)
           output = self.linear(hidden)
           outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return hidden, outputs

    def init_hidden(self, batch_size=1):
        return Variable(torch.zeros(self.layers, batch_size, self.hid_size))


#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(X, Y, batch_size=5):
    loss = 0
    hidden = model.init_hidden(batch_size)
    optimizer.zero_grad()
    hidden, out = model(x, hidden)
    loss = criterion(out, Y)
    loss.backwards()
    optimizer.step()

# train and test data:
# I will be feeding sequences of 30 elements to the RNN, and the output

import numpy as np
import torch

np.random.seed(2)

T = 20
L = 1000
N = 100

x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float64')
torch.save(data, open('traindata.pt', 'wb'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

class Sequence2(nn.Module):

        
    # Options that can be used for the recurrent module, this makes the class more general without any change 
    # as pytorch has a nice unified interface for the 3 classes
    module_dict = {
                   "RNN": nn.RNN,
                   "LSTM": nn.LSTM,
                   "GRU": nn.GRU,
                   }
    
    def __init__(self, in_size, hid_size, out_size, layers, future=0, bias=True, dropout=0, bidirectional=False, cell="LSTM"):
        super(Sequence2, self).__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.out_size = out_size
        self.layers = layers
        self.future = future
        self.CELL_TYPE = Sequence2.module_dict[cell]
        self.lstm = self.CELL_TYPE(input_size=in_size, hidden_size=hid_size, num_layers=layers, bias=bias, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.lstm1 = nn.LSTMCell(in_size, hid_size)
        self.lstm2 = nn.LSTMCell(hid_size, hid_size)
        self.linear = nn.Linear(hid_size, out_size)

    def forward(self, input, future = 0):
        outputs = []
        out, hidden = self.lstm(input)
        out = self.linear(out)
        outputs += [out]
        hidd = hidden
        output = out
        for i in range(future):# if we should predict the future
            output, hidd = self.lstm(output, hidd)
            output = self.linear(output)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs
    
    
def train(seq = Sequence(), data=None):
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = data if data is not None else torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])
    # build the model
    
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    #begin to train
    for i in range(15):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            print(input.shape)
            out = seq(input.view(97,999,1))
            loss = criterion(out.view(97,999), target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 30
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.pdf'%i)
        plt.close()

in_size = 1
hid_size = 30
out_size = 1
layers = 2
bias = True
dropout = 0.
bidirectional = False
batch_first=False

rnn = nn.RNN(input_size=in_size, hidden_size=hid_size, num_layers=layers, bias=bias, dropout=dropout, bidirectional=bidirectional, batch_first=batch_first)

data = torch.load('traindata.pt')
input = torch.from_numpy(data[3:, :-1]).float()
target = torch.from_numpy(data[3:, 1:]).float()
test_input = torch.from_numpy(data[:3, :-1]).float()
test_target = torch.from_numpy(data[:3, 1:]).float()

input.shape

#rnn.forward(input.view(97,999,1))

model = SimpleLSTMSequence(1, 20, 1, 2)#.cuda()
model = Sequence2(1, 30, 1, 2)
train(seq = model)
#train()





