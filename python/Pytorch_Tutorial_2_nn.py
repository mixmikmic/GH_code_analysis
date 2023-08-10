# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train_df = pd.read_csv('../input/column_2C_weka.csv')
train_df.head()

import torch
from torch.autograd import Variable
dtype = torch.FloatTensor

dict1 = {'Normal':0 , 'Abnormal':1}
train_df['class'] = train_df['class'].map(dict1)
target = train_df['class'].values
del train_df['class']
train_x = train_df.as_matrix()
train_x = Variable(torch.from_numpy(train_x).type(dtype))
target = Variable(torch.from_numpy(target).type(dtype))

m = train_df.shape[1]
n_input = m
n_hidden = 10
n_output = 1
nn_model = torch.nn.Sequential(
    torch.nn.Linear(n_input,n_hidden),
    torch.nn.ReLU(),
    torch.nn.Linear(n_hidden,n_output),
    torch.nn.Sigmoid())
loss_fn = torch.nn.BCELoss() ## creating an object of Binary Cross entropy loss 
alpha = .002 ## learning rate
n_iter = 1000 ## no of iterations

cost = []
iter1 = []
y_pred 
for t in range(n_iter):
    ## forward propagation to compute the predicted output y_pred form input data train_x
    y_pred = nn_model(train_x)
    
    ## computing the cost
    loss = loss_fn(y_pred,target)
    print('iter'+str(t)+' loss = '+str(loss.data[0]))
    cost.append(loss.data[0])
    iter1.append(t)
    
    ## inialize the grads to zero before computing them
    nn_model.zero_grad()
    
    ## computes the gradients and stores them to param.grad
    loss.backward()
    
    ## updating the params
    for param in nn_model.parameters():
        param.data =  param.data - alpha * param.grad.data

import matplotlib.pyplot as plt
plt.scatter(iter1[20:],cost[20:])
plt.show()

predicted = nn_model(train_x).type(dtype)
predicted[predicted>=.5]=1
predicted[predicted<.5]=0
predicted = predicted.squeeze()
torch.sum(predicted.data==target.data)/310



