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

print('shape of the data = '+str(train_df.shape))
train_df['class'].value_counts()

import torch
dtype = torch.FloatTensor

dict1 = {'Normal':0 , 'Abnormal':1}
train_df['class'] = train_df['class'].map(dict1)
target = train_df['class'].values
del train_df['class']
train_x = train_df.as_matrix()
x_tor = torch.from_numpy(train_x)

x_tor.shape

y_tor = torch.from_numpy(target).type(dtype)
y_tor.shape

n_feature = x_tor.shape[1]
n_output = 1
W = torch.zeros(n_output,n_feature).type(dtype)
b = 0
alpha = .00000002
n_iter = 5000

cost = []
iter = []
x_train = x_tor.t().type(dtype)
m = x_tor.shape[0]
for t in range(n_iter):
    ## forward propagatiop
    z = W.mm(x_train)+b
    y_pred = torch.sigmoid(z).type(dtype)
    
    ## computing log loss 
    loss = -(torch.sum(torch.log(y_pred)*y_tor)+torch.sum(torch.log(1-y_pred)*(1-y_tor)))/m
    cost.append(loss)
    iter.append(t)
    
    ## computing the gradient
    dW = x_train.mm((y_pred-y_tor).t())
    db = torch.sum((y_pred-y_tor).t())
    
    ##updating W,b
    W = W - alpha*dW.t()
    b = b - alpha*db

import matplotlib.pyplot as plt
plt.scatter(iter,cost)
plt.show()

print('Parameters after training = ',W,b)

y_pred[y_pred>=.5] = 1
y_pred[y_pred<.5] = 0
print("Accuracy after training : "+str(torch.sum(y_pred==y_tor)/m))



