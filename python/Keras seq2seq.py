import tensorflow as tf
import numpy as np 
import random
import math
from matplotlib import pyplot as plt
import os
import copy

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns

df = pd.read_csv('./PRSA_data_2010.1.1-2014.12.31.csv')
print(df.head())

cols_to_plot = ["pm2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
i = 1
# plot each column
plt.figure(figsize = (10,12))
for col in cols_to_plot:
    plt.subplot(len(cols_to_plot), 1, i)
    plt.plot(df[col])
    plt.title(col, y=0.5, loc='left')
    i += 1
plt.show()

## Fill NA with 0 
#print(df.isnull().sum())
df.fillna(0, inplace = True)

## One-hot encode 'cbwd'
temp = pd.get_dummies(df['cbwd'], prefix='cbwd')
df = pd.concat([df, temp], axis = 1)
del df['cbwd'], temp

## Split into train and test - I used the last 1 month data as test, but it's up to you to decide the ratio
#df_train = df.iloc[:(-31*24), :].copy()
#df_test = df.iloc[-31*24:, :].copy()
train_size = 365 * 24 * 4
df_train = df.iloc[:(train_size), :].copy()
df_test = df.iloc[train_size:, :].copy()

## take out the useful columns for modeling - you may also keep 'hour', 'day' or 'month' and to see if that will improve your accuracy
X_train = df_train.loc[:, ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv']].values.copy()
X_test = df_test.loc[:, ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv']].values.copy()
y_train = df_train['pm2.5'].values.copy().reshape(-1, 1)
y_test = df_test['pm2.5'].values.copy().reshape(-1, 1)

## z-score transform x - not including those one-how columns!
for i in range(X_train.shape[1]-4):
    temp_mean = X_train[:, i].mean()
    temp_std = X_train[:, i].std()
    X_train[:, i] = (X_train[:, i] - temp_mean) / temp_std
    X_test[:, i] = (X_test[:, i] - temp_mean) / temp_std
    
# z-score transform y
y_mean = y_train.mean()
y_std = y_train.std()
y_train = (y_train - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

input_seq_len = 24*10#30
output_seq_len = 24*10#5

# previous batch size was 10
def generate_train_samples(x = X_train, y = y_train, batch_size = 10, input_seq_len = input_seq_len, output_seq_len = output_seq_len):

    total_start_points = len(x) - input_seq_len - output_seq_len
    start_x_idx = np.random.choice(range(total_start_points), batch_size, replace = False)
    
    input_batch_idxs = [list(range(i, i+input_seq_len)) for i in start_x_idx]
    input_seq = np.take(x, input_batch_idxs, axis = 0)
    
    output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in start_x_idx]
    output_seq = np.take(y, output_batch_idxs, axis = 0)
    
    return input_seq, output_seq # in shape: (batch_size, time_steps, feature_dim)

def generate_test_samples(x = X_test, y = y_test, input_seq_len = input_seq_len, output_seq_len = output_seq_len):
    
    total_samples = x.shape[0]
    
    input_batch_idxs = [list(range(i, i+input_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
    input_seq = np.take(x, input_batch_idxs, axis = 0)
    
    output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
    output_seq = np.take(y, output_batch_idxs, axis = 0)
    
    return input_seq, output_seq

x, y = generate_train_samples()
print(x.shape, y.shape)

test_x, test_y = generate_test_samples()
print(test_x.shape, test_y.shape)

import seq2seq
from seq2seq.models import Seq2Seq, SimpleSeq2Seq

#model = Seq2Seq(batch_input_shape=x.shape, hidden_dim=10, output_length=8, output_dim=20, depth=4)
# model = Seq2Seq(batch_input_shape=x.shape, output_dim=8, output_length=5)
# model.compile(loss='mse', optimizer='rmsprop')

model = Seq2Seq(batch_input_shape=(None, test_x.shape[1], 11), hidden_dim=100, 
                    output_length=output_seq_len, output_dim=1, depth=3)
model.compile(loss='mse', optimizer='rmsprop')

test_x.shape

test_y.shape

yhat = model.predict(test_x)

dim1, dim2 = yhat.shape[0], yhat.shape[1]

preds_flattened = yhat.reshape(dim1*dim2, 1)
unscaled_yhat = pd.DataFrame(preds_flattened, columns=['pm2.5']).apply(lambda x: (x*y_std) + y_mean)
yhat_inv = unscaled_yhat.values

test_y.shape

test_y

# If I didn't scale the test_y set first...

test_y.shape

dim1*dim2

test_y.shape

test_y_flattened = test_y.reshape(dim1*dim2, 1)
df_y_inv = pd.DataFrame(test_y_flattened, columns=['pm2.5'])
#unscaled_y = pd.DataFrame(test_y_flattened, columns=['pm2.5']).apply(lambda x: (x*y_std) + y_mean)
#y_inv = unscaled_y.values

y_inv = df_y_inv.values

pd.concat((df_y_inv,unscaled_yhat),axis=1)



print("Test mse is: ", np.mean((y_inv - yhat_inv)**2))

rmse = np.sqrt(8662)

rmse

plot_test(unscaled_yhat.iloc[:31*24*2,], df_y_inv.iloc[:31*24*2,])

## remove duplicate hours and concatenate into one long array
test_y_expand = np.concatenate([test_y[i].reshape(-1) for i in range(0, test_y.shape[0], 5)], axis = 0)
preds_expand = np.concatenate([test_preds[i].reshape(-1) for i in range(0, test_preds.shape[0], 5)], axis = 0)

preds_expand.shape

def plot_test(preds_expand, test_y_expand):
    fig, ax = plt.subplots(figsize=(17,8))
    ax.set_title("Test Predictions vs. Actual For Last Year")
    ax.plot(preds_expand, color = 'red', label = 'predicted')
    ax.plot(test_y_expand, color = 'green', label = 'actual')
    plt.legend(loc="upper left")
    plt.show()

plot_test(preds_expand[:], test_y_expand[:31*24*2,])

mse = np.mean((final_preds - test_y)**2)

np.sqrt(mse)





