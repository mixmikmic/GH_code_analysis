# the required imports
import numpy as np
import pandas as pd
from linear_aproximation import Model
from environment import network
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

path = '/Users/mawongh/OneDrive/REFERENCE FILE/D/Disertation/brainstorming/'
dataset = pd.read_pickle(path + 'full_dataset.pickle')

dataset.tail(3)

np.random.seed(1898)
N = len(dataset)
sample_sizes = np.array([N * .25, N * .40, N * .50]).astype(int)
sample_indexes = [np.random.choice(np.arange(N), size = sz, replace=False)
                 for sz in sample_sizes]
# np.random.choice(np.arange(10), size =5, replace=False)
datasets = [dataset.iloc[idx] for idx in sample_indexes]

datasets[2].head()

len(y)

# baseline model 1 - global average
ds_idx = 2
# feature selection, only the rewards
y = datasets[ds_idx].reward.values

#splitting dataset into train/test
y_train, y_test = train_test_split(y, test_size=0.15, random_state=42)

#training the model

global_average = np.mean(y_train)

#predicting
y_hat = np.repeat(global_average, len(y_test))
x = np.arange(len(y_test))

plt.plot(x, y_test)
plt.plot(x, y_hat)
plt.show()


print('datasize: {}'.format(len(datasets[ds_idx])))

y_train_hat = np.repeat(global_average, len(y_train))
train_MSE = mean_squared_error(y_train, y_train_hat)
print('train set MSE: {}'.format(train_MSE))

test_MSE = mean_squared_error(y_test, y_hat)
print('test set MSE: {}'.format(test_MSE))

# baseline model 2 - based on actions
# ds_idx =2

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)

X = enc.fit_transform(datasets[ds_idx].action.values.reshape(-1,1))
# X = np.array([model.sa2x_v1(datasets[idx].state[i], int(datasets[idx].action[i])) 
#               for i in datasets[idx].index])
y = datasets[ds_idx].reward.values

# gets the training and test sets
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.15, random_state=42)

reg = linear_model.SGDRegressor(alpha = 0.001, n_iter = 50)
print('fitting model 2, based on actions...')
print(reg)
reg.fit(X_train, y_train)
# print(reg.coef_)

# Do the prediction and calculate the performance (MSE) for model 1
# Xtest_transformed = scaler.transform(X_test)
# x = np.arange(len(Xtest_transformed))
x = np.arange(len(X_test))
y_hat = reg.predict(X_test)
plt.plot(x, y_test)
plt.plot(x, y_hat)
plt.show()

print('datasize: {}'.format(len(datasets[ds_idx])))
y_train_hat = reg.predict(X_train)
train_MSE = mean_squared_error(y_train, y_train_hat)
print('train set MSE: {}'.format(train_MSE))

test_MSE = mean_squared_error(y_test, y_hat)
print('test set MSE: {}'.format(test_MSE))

# model 3 - full linear model with domain knowledge inclusion

# ds_idx =2
# # Instantiate the model that includes the state to features implementation of the function
model = Model()

print('Constructing the features....')
X = np.array([model.sa2x_v1(datasets[ds_idx].state[i], int(datasets[ds_idx].action[i])) 
              for i in datasets[ds_idx].index])
y = datasets[ds_idx].reward.values

# gets the training and test sets
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.15, random_state=42)

# the linear model
reg = linear_model.SGDRegressor(alpha = 0.001, n_iter = 50)
print('fitting model 3...')
print(reg)
reg.fit(X_train, y_train)
# print(reg.coef_)

# Do the prediction and calculate the performance (MSE) for model 1
# Xtest_transformed = scaler.transform(X_test)
# x = np.arange(len(Xtest_transformed))
x = np.arange(len(X_test))
y_hat = reg.predict(X_test)
plt.plot(x, y_test)
plt.plot(x, y_hat)
plt.show()

print('datasize: {}'.format(len(datasets[ds_idx])))
y_train_hat = reg.predict(X_train)
train_MSE = mean_squared_error(y_train, y_train_hat)
print('train set MSE: {}'.format(train_MSE))

test_MSE = mean_squared_error(y_test, y_hat)
print('test set MSE: {}'.format(test_MSE))

# plt.plot(dataset.index, dataset['01_step'])
# plt.show()
# plt.plot(dataset.index, dataset.reward)
# plt.show()

# # Applying the function to the state and adding the location
# state = [transform(s) for s in dataset.state]
# raw_state = np.array([np.append(loc, s) for s in state])

# # Instantiate the model that includes the state to features implementation of the function
model = Model()

# print(type(raw_state))
# print(type(dataset_sub1.action.values))
# i = 500
# X = model.sa2x_v1(raw_state[20], int(dataset_sub1.action[20]))
# print(X)
# # contructs the features and creates the X (features vectors) and the y (target)
# # Xraw = raw_state.copy()
idx = 0

X = np.array([model.sa2x_v1(datasets[idx].state[i], int(datasets[idx].action[i])) 
              for i in datasets[idx].index])
y = datasets[idx].reward.values

# s = [[1,2], [3,4], [5,6]]
# a = 0,15,45
# for i,j in zip(s,a):
#     print(i,j)

# gets the training and test sets
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.15, random_state=42)

# Xraw_train, Xraw_test, yraw_train, yraw_test = train_test_split(
#                                     Xraw, y, test_size=0.10, random_state=42)

# baseline model 1 - global average
global_average = np.mean(y_train)
y_hat = np.repeat(global_average, len(y_test))
x = np.arange(len(X_test))

plt.plot(x, y_test)
plt.plot(x, y_hat)
plt.show()

y_train_hat = np.repeat(global_average, len(y_train))
train_MSE = mean_squared_error(y_train, y_train_hat)
print('train set MSE: {}'.format(train_MSE))

test_MSE = mean_squared_error(y_test, y_hat)
print('test set MSE: {}'.format(test_MSE))

# baseline 2 - average based on actions
df = datasets[0].groupby('action')
df2 = df.agg({'reward':'mean'})
# df2.reward[125]


len(X_train)







reg = linear_model.SGDRegressor(alpha = 0.001, n_iter = 50)
print('fitting model 1B...')
print(reg)
reg.fit(X_train, y_train)
# print(reg.coef_)

# Do the prediction and calculate the performance (MSE) for model 1
# Xtest_transformed = scaler.transform(X_test)
# x = np.arange(len(Xtest_transformed))
x = np.arange(len(X_test))
y_hat = reg.predict(X_test)
plt.plot(x, y_test)
plt.plot(x, y_hat)
plt.show()

y_train_hat = reg.predict(X_train)
train_MSE = mean_squared_error(y_train, y_train_hat)
print('train set MSE: {}'.format(train_MSE))

test_MSE = mean_squared_error(y_test, y_hat)
print('test set MSE: {}'.format(test_MSE))

# Do the prediction and calculate the performance (MSE) for model 1
print('Model with the raw state vector')
x = np.arange(len(Xraw_test))
yraw_hat = regraw.predict(Xraw_test)
plt.plot(x, yraw_test)
plt.plot(x, yraw_hat)
plt.show()

test_MSE = mean_squared_error(yraw_test, yraw_hat)
print('test set MSE: {}'.format(test_MSE))

yraw_train_hat = regraw.predict(Xraw_train)
train_MSE = mean_squared_error(yraw_train, yraw_train_hat)
print('train set MSE: {}'.format(train_MSE))

comp = pd.read_csv(path + 'linear_model_comparison.csv')

comp

import seaborn as sns

comp_2 = pd.melt(comp, id_vars=['algorithm', 'sample_size'], value_name='mse')
# comp_2['datasample_sz'] = [lambda x:x.split('.')[0] for x in comp_2.variable]
comp_2['datasample_sz'] = [val.split(' ')[1] for val in comp_2['variable'].values]
comp_2.head()

sns.boxplot(x="sample_size", y="mse", data=comp_2)
plt.show()



