# Imports
import numpy as np
import os
from utilities import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('ls', '/home/arasdar/datasets/har/har-data/train/Inertial_Signals/')

# HAR classification 
# Author: Burak Himmetoglu
# 8/15/2017

import pandas as pd 
import numpy as np
import os

# # test and train read
# Xtrain, Ytrain, list_ch_train = read_data(data_path="../../../datasets/har/har-data/", split="train")
# Xtest, Ytest, list_ch_test = read_data(data_path="../../../datasets/har/har-data/", split="test")
data_path='/home/arasdar/datasets/har/har-data/' 
split='train'
print(data_path, split)
 
# def read_data(data_path, split = "train"):
# 	""" Read data """
# Fixed params
n_class = 6
n_steps = 128

# Paths
path_ = os.path.join(data_path, split)
path_signals = os.path.join(path_, 'Inertial_Signals')
print(path_, path_signals)

# Read labels and one-hot encode
label_path = os.path.join(path_, 'y_' + split + '.txt')
print(label_path)
labels = pd.read_csv(label_path, header = None)
print(labels.shape, labels.dtypes, labels.head())

# Read time-series data
channel_files = os.listdir(path_signals)
channel_files.sort()
n_channels = len(channel_files)
posix = len(split) + 5
print(channel_files, n_channels, posix)

# Initiate array
list_of_channels = []
X = np.zeros((len(labels), n_steps, n_channels))
i_ch = 0
for fil_ch in channel_files:
    channel_name = fil_ch[:-posix]
    dat_ = pd.read_csv(os.path.join(path_signals,fil_ch), delim_whitespace = True, header = None)
    X[:,:,i_ch] = dat_.as_matrix()
  
    # Record names
    list_of_channels.append(channel_name)

    # iterate
    i_ch += 1
    print(channel_name, X.shape, len(list_of_channels), i_ch)

# # Return 
# return X, labels[0].values, list_of_channels
print(X.shape, labels[0].values, list_of_channels)

# Fixing the bug in the HAR dataset
import os
# Fixed params
n_class = 6
n_steps = 128

# Paths
# path_ = os.path.join(data_path='./data/', split='train')
path_ = os.path.join('../../data/har-data/', 'train')
print(path_)
path_signals = os.path.join(path_, "Inertial_Signals")
print(path_signals)

# split = 'train'

# # Read labels and one-hot encode
# label_path = os.path.join(path_, "y_" + split + ".txt")
# label_path
# labels = pd.read_csv(label_path, header = None)
# labels.shape

# # Read time-series data
# # path_signals, os.listdir(path='./data/train/Inertial Signals')
# channel_files = os.listdir(path=path_signals)
# # channel_files.sort()
# # n_channels = len(channel_files)
# # posix = len(split) + 5

X_train, labels_train, list_ch_train = read_data(data_path="../../../arasdar/datasets/har-data/", split="train") # train
X_test, labels_test, list_ch_test = read_data(data_path="../../../arasdar/datasets/har-data/", split="test") # test

assert list_ch_train == list_ch_test, "Mistmatch in channels!"

print ("Training data shape: N = {:d}, steps = {:d}, channels = {:d}".format(X_train.shape[0],
                                                                             X_train.shape[1],
                                                                             X_train.shape[2]))
print ("Test data shape: N = {:d}, steps = {:d}, channels = {:d}".format(X_test.shape[0],
                                                                         X_test.shape[1],
                                                                         X_test.shape[2]))

X_train.shape, X_train.dtype, labels_train.shape, labels_train.dtype

labels_train.max(), labels_test.max()

(np.mean(labels_train==0), np.mean(labels_train==1), np.mean(labels_train==2),
np.mean(labels_train==3), np.mean(labels_train==4), np.mean(labels_train==5), np.mean(labels_train==6), 
np.mean(labels_train==7))

(np.mean(labels_test==0), np.mean(labels_test==1), np.mean(labels_test==2),
np.mean(labels_test==3), np.mean(labels_test==4), np.mean(labels_test==5), np.mean(labels_test==6), 
np.mean(labels_test==7))

# Mean value for each channel at each step
all_data = np.concatenate((X_train,X_test), axis = 0)
means_ = np.zeros((all_data.shape[1],all_data.shape[2]))
stds_ = np.zeros((all_data.shape[1],all_data.shape[2]))

for ch in range(X_train.shape[2]):
    means_[:,ch] = np.mean(all_data[:,:,ch], axis=0)
    stds_[:,ch] = np.std(all_data[:,:,ch], axis=0)
    
df_mean = pd.DataFrame(data = means_)
df_std = pd.DataFrame(data = stds_)

all_data.shape, X_train.shape, X_test.shape, means_.shape, stds_.shape

df_std.hist()
plt.show()

df_mean.hist()
plt.show()

X_train, X_test = standardize(X_train, X_test)

# Check Mean value for each channel at each step
all_data = np.concatenate((X_train,X_test), axis = 0)
means_ = np.zeros((all_data.shape[1],all_data.shape[2]))
stds_ = np.zeros((all_data.shape[1],all_data.shape[2]))

for ch in range(X_train.shape[2]):
    means_[:,ch] = np.mean(all_data[:,:,ch], axis=0)
    stds_[:,ch] = np.std(all_data[:,:,ch], axis=0)
    
df_mean = pd.DataFrame(data = means_)
df_std = pd.DataFrame(data = stds_)

df_mean.hist()
plt.show()

df_mean.hist()
plt.show()

df_std.shape, df_std.dtypes

X_data = np.concatenate((X_train, X_test))

X_data.shape, X_data.dtype, X_train.shape, X_test.shape, X_train.dtype, X_test.dtype

X_data_mean = X_data.mean(axis=0, dtype=float)

X_data_mean.shape, X_data_mean.dtype

X_data_mean_DataFrame = pd.DataFrame(data=X_data_mean)

X_data_mean_DataFrame.hist()
plt.show()







