import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

CIFAR_DIR = 'CIFAR-10-batches-py/'

def unpickle(file):
    with open(file,'rb') as fo: #rb stands for read binary
        cifar_dict = pickle.load(fo,encoding='bytes')
    return cifar_dict

dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4', 'data_batch_5','test_batch']

all_data = [i for i in range(0,7)]

for i,direc in zip(all_data,dirs):
    all_data[i] = unpickle(CIFAR_DIR+direc)

batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]

data_batch1.keys()

X = data_batch1[b'data']
X.shape

X = X.reshape(10000,3,32,32).transpose(0,2,3,1).astype('uint8')

X[101].shape

plt.imshow(X[101])

X = data_batch1[b'data']

all_images = X.reshape(10000,3,32,32)

sample = all_images[0]

sample.shape

plt.imshow(sample.transpose(1,2,0))

