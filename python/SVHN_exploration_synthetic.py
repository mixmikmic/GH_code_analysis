# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from PIL import Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
import cPickle as pickle

# Config the matlotlib backend as plotting inline in IPython
get_ipython().magic('matplotlib inline')
get_ipython().magic('autosave 300')
np.set_printoptions(threshold=np.inf)

train_pickle_file = './SVHN_basic_train_labels.pickle'

with open(train_pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_labels = save['train_image_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_labels.shape)
print(train_labels[0:2,:])

train_pickle_file = 'SVHN_basic_train_data_basic.pickle'

with open(train_pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  del save  # hint to help gc free up memory
  print('Training set',train_dataset.shape)
print(train_labels[0:2,:])

test_pickle_file = './SVHN_basic_test_labels.pickle'

with open(test_pickle_file, 'rb') as f:
  save = pickle.load(f)
  test_labels = save['test_image_labels']
  del save  # hint to help gc free up memory
  print('Test set', test_labels.shape)
print(test_labels[0:2,:])

test_pickle_file = 'SVHN_basic_test_data_basic.pickle'

with open(test_pickle_file, 'rb') as f:
  save = pickle.load(f)
  test_dataset = save['test_dataset']
  del save  # hint to help gc free up memory
  print('Test set',test_dataset.shape)
print(test_labels[0:2,:])


def plot_length_and_labels(label_data):
    numBins = 20
    numBins_len = 5
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    for i in range(6):
        ax = plt.subplot(3,3,i+1)
        if i == 0:
            ax.hist(label_data[:,i],numBins_len,color='blue',alpha=0.5)
        else:
            ax.hist(label_data[:,i],numBins,color='green',alpha=0.5)
        
    plt.show()    

print("\nTraining Labels\n")
plot_length_and_labels(train_labels)


print("\nTest Labels\n")
plot_length_and_labels(test_labels)

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')


def visualize_samples(data, labels, indices):
    for count in range(indices.shape[0]):
        print("\tLabel:",labels[indices[count]])
        ax = plt.subplot(2,2,count+1)
        ax.imshow(data[indices[count]], cmap='Greys_r')
    plt.show()

print("Samples from Train Data")
random_indices=np.random.randint(0,train_labels.shape[0],4)
visualize_samples(train_dataset, train_labels, random_indices)

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

print("Samples from test Data")
random_indices=np.random.randint(0,test_labels.shape[0],4)
visualize_samples(test_dataset, test_labels, random_indices)



