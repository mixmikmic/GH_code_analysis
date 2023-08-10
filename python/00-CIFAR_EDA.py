get_ipython().system('pip install keras')

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import tarfile

cifar_file = 'cifar-10-python.tar.gz'
tar = tarfile.open(cifar_file, "r:gz")
tar.extractall()
tar.close()

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

for n in range(1,6):
    cifar_dict = unpickle('cifar-10-batches-py/data_batch_'+str(n))
    #build a single structure containing the data matrix X, and the target column vector Y
    if n == 1:
        X_train = np.array(cifar_dict[b'data'])
        y_train = np.array(cifar_dict[b'labels'])
    else:
        X_train = np.concatenate([X_train, cifar_dict[b'data']], axis=0)
        y_train = np.concatenate([y_train, cifar_dict[b'labels']], axis=0)
X_train.shape, y_train.shape

cifar_dict = unpickle('cifar-10-batches-py/test_batch')
X_test = np.array(cifar_dict[b'data'])
y_test = np.array(cifar_dict[b'labels'])

X_test.shape, y_test.shape

class_names_dict = unpickle('cifar-10-batches-py/batches.meta')

readable = lambda nam: str(nam)[2:-1]

class_names_list = []
for name in class_names_dict[b'label_names']:
    class_names_list.append(str(name)[2:-1])
    
class_names_list

X_train.dtype, y_train.dtype, X_test.dtype, y_train.dtype

y_train[0:5]

def show_unique_images(images, labels, class_names=class_names_list):
    unique_labels = []
    unique_indices = []
    
    fig = plt.figure(figsize=(25,10))
    
    n = 0
    for i in range(len(labels)):
        if labels[i] not in unique_labels:
            image = images[i].reshape(3,32,32).transpose(1,2,0)
            plt.subplot(2,5,n+1)
            n += 1
            plt.imshow(image, interpolation="nearest")
            plt.title(class_names[labels[i]])
            unique_labels.append(labels[i])
            unique_indices.append(i)
    plt.show()
    
    return unique_indices

def show_images(images, labels, indices, class_names=class_names_list):
    fig = plt.figure(figsize=(25,10))
    n = 0
    for i in indices:
        image = images[i].reshape(3,32,32).transpose(1,2,0)
        plt.subplot(2,5,n+1)
        n += 1
        plt.imshow(image)
        plt.title(class_names[labels[i]])
    plt.show()
    
    return None

unique_indices = show_unique_images(X_train, y_train)

unique_indices

X_train.shape

c = [x for x in range(50000) if x not in unique_indices]
show_unique_images(X_train[c], y_train[c])

def drop_color_channel(images, labels, indices, drop_color):
    new_images = []
    
    for image in images[indices]:
        #Drop red
        if drop_color == 0:
            new_images.append(np.concatenate([np.zeros(1024), image[1024:]], axis=0))
        #Drop green
        if drop_color == 1:
            new_images.append(np.concatenate([image[:1024], np.zeros(1024), image[2048:]], axis=0))
        #Drop blue
        if drop_color == 2:
            new_images.append(np.concatenate([image[:2048], np.zeros(1024)], axis=0))

    show_images(new_images, labels[indices], list(range(len(new_images))))
    
    return None

y_train[unique_indices]

#Drop the red channel by replacing it with zeros
drop_color_channel(X_train, y_train, unique_indices, 0)

#drop the green channel by replacing it with zeros
drop_color_channel(X_train, y_train, unique_indices, 1)

#Drop the blue channel by replacing it with zeros
drop_color_channel(X_train, y_train, unique_indices, 2)

def display_color_hists(images, labels, indices, class_names=class_names_list):
    fig = plt.figure(figsize=(15,50))
    n = 0
    for i in indices:
        plt.subplot(10,3,n+1)
        plt.hist(images[i][:1024])
        plt.title("red: " + class_names[labels[i]])
        n += 1
        
        plt.subplot(10,3,n+1)
        plt.hist(images[i][1024:2048])
        plt.title("green: " + class_names[labels[i]])
        n += 1
        
        plt.subplot(10,3,n+1)
        plt.hist(images[i][2048:])
        plt.title("blue: " + class_names[labels[i]])
        n += 1
    plt.show()
    
    return None

display_color_hists(X_train, y_train, unique_indices)

np.unique(y_train, return_counts=True)

np.unique(y_test, return_counts=True)

