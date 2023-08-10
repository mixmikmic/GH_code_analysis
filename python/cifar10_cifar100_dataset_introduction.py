from __future__ import print_function
import os
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import numpy as np

import chainer

basedir = './src/cnn/images'

CIFAR10_LABELS_LIST = [
    'airplane', 
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

train, test = chainer.datasets.get_cifar10()

print('len(train), type ', len(train), type(train))
print('len(test), type ', len(test), type(test))

print('train[0]', type(train[0]), len(train[0]))

x0, y0 = train[0]
print('train[0][0]', x0.shape, x0)
print('train[0][1]', y0.shape, y0, '->', CIFAR10_LABELS_LIST[y0])

def plot_cifar(filepath, data, row, col, scale=3., label_list=None):
    fig_width = data[0][0].shape[1] / 80 * row * scale
    fig_height = data[0][0].shape[2] / 80 * col * scale
    fig, axes = plt.subplots(row, 
                             col, 
                             figsize=(fig_height, fig_width))
    for i in range(row * col):
        # train[i][0] is i-th image data with size 32x32
        image, label_index = data[i]
        image = image.transpose(1, 2, 0)
        r, c = divmod(i, col)
        axes[r][c].imshow(image)  # cmap='gray' is for black and white picture.
        if label_list is None:
            axes[r][c].set_title('label {}'.format(label_index))
        else:
            axes[r][c].set_title('{}: {}'.format(label_index, label_list[label_index]))
        axes[r][c].axis('off')  # do not show axis value
    plt.tight_layout()   # automatic padding between subplots
    plt.savefig(filepath)

plot_cifar(os.path.join(basedir, 'cifar10_plot.png'), train, 4, 5, 
           scale=4., label_list=CIFAR10_LABELS_LIST)
plot_cifar(os.path.join(basedir, 'cifar10_plot_more.png'), train, 10, 10, 
           scale=4., label_list=CIFAR10_LABELS_LIST)

CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

train_cifar100, test_cifar100 = chainer.datasets.get_cifar100()

print('len(train_cifar100), type ', len(train_cifar100), type(train_cifar100))
print('len(test_cifar100), type ', len(test_cifar100), type(test_cifar100))

print('train_cifar100[0]', type(train_cifar100[0]), len(train_cifar100[0]))

x0, y0 = train_cifar100[0]
print('train_cifar100[0][0]', x0.shape)  # , x0
print('train_cifar100[0][1]', y0.shape, y0)

plot_cifar(os.path.join(basedir, 'cifar100_plot_more.png'), train_cifar100,
           10, 10, scale=4., label_list=CIFAR100_LABELS_LIST)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

metadata = unpickle(os.path.join('./src/cnn/assets', 'meta'))
print(metadata)



