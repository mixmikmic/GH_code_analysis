import os
import sys
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(os.path.join(module_path, 'cnn-keras-update'))

import fs
import datetime

from transformation import get_normalization_transform
from MiniBatchGenerator import MiniBatchGenerator

from keras.models import load_model
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import keras.backend as K

import PIL
from PIL import Image

import utils
import models

get_ipython().magic('matplotlib inline')

def deprocess(data):
    return data.transpose((2,1,0)) * 255

from dataset import TinyImdbWikiGenderDataset as Dataset

DATASET_DIR = '../data/imdb-wiki-tiny-dataset'

# Initialize the datasets
ds_train = Dataset(DATASET_DIR, 'train')
ds_val = Dataset(DATASET_DIR, 'val')

print(ds_train.sample_class(0))
print(ds_train.sample_classname(0))
Image.fromarray(deprocess(ds_train.sample(0)).astype(np.uint8))

print(ds_val.sample_class(0))
print(ds_val.sample_classname(0))
Image.fromarray(deprocess(ds_val.sample(0)).astype(np.uint8))

# Initialize the preprocessing pipeline
tform = get_normalization_transform(
  means=ds_train.get_mean(per_channel=True),
  stds=ds_train.get_stddev(per_channel=True),
  augmentation=True
)

mean = ds_train.samples().mean(axis=(0,2,3))
stddev = ds_train.samples().std(axis=(0,2,3))
print(mean)
print(stddev)

def deprocess(data):
    data = data.copy()
    data[0] *= stddev[0]
    data[1] *= stddev[1]
    data[2] *= stddev[2]
    data[0] += mean[0]
    data[1] += mean[1]
    data[2] += mean[2]
    return data.transpose((2,1,0)) * 255

def flip(X):
    return X.transpose(2,1,0)

X = ds_train.samples()[0:2]
Image.fromarray(flip(X[0]).astype(np.uint8))

Image.fromarray(deprocess(X[0]).astype(np.uint8))

print(ds_train.sample_class(0))
print(ds_train.sample_classname(0))
Image.fromarray(deprocess(tform.apply(ds_train.samples()[0:2])[0]).astype(np.uint8))

print(ds_val.sample_class(0))
print(ds_val.sample_classname(0))
Image.fromarray(deprocess(tform.apply(ds_val.sample(0))).astype(np.uint8))

print(ds_val.sample_class(0))
print(ds_val.sample_classname(0))
Image.fromarray(deprocess(tform.apply(ds_val.sample(0))).astype(np.uint8))

import numpy.testing as npt

for x in range(48):
    for y in range(48):
        for c in range(3):
            assert (ds_train.sample(0)[c,y,x] - mean[c]) / stddev[c] == tform.apply(ds_train.sample(0))[c,y,x]

for x in range(48):
    for y in range(48):
        for c in range(3):
            npt.assert_almost_equal((tform.apply(ds_train.sample(0))[c,y,x] * stddev[c]) + mean[c], ds_train.sample(0)[c,y,x], decimal=5)

def show_batch(X_batch, y_batch, labels, figsize=(20,5)):
    l = X_batch.shape[0]
    fig = plt.figure(figsize=figsize)
    
    for i in range(0,l):
        ax = fig.add_subplot(1, l, i+1)
       
        X = X_batch[i]
        y = y_batch[i]
        d, h, w = X.shape
        
        im = Image.fromarray(deprocess(X).astype(np.uint8))
        im = im.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        
        # Plot the image
        ax.imshow(im)
        ax.set_xlim((0, w))
        ax.set_ylim((0, h))
        
        ax.set_title(labels[y] if y in range(len(labels)) else str(y))
        
        ax.grid(True)
        # Remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # Remove major ticks
        for tic in ax.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
        
    plt.show()

import theano

def show_activations(model, X, figsize=(20,1)):    
    
    for layer in model.layers:
        try:
            convout1_f = theano.function(model.inputs, [layer.output])
            act = convout1_f([X])[0] * 255
            out = act.shape
        except:
            out = None
        
        if out and len(out) > 2:
            _, f, h, w = out
            
            fig = plt.figure(figsize=figsize)
            plt.title(layer.name)
            plt.gray()
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            
            for i in range(f):
                ax = fig.add_subplot(1, f, i+1)

                im = Image.fromarray(act[0][i].transpose().astype(np.uint8))
                im = im.transpose(PIL.Image.FLIP_TOP_BOTTOM)

                # Plot the image
                ax.imshow(im)
                ax.set_xlim((0, w))
                ax.set_ylim((0, h))

                ax.grid(False)
                # Remove tick labels
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                # Remove major ticks
                for tic in ax.xaxis.get_major_ticks():
                    tic.tick1On = tic.tick2On = False
                    tic.label1On = tic.label2On = False
                for tic in ax.yaxis.get_major_ticks():
                    tic.tick1On = tic.tick2On = False
                    tic.label1On = tic.label2On = False

            plt.show()

i = 5
X1 = MiniBatchGenerator(ds_train, ds_train.size(), tform).batch(0)[0][i]
X2 = tform.apply(ds_train.sample(i))

for x in range(48):
    for y in range(48):
        for c in range(3):
            assert X1[c,y,x] == X2[c,y,x] 

mb_train = MiniBatchGenerator(ds_train, 10, tform)
mb_val = MiniBatchGenerator(ds_val, 10, tform)

for i in range(10):
    X_batch, y_batch, ids = mb_train.batch(i)
    show_batch(X_batch, y_batch, ds_train.label_names)

for i in range(10):
    X_batch, y_batch, ids = mb_val.batch(i)
    show_batch(X_batch, y_batch, ds_val.label_names)

print(y_batch)
np_utils.to_categorical(y_batch, mb_train.dataset.nclasses())

model = models.get_simple_cnn(input_shape=(3,48,48), n_classes=2)
opt = RMSprop(lr=0.005, rho=0.9, epsilon=1e-08, decay=0.999)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

i = 0
X_batch, y, ids = mb_train.batch(i)
Y_batch = np_utils.to_categorical(y, mb_train.dataset.nclasses())
model.train_on_batch(X_batch, Y_batch)

f = 1
for i in range(1,10):
    if hasattr(model.layers[i], "W"):
        w, b = model.layers[i].get_weights()
        print(w[f])

Image.fromarray(deprocess(X_batch[0]).astype(np.uint8))

show_activations(model, X_batch[0])

Image.fromarray(deprocess(X_batch[1]).astype(np.uint8))

show_activations(model, X_batch[1])

