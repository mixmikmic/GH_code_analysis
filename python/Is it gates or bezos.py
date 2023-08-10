from keras import applications

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import (Dropout, Flatten, Dense, Conv2D, 
                          Activation, MaxPooling2D)

from sklearn.cross_validation import train_test_split

from keras_tqdm import TQDMNotebookCallback

import os, glob
from tqdm import tqdm
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import shutil

# dimensions of our images. label = bezos, gates 2 labels
img_width, img_height = 128, 128

train_data_dir = 'train_images/'
validation_data_dir = 'valid_images'

epochs = 50
batch_size = 16

# Find version of keras after 1.x MUCH diff from 2.x

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3)))
model.add(Activation('relu')) #tanh
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu')) #tanh
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(96))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1)) # binary
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# 10 types of ships: Dense(10) Activation 'softmax' loss categorical_crossentropy

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary', shuffle=True)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
# need not fit all yrou stuff in memory, like a bunch of numpy arrays or something

model.fit_generator(
    train_generator,
    steps_per_epoch= 3000 // 16, # give me more data
    epochs=100,
    verbose=0,
    callbacks=[TQDMNotebookCallback()],
    validation_data=validation_generator,
    validation_steps= 300 // 16)

# bezoes, bezoes, bezoes = 50% of the time. 
# overfitting: get more data, augment eisting data.....
# conv network: filters find attributes anywhere in the image, ignores "where" it found it -- anywhere

# use a pretrained net for features
# Generate some features "features" --> 

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

model = VGG16(weights='imagenet', include_top=False)

model.summary()



# Should be (x,x,x,512d) vectors -- better than our 80ish% scratch convnet
# First, did we really randomize our data properly, or did our validation set give us a boost

import time

start = time.time()

train_gen = train_datagen.flow_from_directory(
        'train_images/',
        target_size=(128, 128),
        batch_size=1,
        class_mode=None,  # only data, no labels -- we're not trying to predict anything here
        shuffle=False)  # keep data in same order as labels

valid_gen = train_datagen.flow_from_directory(
        'valid_images/',
        target_size=(128, 128),
        batch_size=1,
        class_mode=None,  # only data, no labels -- we're not trying to predict anything here
        shuffle=False)  # keep data in same order as labels

train_probs = model.predict_generator(train_gen, 1099, workers=3, verbose=1)
valid_probs = model.predict_generator(valid_gen, 100, workers=3, verbose=1)

end = time.time()

print(end - start)

# MLP: since the 4x4x512d vector arguably is not a sequence, like text, audio, time series data
# And it doesn't have higher dimensional features which exhibit spatial invarince...
# The problem almost falls into the domain of traditional ML algorithms
# If we want to use deep learning to convert these 512d vectors to a prediction, we can use a DNN
# aka a multilayer perceptron

# idea of a generator -- imagine you had to fine the count of the word "dogged" in 100 billion lines of text
# the file is almost taking up your entire hard driver. You can use a generator to avoid reading the entire
# thing into RAM, just process one line at a time, and only keep track of a object to count "dogged", etc...
print(train_probs.shape)
print(valid_probs.shape)

print(train_gen.classes.shape)
print(valid_gen.classes.shape)

train_probs[0].ravel().shape #PCA that down SVM

bn_model = Sequential()
bn_model.add(Flatten(input_shape=train_probs.shape[1:]))

bn_model.add(Dense(128, activation='relu'))
bn_model.add(Dropout(0.5))

bn_model.add(Dense(256, activation='relu'))
bn_model.add(Dropout(0.5))

bn_model.add(Dense(512, activation='relu'))
bn_model.add(Dropout(0.5))

bn_model.add(Dense(1, activation='sigmoid'))

bn_model.compile(optimizer='adam',
              loss='binary_crossentropy', metrics=['accuracy'])

bn_model.fit(train_probs, train_gen.classes,
          epochs=50,
          batch_size=16,
          validation_data=(valid_probs, valid_gen.classes), shuffle=True)

# Discussion -- since they are both white older males, 
# it's not clear that imagenet really picked up on any differences
# They are white/caucasion, they are over 50, affulent, wear nice clothing, tank top all the time

# what if we add more data? 500 600, 2000 85%, 80 85%
# fine tuning --> 
# python searcher.py "bill gates" --count 1000 --label gates_1k and the same for bezos

from collections import Counter

more_im = glob.glob("/Users/jeff/experiments/images/*/*.jpg")
more_im = shuffle(more_im)

Counter([x.split("/")[-2] for x in more_im]).most_common()

# For the additional images we downloaded
get_ipython().magic('mkdir more_train')
get_ipython().magic('mkdir more_valid')

get_ipython().magic('mkdir more_train/gates')
get_ipython().magic('mkdir more_train/bezos')

get_ipython().magic('mkdir more_valid/gates')
get_ipython().magic('mkdir more_valid/bezos')

bezos = [x for x in more_im if "bezos" in x.split("/")[-2]]
gates = [x for x in shuffle(more_im) if "gates" in x.split("/")[-2]][:712]

print(len(bezos))
print(len(gates))

for_labeling = bezos + gates
for_labeling = shuffle(for_labeling)
assert(len(for_labeling) == 1424) # Thrown if wrong

get_ipython().magic('pwd')

# Gonna move the first 80% into more_train, the last 20% into more valid -- this is not elegant
# but it should help you grok what's happening

import shutil
from tqdm import tqdm


for index, image in tqdm(enumerate(for_labeling)):
    
    if index < 1139:
        label = image.split("/")[-2]
        image_name = image.split("/")[-1]
        if "gates" in label:
            shutil.copy(image, 'more_train/gates/{}'.format(image_name))
        if "bezos" in label:
            shutil.copy(image, 'more_train/bezos/{}'.format(image_name))
            
    if index > 1139:
        label = image.split("/")[-2]
        image_name = image.split("/")[-1]
        if "gates" in label:
            shutil.copy(image, 'more_valid/gates/{}'.format(image_name))
        if "bezos" in label:
            shutil.copy(image, 'more_valid/bezos/{}'.format(image_name))   

len(os.listdir('more_valid/bezos'))

# we're getting more data: we're usig a larger validation set
# since gates and bezos are mentioned together SO OFTEN....I bet we actually pictures of the wrong 
# guy in each folder since they come up. 


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'more_train/',
    target_size=(128, 128),
    batch_size=16,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    'more_valid/',
    target_size=(128, 128),
    batch_size=16,
    class_mode='binary')

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3)))
model.add(Activation('relu')) #tanh
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu')) #tanh
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(96))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1)) # binary
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch= 1139 // 16, # give me more data
    epochs=10,
    validation_data=validation_generator,
    validation_steps= 284 // 16)



train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'train_images/',
    target_size=(128, 128),
    batch_size=16,
    class_mode='binary', shuffle=True)

validation_generator = test_datagen.flow_from_directory(
    'valid_images/',
    target_size=(128, 128),
    batch_size=16,
    class_mode='binary', shuffle=False)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3)))
model.add(Activation('relu')) #tanh
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu')) #tanh
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(96))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1)) # binary
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch= 2000 // 16, # give me more data
    epochs=50, # This will be 
    validation_data=validation_generator,
    validation_steps= 300 // 16)

# Then try without shuffling
# Try validation_generator.class_indices and validation_generator.classes. pprint it and see how its useful to you.

model.predict('train')

get_ipython().magic('reset')



