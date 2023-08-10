get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

import os, sys
import itertools, functools
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K

import tf_threads
tfconfig = tf_threads.limit(tf, 2)
session = tf.Session(config=tfconfig)
K.set_session(session)

import h5py
from skimage.io import imread, imshow
from skimage.transform import resize
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Reshape, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

DATASET = lambda fname = '': os.path.join('datsets/transfer_learning/DogsCats', fname)
assert os.path.exists(DATASET())

plt.figure()

print("Showing the train/cat.100.jpg, full path : {}".format(
            DATASET('train/cat.100.jpg')
        )
     )

imshow(imread(DATASET('train/cat.100.jpg')))

imshow(imread(DATASET("train/cat.100.jpg")))

help(imshow)

help(imread)

filename = DATASET("train/dog.200.jpg")
imshow(imread(filename))

images = Input(shape = (150, 150, 3))

vgg16 = VGG16(weights='imagenet', include_top=False)

# Lock the VGG16 Layers
for layer in vgg16.layers:
    layer.trainable = False 
    
classifier = [
    Flatten(input_shape = vgg16.output_shape[1:]),
    
    # Size, and Neuron Type
    Dense(256, activation="relu", name = 'dense_1'),
    Dropout(0.5),

    # Size, and Neuron Type
    Dense(1, activation="sigmoid", name = 'dense_2'),
]

y_pred = functools.reduce(lambda f1, f2: f2(f1), [images, vgg16]+classifier)

model = Model(inputs = [images], outputs = [y_pred])

model.compile(loss="binary_crossentropy",
    optimizer=SGD(lr=0.0001, momentum=0.9),
    metrics=["accuracy"])

model.summary()

BATCH_SIZE = 20

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale = 1. / 255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
    DATASET('TransferLearning/train'),
    target_size= (150, 150),
    batch_size = BATCH_SIZE,
    class_mode= "binary")

validation_generator = test_datagen.flow_from_directory(
    DATASET('TransferLearning/validation'),
    target_size=(150, 150),
    batch_size = BATCH_SIZE,
    class_mode = "binary")

try:
    model.fit_generator(
        train_generator, steps_per_epoch = 12500 // BATCH_SIZE,
        validation_data=validation_generator, validation_steps=800 // BATCH_SIZE,
        epochs=1) # Normally a few more epochs will be expected.
except KeyboardInterrupt:
    """Kernal -> Interrupt"""

model.save_weights('./weights_dogs_cats.h5')
os.path.exists('./weights_dogs_cats.h5')

class DogsVsCats(Model):
    def __init__(self): 
        self.images = Input(shape = (150, 150, 3))
        self.vgg16 = VGG16(weights = "imagenet", include_top = False)
        
        classifier = [
            Flatten(input_shape = self.vgg16.output_shape[1:]),
            Dense(256, activation = "relu", name = "dense_1"),
            Dropout(0.5),
            Dense(1, activation = "sigmoid", name = "dense_2")
        ]

        self.prediction = functools.reduce(lambda f1, f2: f2(f1), [self.images, self.vgg16]+classifier)
        
        super(DogsVsCats, self).__init__(
            inputs = [self.images],
            outputs = [self.prediction]
        )
        
        self.compile(loss = "binary_crossentropy", 
                    optimizer = SGD(lr = 0.0001, momentum = 0.9), 
                    metrics = ["accuracy"])
        
    def freeze_vgg16(trainable = False):
        for layer in self.vgg16.layers:
            layer.trainable = trainable

new_model = DogsVsCats()
new_model.load_weights("./weights_dogs_cats.h5")

loss, accuracy = new_model.evaluate_generator(validation_generator, steps = 800 // BATCH_SIZE)
print('loss:', loss, 'accuracy:', accuracy)

im_test = imread(DATASET('test1/100.jpg'))
imshow(im_test)
im_test = resize(im_test, (150, 150), mode = 'reflect')
y_pred = new_model.predict(np.expand_dims(im_test, 0)).squeeze()
print(['Cat', 'Dog'][y_pred>=0.5])

assert os.path.exists('/dsa/data/all_datasets/transfer_learning/DogsCats/weights_dogs_cats.h5')

pretrained_model = DogsVsCats()
pretrained_model.load_weights("/dsa/data/all_datasets/transfer_learning/DogsCats/weights_dogs_cats.h5")

loss, accuracy = pretrained_model.evaluate_generator(validation_generator, steps = 800)
print("loss:", loss, "accuracy:", accuracy)

pepper_pic = "fan1.JPG"

im_test = imread(pepper_pic)
im_test = resize(im_test, (150, 150), mode = "reflect")
imshow(im_test)
y_pred = pretrained_model.predict(np.expand_dims(im_test, 0)).squeeze()
print(["CAT", "DOG"][y_pred >= 0.5])

weight_source = '/dsa/data/all_datasets/transfer_learning/DogsCats/weights_dogs_cats.h5'
pic_source = "PEPPER6.jpg"
model = DogsVsCats()
model.load_weights(weight_source)
image_test = imread(pic_source)
image_test = resize(image_test, (150, 150), mode = "reflect")
imshow(image_test)
y_pred = model.predict(np.expand_dims(image_test, 0)).squeeze()
print(["IT'S A CAT", "IT'S A DOGGO"][y_pred >= 0.50])

