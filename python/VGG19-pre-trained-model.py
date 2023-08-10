reset -fs

import numpy as np
import pandas as pd
import os
import glob
import pickle
import gzip
import h5py
import dl_functions
from IPython.display import display
from keras.models import Sequential, model_from_json, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, Input
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.applications.vgg19 import VGG19, preprocess_input
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

VGG19(weights='imagenet')

vgg19 = VGG19(weights='imagenet')

vgg19.summary()

IMG_SIZE = 224

ok_images='/Users/carles/Desktop/data/ok'

nok_images='/Users/carles/Desktop/data/nok'

X = np.vstack((dl_functions.normalize_images_array(ok_images, IMG_SIZE), dl_functions.normalize_images_array(nok_images, IMG_SIZE)))

X_ok = X[:10000]

X_nok = X[10000:]

ARRAY_SAMPLE = 200

X_ok_sample = X_ok[np.random.randint(0, X_ok.shape[0], ARRAY_SAMPLE)]

X_nok_sample = X_nok[np.random.randint(0, X_nok.shape[0], ARRAY_SAMPLE)]

X = np.vstack((X_ok_sample, X_nok_sample))

y = np.vstack((np.array([1]*(len(X)/2)).reshape((len(X)/2), 1), np.array([0]*(len(X)/2)).reshape((len(X)/2), 1)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

y_train_sparse = np_utils.to_categorical(y_train, 2)

y_test_sparse = np_utils.to_categorical(y_test, 2)

# https://github.com/fchollet/keras/issues/4465
def pretrained_model(img_shape, num_classes, layer_type):
    # Using VGG19 as base model.
    base_model = VGG19(weights='imagenet')
    # We extract features up to the indicated layer.
    model = Model(input=base_model.input, output=base_model.get_layer('block5_pool').output)
    
    # This is the input format of our images.
    keras_input = Input(shape=img_shape, name='image_input')
    
    # Use the generated model from VGG19.
    output_model = model(keras_input)
    
    # Add fully-connected layers.
    x = Flatten(name='flatten')(output_model)
    x = Dense(4096, activation=layer_type, name='fc1')(x)
    x = Dense(4096, activation=layer_type, name='fc2')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    # Creating an instance of our model that uses VGG19 extracted features.
    pretrained_model = Model(inputs=keras_input, outputs=x)
    pretrained_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    return pretrained_model

model = pretrained_model(X_train.shape[1:], 2, 'relu')

model.fit(X_train, y_train_sparse, batch_size=128, epochs=5, validation_split=0.1, verbose=1)

score = model.evaluate(X_test, y_test_sparse, verbose=True)

print('Test loss: {:0,.4f}'.format(score[0]))
print('Test accuracy: {:.2%}'.format(score[1]))

predicted_images = []
for i in model.predict(X_test):
    predicted_images.append(np.where(np.max(i) == i)[0])

dl_functions.show_confusion_matrix(confusion_matrix(y_test, predicted_images), ['Class 0', 'Class 1'])

