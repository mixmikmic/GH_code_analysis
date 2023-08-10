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
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, Input, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

InceptionV3(weights='imagenet')

in_v3 = InceptionV3(weights='imagenet')

in_v3.summary()

IMG_SIZE = 299

ok_images='/Users/carles/Downloads/data/ok'

nok_images='/Users/carles/Downloads/data/nok'

X = np.vstack((dl_functions.normalize_images_array(ok_images, IMG_SIZE), dl_functions.normalize_images_array(nok_images, IMG_SIZE)))

y = np.vstack((np.array([1]*(len(X)/2)).reshape((len(X)/2), 1), np.array([0]*(len(X)/2)).reshape((len(X)/2), 1)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

y_train_sparse = np_utils.to_categorical(y_train, 2)

y_test_sparse = np_utils.to_categorical(y_test, 2)

base_model = InceptionV3(weights='imagenet', include_top=False)
# add a global spatial average pooling layer
x = base_model.output
x = GlobalMaxPooling2D()(x)
# Add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# Add a logistic layer - we have 2 classes.
predictions = Dense(2, activation='softmax')(x)

# This is the model we will train.
model = Model(input=base_model.input, output=predictions)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train_sparse, batch_size=128, epochs=2, validation_split=0.1, verbose=1)

score = model.evaluate(X_test, y_test_sparse, verbose=True)

print('Test loss: {:0,.4f}'.format(score[0]))
print('Test accuracy: {:.2%}'.format(score[1]))

predicted_images = []
for i in model.predict(X_test):
    predicted_images.append(np.where(np.max(i) == i)[0])

dl_functions.show_confusion_matrix(confusion_matrix(y_test, predicted_images), ['Class 0', 'Class 1'])

