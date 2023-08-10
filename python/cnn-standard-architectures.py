import warnings
warnings.filterwarnings('ignore')

get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')

import matplotlib.pylab as plt
import numpy as np

from distutils.version import StrictVersion

import sklearn
print(sklearn.__version__)

assert StrictVersion(sklearn.__version__ ) >= StrictVersion('0.18.1')

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
print(tf.__version__)

assert StrictVersion(tf.__version__) >= StrictVersion('1.1.0')

import keras
print(keras.__version__)

assert StrictVersion(keras.__version__) >= StrictVersion('2.0.0')

import pandas as pd
print(pd.__version__)

assert StrictVersion(pd.__version__) >= StrictVersion('0.20.0')

# for VGG, ResNet, and MobileNet
INPUT_SHAPE = (224, 224)

# for InceptionV3, InceptionResNetV2, Xception
# INPUT_SHAPE = (299, 299)

import os
import skimage.data
import skimage.transform
from keras.utils.np_utils import to_categorical
import numpy as np

def load_data(data_dir, type=".ppm"):
    num_categories = 6

    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(type)]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    images64 = [skimage.transform.resize(image, INPUT_SHAPE) for image in images]
    y = np.array(labels)
    y = to_categorical(y, num_categories)
    X = np.array(images64)
    return X, y

# Load datasets.
ROOT_PATH = "./"
original_dir = os.path.join(ROOT_PATH, "speed-limit-signs")
original_images, original_labels = load_data(original_dir, type=".ppm")

X, y = original_images, original_labels

# !curl -O https://raw.githubusercontent.com/DJCordhose/speed-limit-signs/master/data/augmented-signs.zip
# from zipfile import ZipFile
# zip = ZipFile('augmented-signs.zip')
# zip.extractall('.')

data_dir = os.path.join(ROOT_PATH, "augmented-signs")
augmented_images, augmented_labels = load_data(data_dir, type=".png")

# merge both data sets

all_images = np.vstack((X, augmented_images))
all_labels = np.vstack((y, augmented_labels))

# shuffle
# https://stackoverflow.com/a/4602224

p = numpy.random.permutation(len(all_labels))
shuffled_images = all_images[p]
shuffled_labels = all_labels[p]
X, y = shuffled_images, shuffled_labels

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train.shape, y_train.shape

from keras.applications.xception import Xception

model = Xception(classes=6, weights=None)

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# !rm -rf ./tf_log
# https://keras.io/callbacks/#tensorboard
tb_callback = keras.callbacks.TensorBoard(log_dir='./tf_log')
# To start tensorboard
# tensorboard --logdir=./tf_log
# open http://localhost:6006

# Depends on harware GPU architecture, model is really complex, batch needs to be small (this works well on K80)
BATCH_SIZE = 25

early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)

get_ipython().magic('time model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[tb_callback, early_stopping_callback], batch_size=BATCH_SIZE)')

train_loss, train_accuracy = model.evaluate(X_train, y_train, batch_size=BATCH_SIZE)
train_loss, train_accuracy

test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
test_loss, test_accuracy

original_loss, original_accuracy = model.evaluate(original_images, original_labels, batch_size=BATCH_SIZE)
original_loss, original_accuracy

model.save('xception-augmented.hdf5')

get_ipython().system('ls -lh xception-augmented.hdf5')

from keras.applications.resnet50 import ResNet50

model = ResNet50(classes=6, weights=None)

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)

get_ipython().system('rm -rf ./tf_log')
# https://keras.io/callbacks/#tensorboard
tb_callback = keras.callbacks.TensorBoard(log_dir='./tf_log')
# To start tensorboard
# tensorboard --logdir=./tf_log
# open http://localhost:6006

# Depends on harware GPU architecture, model is really complex, batch needs to be small (this works well on K80)
BATCH_SIZE = 50

# https://github.com/fchollet/keras/issues/6014
# batch normalization seems to mess with accuracy when test data set is small, accuracy here is different from below
get_ipython().magic('time model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=BATCH_SIZE, callbacks=[tb_callback, early_stopping_callback])')
# %time model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=BATCH_SIZE, callbacks=[tb_callback])

train_loss, train_accuracy = model.evaluate(X_train, y_train, batch_size=BATCH_SIZE)
train_loss, train_accuracy

test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
test_loss, test_accuracy

original_loss, original_accuracy = model.evaluate(original_images, original_labels, batch_size=BATCH_SIZE)
original_loss, original_accuracy

model.save('resnet-augmented.hdf5')

get_ipython().system('ls -lh resnet-augmented.hdf5')



