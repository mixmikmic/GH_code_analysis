import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join as opj
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
import os
from keras import optimizers
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, Cropping2D
from keras.layers import MaxPooling2D, ZeroPadding2D, BatchNormalization, Activation
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from keras.optimizers import Adam

plt.rcParams['figure.figsize'] = 10, 10
get_ipython().magic('matplotlib inline')

data_dir = '/home/ubuntu/data/iceberg'

import keras
print(keras.__version__)

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
        '/home/ubuntu/data/iceberg/pngs/trainWithAug',
        target_size=(224, 224),
        batch_size=30,
        shuffle=False,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        '/home/ubuntu/data/iceberg/pngs/valid',
        target_size=(224, 224),
        batch_size=20,
        shuffle=False,
        class_mode='categorical')

vgg_base = VGG16(weights='imagenet', include_top=False)

def create_precomputed_data(model, generator):
    filenames = generator.filenames
    conv_features = model.predict_generator(generator, (generator.n/generator.batch_size))
    labels_onehot = to_categorical(generator.classes)
    labels = generator.classes
    return (filenames, conv_features, labels_onehot, labels)

trn_filenames, trn_conv_features, trn_labels, trn_labels_1 = create_precomputed_data(vgg_base, train_generator)
val_filenames, val_conv_features, val_labels, val_labels_1 = create_precomputed_data(vgg_base, validation_generator)

assert len(trn_filenames) == 13200, "trn_filenames not as expected"
assert trn_conv_features.shape == (13200, 7, 7, 512), "trn_conv_features not as expected"
assert trn_labels.shape == (13200, 2), "trn_labels not as expected"

assert len(val_filenames) == 400, "val_filenames not as expected"
assert val_conv_features.shape == (400, 7, 7, 512), "val_conv_features not as expected"
assert val_labels.shape == (400, 2), "val_labels not as expected"

RESULTS_DIR = '/home/ubuntu/data/iceberg/results'

import bcolz
def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def save_precomputed_data(filenames, conv_feats, labels, features_base_name="VGG16_conv_feats_with_aug/trn_"):
    save_array(RESULTS_DIR+"/"+features_base_name+'filenames.dat', np.array(filenames))
    save_array(RESULTS_DIR+"/"+features_base_name+'conv_feats.dat', conv_feats)
    save_array(RESULTS_DIR+"/"+features_base_name+'labels.dat', np.array(labels))
    
save_precomputed_data(trn_filenames, trn_conv_features, trn_labels, "VGG16_conv_feats_with_aug/trn_")
save_precomputed_data(val_filenames, val_conv_features, val_labels, "VGG16_conv_feats_with_aug/val_")

import bcolz
def load_array(fname):
    return bcolz.open(fname)[:]

def load_precomputed_data(features_base_name="VGG16_conv_feats_with_aug/trn_"):
    filenames = load_array(RESULTS_DIR+"/"+features_base_name+'filenames.dat').tolist()
    conv_feats = load_array(RESULTS_DIR+"/"+features_base_name+'conv_feats.dat')
    labels = load_array(RESULTS_DIR+"/"+features_base_name+'labels.dat')
    return filenames, conv_feats, labels

trn_filenames, trn_conv_features, trn_labels = load_precomputed_data("VGG16_conv_feats_with_aug/trn_")
val_filenames, val_conv_features, val_labels = load_precomputed_data("VGG16_conv_feats_with_aug/val_")

classifier_input_shape = (7, 7, 512)
classifier_input = Input(shape=classifier_input_shape)

nf = 128
p = 0. # adding any dropout at all means it doesnt train at all

x = Conv2D(nf,(3,3), activation='relu', padding='same')(classifier_input)
x = Dropout(p)(x)
x = Conv2D(2,(3,3), padding='same')(x)

x = GlobalAveragePooling2D()(x)
x = Activation('softmax')(x)

classifier_model_v2 = Model(classifier_input, x)

classifier_model_v2.compile(Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

from keras import backend as K

K.set_value(classifier_model_v2.optimizer.lr, 0.001)
K.eval(classifier_model_v2.optimizer.lr)

classifier_model_v2.fit(trn_conv_features, trn_labels, 
                                          batch_size=64, 
                                          epochs=10,
                                          validation_data=(val_conv_features, val_labels),
                                          shuffle=True)

classifier_model_v2.save('/home/ubuntu/data/iceberg/results/weights/VGG16_plus_simplev2_1train_08val.h5')



