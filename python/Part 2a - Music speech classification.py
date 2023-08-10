import os

DATA_PATH = '/data/music/GTZAN_music_speech'
AUDIO_PATH = os.path.join(DATA_PATH, 'wav')

# General Imports

import argparse
import csv
import datetime
import glob
import math
import sys
import time
import numpy as np
import pandas as pd # Pandas for reading CSV files and easier Data handling in preparation

# Deep Learning

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, merge
from keras.layers.normalization import BatchNormalization

from sklearn import preprocessing
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# local imports for audio reading and processing
import audio_spectrogram as rp
from audiofile_read import audiofile_read

csv_file = os.path.join(DATA_PATH,'filelist_wav_relpath_wclasses.txt')
metadata = pd.read_csv(csv_file, index_col=0, sep='\t', header=None)
metadata.head(10)

# create two lists: one with filenames and one with associated classes
filelist = metadata.index.tolist()
classes = metadata[1].values.tolist()

classes[0:5]

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
labelencoder.fit(classes)
print(len(labelencoder.classes_), "classes:", ", ".join(list(labelencoder.classes_)))

classes_num = labelencoder.transform(classes)

# check first few classes (speech)
classes_num[0:5]

# check first few classes (music)
classes_num[-5:]

list_spectrograms = [] # spectrograms are put into a list first

# desired output parameters
n_mel_bands = 40   # y axis
frames = 80        # x axis

# some FFT parameters
fft_window_size=512
fft_overlap = 0.5
hop_size = int(fft_window_size*(1-fft_overlap))
segment_size = fft_window_size + (frames-1) * hop_size # segment size for desired # frames

for filename in filelist:
    print(".", end='')
    filepath = os.path.join(AUDIO_PATH, filename)
    samplerate, samplewidth, wavedata = audiofile_read(filepath,verbose=False)
    sample_length = wavedata.shape[0]

    # make Mono (in case of multiple channels / stereo)
    if wavedata.ndim > 1:
        wavedata = np.mean(wavedata, 1)
        
    # take only a segment; choose start position:
    #pos = 0 # beginning
    pos = int(wavedata.shape[0]/2 - segment_size/2) # center minus half segment size
    wav_segment = wavedata[pos:pos+segment_size]

    # 1) FFT spectrogram 
    spectrogram = rp.calc_spectrogram(wav_segment,fft_window_size,fft_overlap)

    # 2) Transform to perceptual Mel scale (uses librosa.filters.mel)
    spectrogram = rp.transform2mel(spectrogram,samplerate,fft_window_size,n_mel_bands)
        
    # 3) Log 10 transform
    spectrogram = np.log10(spectrogram)
    
    list_spectrograms.append(spectrogram)
        
print("\nRead", len(filelist), "audio files")

wav_segment

len(list_spectrograms)

spectrogram.shape

print("An audio segment is", round(float(segment_size) / samplerate, 2), "seconds long")

# you can skip this if you do not have matplotlib installed

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# show 1 sec wave segment
plt.plot(wav_segment)
filename

# show spectrogram
fig = plt.imshow(spectrogram, origin='lower')
fig.set_cmap('jet')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

# convert the input data to the right data type used by Keras Deep Learning (GPU)
dtype = keras.backend.floatx()
dtype

# a list of many 40x80 spectrograms is made into 1 big array
data = np.array(list_spectrograms, dtype=dtype)
data.shape

# vectorize
N, ydim, xdim = data.shape
data = data.reshape(N, xdim*ydim)
data.shape

# standardize
scaler = preprocessing.StandardScaler()
data = scaler.fit_transform(data)

# show mean and standard deviation: two vectors with same length as data.shape[1]
scaler.mean_, scaler.scale_

testset_size = 0.25 # % portion of whole data set to keep for testing, i.e. 75% is used for training

# Normal (random) split of data set into 2 parts
# from sklearn.model_selection import train_test_split

train_set, test_set, train_classes, test_classes = train_test_split(data, classes_num, test_size=testset_size, random_state=0)

train_classes

test_classes

# The two classes may be unbalanced
print("Class Counts: Class 0:", sum(train_classes==0), "Class 1:", sum(train_classes))

# better: Stratified Split retains the class balance in both sets
# from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=1, test_size=testset_size, random_state=0)
splits = splitter.split(data, classes_num)

for train_index, test_index in splits:
    print("TRAIN INDEX:", train_index)
    print("TEST INDEX:", test_index)
    train_set = data[train_index]
    test_set = data[test_index]
    train_classes = classes_num[train_index]
    test_classes = classes_num[test_index]
# Note: this for loop is only executed once, if n_iter==1 resp. n_splits==1

print(train_set.shape)
print(test_set.shape)
# Note: we will reshape the data later back to matrix form 

print("Class Counts: Class 0:", sum(train_classes==0), "Class 1:", sum(train_classes))

keras.backend.image_data_format()

n_channels = 1 # for grey-scale, 3 for RGB, but usually already present in the data

if keras.backend.image_data_format() == 'channels_last':  # TENSORFLOW
    # Tensorflow ordering (~/.keras/keras.json: "image_dim_ordering": "tf")
    train_set = train_set.reshape(train_set.shape[0], ydim, xdim, n_channels)
    test_set = test_set.reshape(test_set.shape[0], ydim, xdim, n_channels)
else: # THEANO
    # Theano ordering (~/.keras/keras.json: "image_dim_ordering": "th")
    train_set = train_set.reshape(train_set.shape[0], n_channels, ydim, xdim)
    test_set = test_set.reshape(test_set.shape[0], n_channels, ydim, xdim)

train_set.shape

test_set.shape

# we store the new shape of the images in the 'input_shape' variable.
# take all dimensions except the 0th one (which is the number of images)
input_shape = train_set.shape[1:]  
input_shape

#np.random.seed(0) # make results repeatable

model = Sequential()

#conv_filters = 16   # number of convolution filters (= CNN depth)
conv_filters = 32   # number of convolution filters (= CNN depth)

# Layer 1
model.add(Convolution2D(conv_filters, (3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25)) 

# Layer 2
model.add(Convolution2D(conv_filters, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2))) 

# After Convolution, we have a 16*x*y matrix output
# In order to feed this to a Full(Dense) layer, we need to flatten all data
# Note: Keras does automatic shape inference, i.e. it knows how many (flat) input units the next layer will need,
# so no parameter is needed for the Flatten() layer.
model.add(Flatten()) 

# Full layer
model.add(Dense(256, activation='sigmoid')) 

# Output layer
# For binary/2-class problems use ONE sigmoid unit, 
# for multi-class/multi-label problems use n output units and activation='softmax!'
model.add(Dense(1,activation='sigmoid'))

model.summary()

# Define a loss function 
loss = 'binary_crossentropy'  # 'categorical_crossentropy' for multi-class problems

# Optimizer = Stochastic Gradient Descent
optimizer = 'sgd' 

# Compiling the model
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# TRAINING the model
epochs = 15
history = model.fit(train_set, train_classes, batch_size=32, epochs=epochs)

# always execute this, and then a box of accuracy_score below to print the result
test_pred = model.predict_classes(test_set)

# 1 layer
accuracy_score(test_classes, test_pred)

# 2 layer
accuracy_score(test_classes, test_pred)

# 2 layer + 32 convolution filters
accuracy_score(test_classes, test_pred)

# 2 layer + 32 convolution filters + Dropout
accuracy_score(test_classes, test_pred)

model = Sequential()

conv_filters = 16   # number of convolution filters (= CNN depth)
filter_size = (3,3)
pool_size = (2,2)

# Layer 1
model.add(Convolution2D(conv_filters, filter_size, padding='valid', input_shape=input_shape))
#model.add(BatchNormalization())
#model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=pool_size)) 
#model.add(Dropout(0.3))

# Layer 2
model.add(Convolution2D(conv_filters, filter_size, padding='valid', input_shape=input_shape))
#model.add(BatchNormalization())
#model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=pool_size)) 
#model.add(Dropout(0.1))

# In order to feed this to a Full(Dense) layer, we need to flatten all data
model.add(Flatten()) 

# Full layer
model.add(Dense(256))  
#model.add(Activation('relu'))
#model.add(Dropout(0.1))

# Output layer
# For binary/2-class problems use ONE sigmoid unit, 
# for multi-class/multi-label problems use n output units and activation='softmax!'
model.add(Dense(1,activation='sigmoid'))

# MIREX 2015 model
model = Sequential()

conv_filters = 15   # number of convolution filters (= CNN depth)

# Layer 1
model.add(Convolution2D(conv_filters, (12, 8), padding='valid', input_shape=input_shape))
#model.add(BatchNormalization())
#model.add(Activation('relu')) 
model.add(Activation('sigmoid')) 
model.add(MaxPooling2D(pool_size=(2, 1))) 
#model.add(Dropout(0.3))


# In order to feed this to a Full(Dense) layer, we need to flatten all data
model.add(Flatten()) 

# Full layer
model.add(Dense(200, activation='sigmoid'))  
#model.add(Activation('relu'))
model.add(Dropout(0.5))

# Output layer
# For binary/2-class problems use ONE sigmoid unit, 
# for multi-class/multi-label problems use n output units and activation='softmax!'
model.add(Dense(1,activation='sigmoid'))

# Compiling and training the model
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

epochs = 15
history = model.fit(train_set, train_classes, batch_size=32, epochs=epochs)

# Verifying Accuracy on Test Set

test_pred = model.predict_classes(test_set)
accuracy_score(test_classes, test_pred)

from keras.layers.merge import concatenate

# Input only specifies the input shape
input = Input(input_shape)

# CNN layers
# specify desired number of filters
n_filters = 16 
# The functional API allows to specify the predecessor in (brackets) after the new Layer function call
conv_layer1 = Convolution2D(16, (10, 2))(input)  # a vertical filter
conv_layer2 = Convolution2D(25, (2, 10))(input)  # a horizontal filter

# possibly add Activation('relu') here

# Pooling layers
maxpool1 = MaxPooling2D(pool_size=(1,2))(conv_layer1) # horizontal pooling
maxpool2 = MaxPooling2D(pool_size=(2,1))(conv_layer2) # vertical pooling

# we have to flatten the Pooling output in order to be concatenated
flat1 = Flatten()(maxpool1)
flat2 = Flatten()(maxpool2)

# Merge the 2
merged = concatenate([flat1, flat2])

full = Dense(256, activation='relu')(merged)
output_layer = Dense(1, activation='sigmoid')(full)

# finally create the model
model = Model(inputs=input, outputs=output_layer)

model.summary()

# Define a loss function 
loss = 'binary_crossentropy'  # 'categorical_crossentropy' for multi-class problems

# Optimizer = Stochastic Gradient Descent
optimizer = 'sgd' 

# Compiling the model
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# TRAINING the model
epochs = 15
history = model.fit(train_set, train_classes, batch_size=32, epochs=epochs)

test_pred = model.predict(test_set)
test_pred[0:35,0]

test_pred = np.round(test_pred)
accuracy_score(test_classes, test_pred)

