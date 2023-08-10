#!pip install keras
#!pip install numpy
#!pip install imageio
#!pip install matplotlib
#!pip install opencv-python

from __future__ import print_function


from video_file import *

import importlib
try:
    importlib.reload(video_file)
except:
    pass

import cv2
import sys
import os
import csv
import numpy as np
from random import randint
from random import shuffle

from PIL import Image
import imageio
import itertools as it

import tensorflow as tf
import keras
print("Keras version %s" % keras.__version__)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K

print("Tensorflow version %s" % tf.__version__)

import pprint
pp = pprint.PrettyPrinter(depth=6)


# Create the image transformer
transformer = VideoTransform( zoom_range=0.1, rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, shear_range= 0.1, fill_mode='nearest', vertical_flip=False, horizontal_flip=False, horizontal_flip_invert_indices = [], horizontal_flip_reverse_indices = [0,1,2,3,4,5], data_format='channels_last' )

# Paths relative to current python file.
data_path  = ".\\..\\..\\TrainingData\\Processed\\AmateurDefender\\Result\\settings.tsv"

print("Opening training frames from config %s." % (data_path))
position_rel_indexes = [0, 3] # Predict current rod positions and future position in 2 frames
frame_rel_indexes = [0] # Use only current frame as input
training = TrainingInput(transformer, data_path, position_rel_indexes, frame_rel_indexes, 0.2)
training.clear_memory()

# Define our training and validation iterators

image_height       = training.height
image_width        = training.width
image_depth        = training.depth
image_channels     = training.channels
output_size        = 3


def TrainGen():
    while True:
        #print("TrainGen restarting training input.")
        training.move_first_training_frame()
        (frames, output) = training.get_next_training_frame()
        while frames is not None:
            yield (frames, output)
            (frames, output) = training.get_next_training_frame()
            
def ValidateGen():
    while True:
        #print("Validation restarting training input.")
        training.move_first_validation_frame()
        (frames, output) = training.get_next_validation_frame()
        while frames is not None:
            yield (frames, output)
            (frames, output) = training.get_next_validation_frame()

# Generators for training the position
def TrainBatchGen(batch_size):
    gen = TrainGen()
    while True:
        # Build the next batch
        batch_frames = np.zeros(shape=(batch_size, image_depth, image_height, image_width, image_channels), dtype=np.float32)
        batch_outputs = np.zeros(shape=(batch_size, 3), dtype=np.float32)
        for i in range(batch_size):
            (frames, output) = next(gen)
            batch_frames[i,:,:,:,:] = frames
            batch_outputs[i,:] = output[0:3] # Train just the 3 current rod positions as outputs
            #batch_outputs[i,:] = output[3:6] - output[0:3] # Train the difference in the three rod positions as output
            #batch_outputs[i,:] = output
            
        
        #pp.pprint("Yielding batch")
        #pp.pprint(batch_outputs)
        yield (batch_frames, batch_outputs)
        #pp.pprint("Yielded batch")

def ValidateBatchGen(batch_size):
    gen = ValidateGen()
    while True:
        # Build the next batch
        batch_frames = np.zeros(shape=(batch_size, image_depth, image_height, image_width, image_channels), dtype=np.float32)
        batch_outputs = np.zeros(shape=(batch_size, 3), dtype=np.float32)
        for i in range(batch_size):
            (frames, output) = next(gen)
            batch_frames[i,:,:,:,:] = frames
            batch_outputs[i,:] = output[0:3] # Train just the 3 current rod positions as outputs
            #batch_outputs[i,:] = output[3:6] - output[0:3] # Train the difference in the three rod positions as output
            #batch_outputs[i,:] = output
        
        #pp.pprint("Yielding batch")
        #pp.pprint(batch_outputs)
        yield (batch_frames, batch_outputs)
        #pp.pprint("Yielded batch")
            
# Generators for training the difference in position
def TrainBatchGenDpos(batch_size):
    gen = TrainGen()
    while True:
        # Build the next batch
        batch_frames = np.zeros(shape=(batch_size, image_depth, image_height, image_width, image_channels), dtype=np.float32)
        batch_outputs = np.zeros(shape=(batch_size, 3), dtype=np.float32)
        for i in range(batch_size):
            (frames, output) = next(gen)
            batch_frames[i,:,:,:,:] = frames
            
            # Decide if this dpos is valid by looking for a single frame position change > 0.1
            #max_dpos = 0.0
            #for i in range(len(batch_frames)/3):
            
            #batch_outputs[i,:] = output[0:3] # Train just the 3 current rod positions as outputs
            batch_outputs[i,0] = output[3] - output[0] # Train the difference in the three rod positions as output
            batch_outputs[i,1] = output[4] - output[1]
            batch_outputs[i,2] = output[5] - output[2]
            #batch_outputs[i,:] = output
            
        
        #pp.pprint("Yielding batch")
        #pp.pprint(batch_outputs)
        yield (batch_frames, batch_outputs)
        #pp.pprint("Yielded batch")

def ValidateBatchGenDpos(batch_size):
    gen = ValidateGen()
    while True:
        # Build the next batch
        batch_frames = np.zeros(shape=(batch_size, image_depth, image_height, image_width, image_channels), dtype=np.float32)
        batch_outputs = np.zeros(shape=(batch_size, 3), dtype=np.float32)
        for i in range(batch_size):
            (frames, output) = next(gen)
            batch_frames[i,:,:,:,:] = frames
            #batch_outputs[i,:] = output[0:3] # Train just the 3 current rod positions as outputs
            batch_outputs[i,0] = output[3] - output[0] # Train the difference in the three rod positions as output
            batch_outputs[i,1] = output[4] - output[1]
            batch_outputs[i,2] = output[5] - output[2]
            #batch_outputs[i,:] = output
        
        #pp.pprint("Yielding batch")
        #pp.pprint(batch_outputs)
        yield (batch_frames, batch_outputs)
        #pp.pprint("Yielded batch")


# Helper function to plot our validation result
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


def plot_validate(model, frames, output_true, name):
    #(frames, outputs_true) = next(ValidateBatchGen(2000))
    #frames = np.squeeze(frames, axis=(1,))
    #validate_in, validate_out
    #frames = validate_in
    #outputs_true =validate_out
    outputs_predicted = model.predict(frames, batch_size=32, verbose=1)
    print("Predicted.")
    
    
    pp.pprint(outputs_true)
    pp.pprint(outputs_predicted)
    
    
    plt.figure(figsize=(8,30))
    count = len(frames)
    
    plt.subplot(611)
    plt.plot(range(count),outputs_true[0:count,0], range(count),outputs_predicted[0:count,0] )
    plt.ylabel("Rod 1: %s" % name)
    plt.title("First 200 output recordings")
    plt.grid(True)
    
    plt.subplot(612)
    plt.plot(range(count),outputs_true[0:count,1], range(count),outputs_predicted[0:count,1] )
    plt.ylabel("Rod 2: %s" % name)
    plt.title("First output recordings")
    plt.grid(True)
    
    plt.subplot(613)
    plt.plot(range(count),outputs_true[0:count,2], range(count),outputs_predicted[0:count,2] )
    plt.ylabel("Rod 3: %s" % name)
    plt.title("First output recordings")
    plt.grid(True)
    
    
    
    plt.figure(figsize=(8,30))
    plt.subplot(611)
    
    true, predicted = zip(*sorted(zip(outputs_true[0:count,0], outputs_predicted[0:count,0])))
    plt.plot(range(count),true, range(count),predicted )
    plt.ylabel("Rod 1: %s" % name)
    plt.title("First 200 output recordings")
    plt.grid(True)
    
    plt.subplot(612)
    true, predicted = zip(*sorted(zip(outputs_true[0:count,1], outputs_predicted[0:count,1])))
    plt.plot(range(count),true, range(count),predicted, marker='.', markersize = 2, linewidth =0.1, markerfacecolor='black')
    plt.ylabel("Rod 2: %s" % name)
    plt.grid(True)
    
    plt.subplot(613)
    true, predicted = zip(*sorted(zip(outputs_true[0:count,2], outputs_predicted[0:count,2])))
    plt.plot(range(count),true, range(count),predicted, marker='.', markersize = 2, linewidth =0.1, markerfacecolor='black')
    plt.ylabel("Rod 3: %s" % name)
    plt.grid(True)
    
    
    plt.show()
          

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

training.move_first_training_frame()

for k in range(10):
    (frame, position) = training.get_next_training_frame()
    data = np.zeros(shape=(np.shape(frame)[1], np.shape(frame)[2] * np.shape(frame)[0], 3), dtype=np.float32)
    for i in range(np.shape(frame)[0]):
        tmp = frame[i,:,:,:]
        data[:,i*np.shape(frame)[2]:(i+1)*np.shape(frame)[2],:] = tmp


    plt.imshow(data)
    plt.show()
    pp.pprint(position)

training.move_first_training_frame()

print("Shape of training input:")
pp.pprint(np.shape(frame))

print("Shape of training output:")
pp.pprint(np.shape(position))

print("Corresponding Positions:")
pd.DataFrame(position)
pp.pprint(position)



from keras.models import Sequential
from keras.layers import *
from keras.models import Model


# Build the model
pp.pprint("Input shape without batches:")
pp.pprint((image_height, image_width, image_channels))


# Build a functional model design
inputs = Input(shape=(1, image_height, image_width, image_channels,))
x = Conv3D(124,
           kernel_size = (1, 5, 5),
           padding = "same")(inputs)
x = Activation('relu')(x)

x = Conv3D(124,
           kernel_size = (1, 5, 5),
           padding = "same")(x)
x = Activation('relu')(x)

# Split into a horizontal detail and vertical detailed CNN paths
x_height_detailed = MaxPooling3D( pool_size=(1, 2, 1))(x) # (?, 1, 54, 100, 128, 3 )

x_height_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_height_detailed)
x_height_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_height_detailed)
x_height_detailed = MaxPooling3D( pool_size=(1, 2, 1))(x_height_detailed)

x_height_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_height_detailed)
x_height_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_height_detailed)
x_height_detailed = MaxPooling3D( pool_size=(1, 2, 1))(x_height_detailed)


x_height_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_height_detailed)
x_height_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_height_detailed)
x_height_detailed = MaxPooling3D( pool_size=(1, 1, 2))(x_height_detailed)

x_height_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_height_detailed)
x_height_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_height_detailed)
x_height_detailed = MaxPooling3D( pool_size=(1, 1, 2))(x_height_detailed)

x_height_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_height_detailed)
x_height_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_height_detailed)
x_height_detailed = MaxPooling3D( pool_size=(1, 1, 2))(x_height_detailed)

x_height_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_height_detailed)
x_height_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_height_detailed)
x_height_detailed = MaxPooling3D( pool_size=(1, 1, 2))(x_height_detailed)


x_height_detailed = Flatten()(x_height_detailed)





x_width_detailed = MaxPooling3D( pool_size=(1, 1, 2))(x) # (?, 1, 54, 100, 128, 3 )

x_width_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_width_detailed)
x_width_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_width_detailed)
x_width_detailed = MaxPooling3D( pool_size=(1, 1, 2))(x_width_detailed)

x_width_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_width_detailed)
x_width_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_width_detailed)
x_width_detailed = MaxPooling3D( pool_size=(1, 1, 2))(x_width_detailed)

x_width_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_width_detailed)
x_width_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_width_detailed)
x_width_detailed = MaxPooling3D( pool_size=(1, 1, 2))(x_width_detailed)

x_width_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_width_detailed)
x_width_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_width_detailed)
x_width_detailed = MaxPooling3D( pool_size=(1, 2, 1))(x_width_detailed)

x_width_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_width_detailed)
x_width_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_width_detailed)
x_width_detailed = MaxPooling3D( pool_size=(1, 2, 1))(x_width_detailed)

x_width_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_width_detailed)
x_width_detailed = Conv3D(124,
           kernel_size = (1, 3, 3),
           padding = "same",
           activation = "relu")(x_width_detailed)
x_width_detailed = MaxPooling3D( pool_size=(1, 2, 1))(x_width_detailed)



x_width_detailed = Flatten()(x_width_detailed)


x = keras.layers.concatenate([x_height_detailed, x_width_detailed])

#x = Flatten()(x_height_detailed)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='linear')(x)

model = Model(inputs=inputs, outputs=predictions)


#epoch = 45
#WEIGHTS_FNAME = 'config5_iter%i.hdf'
#model.load_weights(WEIGHTS_FNAME % epoch)
#print("Loaded model.")

#model.optimizer.lr.assign(0.00000001)

# For a multi-class classification problem
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001),
              loss='mean_squared_error',
              metrics=['accuracy'])

model.summary()



def mse(y_true, y_pred):
    return K.square(y_pred - y_true)*0.001 # Hackjob so Keras iterations show exponential value of MSE to get precision.


model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001),
              loss='mean_squared_error',
              metrics=[mse])

print("Updated learner.")

# Train the model to predict the future position. This is the control signal to the robot AI
WEIGHTS_FNAME = 'pos_cnn_weights_%i.hdf'
MODELS_FNAME = 'pos_cnn_models_%i.h5'

batch_size = 30
batches_training_per_epoch = int(training.get_training_count() / batch_size)
batches_validation_per_epoch = int(training.get_validation_count() / batch_size)
print("%i training batches, %i validation batches" % (batches_training_per_epoch, batches_validation_per_epoch) )

for epoch in range(15):
    try:
        model.fit_generator(TrainBatchGen(batch_size), batches_training_per_epoch, epochs=epoch+1, verbose=1, callbacks=None, class_weight=None, max_q_size=10, workers=1, validation_data=ValidateBatchGen(batch_size), validation_steps = batches_validation_per_epoch, pickle_safe=False, initial_epoch=epoch)
        model.save_weights(WEIGHTS_FNAME % epoch)
        model.save(MODELS_FNAME % epoch)
        print(("Wrote model to " + WEIGHTS_FNAME )  % epoch)
    except KeyboardInterrupt:
        print("\r\nUser stopped the training.")
        break
    


# Load the best model result
epoch = 4
WEIGHTS_FNAME = 'pos_cnn_weights_%i.hdf'
model.load_weights(WEIGHTS_FNAME % epoch)
print("Loaded model.")


# Plot the real versus predicted values for some of the validation data
(frames, outputs_true) = next(ValidateBatchGen(2000))
plot_validate(model, frames, outputs_true, "Position prediction")

# Load the best position prediction model as the starting point
epoch = 10
WEIGHTS_FNAME = 'pos_cnn_weights_%i.hdf'
model.load_weights(WEIGHTS_FNAME % epoch)
print("Loaded model.")


def mse(y_true, y_pred):
    return K.square(y_pred - y_true)*0.001 # Hackjob so Keras iterations show exponential value of MSE to get precision.


model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001),
              loss='mean_squared_error',
              metrics=[mse])

print("Updated learner.")

# Train the model to predict the future position. This is the control signal to the robot AI
WEIGHTS_FNAME = 'dpos_cnn_weights_%i.hdf'
MODELS_FNAME = 'dpos_cnn_models_%i.h5'


for epoch in range(10000):
    try:
        model.fit_generator(TrainBatchGenDpos(20), 1552, epochs=epoch+1, verbose=1, callbacks=None, class_weight=None, max_q_size=10, workers=1, validation_data=ValidateBatchGenDpos(20), validation_steps = 500, pickle_safe=False, initial_epoch=epoch)
        model.save_weights(WEIGHTS_FNAME % epoch)
        model.save(MODELS_FNAME % epoch)
        print(("Wrote model to " + WEIGHTS_FNAME )  % epoch)
    except KeyboardInterrupt:
        print("\r\nUser stopped the training.")
        break
    

# Load the best position prediction model as the starting point
epoch = 17
WEIGHTS_FNAME = 'dpos_cnn_weights_%i.hdf'
model.load_weights(WEIGHTS_FNAME % epoch)
print("Loaded model.")


# Plot the real versus predicted values for some of the validation data
(frames, outputs_true) = next(ValidateBatchGenDpos(2000))
plot_validate(model, frames, outputs_true, "Difference in position")

def mse(y_true, y_pred):
    return K.square(y_pred - y_true)*0.001 # Hackjob so Keras iterations show exponential value of MSE to get precision.


model.compile(optimizer=keras.optimizers.RMSprop(lr=0.00001),
              loss='mean_squared_error',
              metrics=[mse])

print("Updated learner.")

# Train the model to predict the future position. This is the control signal to the robot AI
WEIGHTS_FNAME = 'dpos_cnn_weights_%i.hdf'
MODELS_FNAME = 'dpos_cnn_models_%i.h5'

start_epoch = epoch+1

for epoch in range(10000):
    epoch += start_epoch
    try:
        model.fit_generator(TrainBatchGenDpos(20), 1552, epochs=epoch+1, verbose=1, callbacks=None, class_weight=None, max_q_size=10, workers=1, validation_data=ValidateBatchGenDpos(20), validation_steps = 500, pickle_safe=False, initial_epoch=epoch)
        model.save_weights(WEIGHTS_FNAME % epoch)
        model.save(MODELS_FNAME % epoch)
        print(("Wrote model to " + WEIGHTS_FNAME )  % epoch)
    except KeyboardInterrupt:
        print("\r\nUser stopped the training.")
        break
    

# Load the best position prediction model as the starting point
epoch = 35
WEIGHTS_FNAME = 'dpos_cnn_weights_%i.hdf'
model.load_weights(WEIGHTS_FNAME % epoch)
print("Loaded model.")


# Plot the real versus predicted values for some of the validation data
(frames, outputs_true) = next(ValidateBatchGenDpos(2000))
plot_validate(model, frames, outputs_true, "Difference in position")

def mse(y_true, y_pred):
    return K.square(y_pred - y_true)*0.001 # Hackjob so Keras iterations show exponential value of MSE to get precision.


model.compile(optimizer=keras.optimizers.RMSprop(lr=0.000001),
              loss='mean_squared_error',
              metrics=[mse])

print("Updated learner.")

# Train the model to predict the future position. This is the control signal to the robot AI
WEIGHTS_FNAME = 'dpos_cnn_weights_%i.hdf'
MODELS_FNAME = 'dpos_cnn_models_%i.h5'

start_epoch = epoch+1

for epoch in range(10000):
    epoch += start_epoch
    try:
        model.fit_generator(TrainBatchGenDpos(20), 1552, epochs=epoch+1, verbose=1, callbacks=None, class_weight=None, max_q_size=10, workers=1, validation_data=ValidateBatchGenDpos(20), validation_steps = 500, pickle_safe=False, initial_epoch=epoch)
        model.save_weights(WEIGHTS_FNAME % epoch)
        model.save(MODELS_FNAME % epoch)
        print(("Wrote model to " + WEIGHTS_FNAME )  % epoch)
    except KeyboardInterrupt:
        print("\r\nUser stopped the training.")
        break

# Load the best position prediction model as the starting point
epoch = 61
WEIGHTS_FNAME = 'dpos_cnn_weights_%i.hdf'
model.load_weights(WEIGHTS_FNAME % epoch)
print("Loaded model.")


# Plot the real versus predicted values for some of the validation data
(frames, outputs_true) = next(ValidateBatchGenDpos(2000))
plot_validate(model, frames, outputs_true, "Difference in position")

def mse(y_true, y_pred):
    return K.square(y_pred - y_true)*0.001 # Hackjob so Keras iterations show exponential value of MSE to get precision.


model.compile(optimizer=keras.optimizers.RMSprop(lr=0.00001),
              loss='mean_squared_error',
              metrics=[mse])

print("Updated learner.")

# Train the model to predict the future position. This is the control signal to the robot AI
WEIGHTS_FNAME = 'dpos_cnn_weights_%i.hdf'
MODELS_FNAME = 'dpos_cnn_models_%i.h5'

start_epoch = epoch+1

for epoch in range(10000):
    epoch += start_epoch
    try:
        model.fit_generator(TrainBatchGenDpos(20), 1552, epochs=epoch+1, verbose=1, callbacks=None, class_weight=None, max_q_size=10, workers=1, validation_data=ValidateBatchGenDpos(20), validation_steps = 500, pickle_safe=False, initial_epoch=epoch)
        model.save_weights(WEIGHTS_FNAME % epoch)
        model.save(MODELS_FNAME % epoch)
        print(("Wrote model to " + WEIGHTS_FNAME )  % epoch)
    except KeyboardInterrupt:
        print("\r\nUser stopped the training.")
        break

# Load the best position prediction model as the starting point
epoch = 389
WEIGHTS_FNAME = 'dpos_cnn_weights_%i.hdf'
model.load_weights(WEIGHTS_FNAME % epoch)
print("Loaded model.")


# Plot the real versus predicted values for some of the validation data
(frames, outputs_true) = next(ValidateBatchGenDpos(2000))
plot_validate(model, frames, outputs_true, "Difference in position")

