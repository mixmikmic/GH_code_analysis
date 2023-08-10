from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential    # Importing Sequential Model
from keras.layers.core import Dense,Dropout, Activation  #  Importing  Dense Layers,Dropouts and Activation functions
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils  
np.random.seed(1671) # for reproducibility -> Once you put the same seed you get same patterns of random numbers.

# network and training
NB_EPOCH = 20  # 20-> times the model is exposed to the training set.
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10 # number of outputs = number of digits
OPTIMIZER = RMSprop()
N_HIDDEN = 512 # Neurons
VALIDATION_SPLIT=0.2 # how much TRAIN is reserved for VALIDATION
DROPOUT = 0.3

# Data: shuffled and split between train and test sets

(X_train, y_train_label), (X_test, y_test_label) = mnist.load_data()

#X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784

RESHAPED = 784 #(28*28)
X_train = X_train.reshape(60000, RESHAPED)
X_test  =  X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize -> Involve only rescaling to arrive at value relative to some size variables.

X_train /= 255 # Pixel values are 0 to 255 -> So we are normalizing training data by dividing it by 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train_label = np_utils.to_categorical(y_train_label, NB_CLASSES) 
Y_test_label = np_utils.to_categorical(y_test_label, NB_CLASSES)


# np_utils.to_categorical Used to convert the array of labelled data to one Hot vector-> Binarization of category

# Final hidden layer  with 10 outputs
# final stage is softmax
model = Sequential() # Sequential Model.
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,))) # 1st Hidden Layer --> 512 neurons and input dimension ->784
model.add(Activation('relu')) # Activation function for 1st Hidden Layer
model.add(Dropout(DROPOUT))

model.add(Dense(256))  # 2nd Hidden Layer --> 256 neurons
model.add(Activation('relu')) # Activation function for 2nd Hidden Layer
model.add(Dropout(DROPOUT))


model.add(Dense(NB_CLASSES)) # Final layer with 10 neurons == > no of outputs
model.add(Activation('softmax')) # Final layer activation will be 'softmax'

model.summary()

# Compiling a model in keras
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

# Training a model in keras

# Once the model is compiled it can be trained with the fit() function

history = model.fit(X_train, Y_train_label,
batch_size=BATCH_SIZE, epochs=NB_EPOCH,
verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# Finally calucating the score.
score = model.evaluate(X_test, Y_test_label, verbose=VERBOSE)
print("Test score:", score[0])
print('Test accuracy:', score[1])

