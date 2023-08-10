# system level
import sys

# arrays
import numpy as np

# hdf5
import h5py

# keras
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Activation, Dropout, merge, Embedding, LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

# sklearn (for machine learning)
from sklearn import metrics

# plotting
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# theano
import theano
print "CONFIG:", theano.config.device

# ------------------------------------------------------------------------------
# Input variables
# ------------------------------------------------------------------------------

# training variables
nb_train = 200
nb_valid = 200
nb_test = 200
nb_epoch = 20
data_dir = "001"
filename_data = "testdata.h5"

# data locations
dir_data = "/Users/nord/Dropbox/TheCNNon/data" + data_dir + "/"
file_data = dir_data + filename_data # x data (images)
f_model = dir_data + "model.json" # model data (architecture)
f_weights = dir_data + "weights.h5" # model data (weights that we fit for)

data.close()
nb_pix = 10000
nb_data = 1000
x = np.random.rand(nb_data, nb_pix)
y = np.random.rand(nb_data)
data = h5py.File(file_data,'w')
data.create_dataset('x', data=x)
data.create_dataset('y', data=y)
data.close()

# ------------------------------------------------------------------------------
# Read in Data
# ------------------------------------------------------------------------------

# load data
data = h5py.File(file_data,'r')
x_data = data['X'][:]
y_data = data['y'][:]
nb_pix = len(x_data[0])


# check data sizes
statement =  "#TrainingSamples + #ValidSamples #TestSamples > TotalSamples, exiting!!!"
nb_total = nb_train + nb_test + nb_valid
assert nb_total <= len(x_data), statement

# indices for where to slice the arrays
ind_valid_start = ind_train_end = nb_train
ind_valid_end = ind_test_start = nb_train + nb_valid
ind_test_end = nb_train + nb_valid + nb_test

# slice the image arrays
x_train = x_data[:ind_train_end, :]
x_valid = x_data[ind_valid_start: ind_valid_end,  :]
x_test = x_data[ind_test_start: ind_test_end,  :]

# slice the label arrays
y_train = y_data[:ind_train_end]
y_valid = y_data[ind_valid_start: ind_valid_end]
y_test = y_data[ind_test_start: ind_test_end]

# cast data types
x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')


print "Data dimensions: "
print "Input data: ", np.shape(x_data), np.shape(y_data)
print "Training set: ", np.shape(x_train), np.shape(y_train)
print "Validation set: ", np.shape(x_valid), np.shape(y_valid)
print "Test Set: ", np.shape(x_test), np.shape(y_test)
print

# ------------------------------------------------------------------------------
# generate the model architecture
# ------------------------------------------------------------------------------
#https://keras.io/layers/recurrent/

# Embedding
max_features = 20
maxlen = nb_pix
embedding_size = 100
input_shape = x_train.shape

# Convolution
filter_length = 10
nb_filter = 32
pool_length = 10

# LSTM
lstm_output_size = 100


print 'Sequential Model'
model = Sequential()
print 'Embedding'
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
print 'Dropout'
model.add(Dropout(0.25))
print 'Convolution'
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
print 'MaxPooling'
model.add(MaxPooling1D(pool_length=pool_length))
print 'LSTM'
model.add(LSTM(lstm_output_size))
print 'Dense'
model.add(Dense(1))
print 'Activation'
model.add(Activation('sigmoid'))

# Compile Model
print 'compile model'
#model.compile(loss="mean_squared_error", optimizer="rmsprop")  
model.compile(loss="mse", optimizer="rmsprop")  

# ------------------------------------------------------------------------------
# Train model
# ------------------------------------------------------------------------------

# Train 
history = model.fit(
                    x_train, y_train,
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,
                    validation_data=(x_valid, y_valid),
                    #shuffle=shuffle,
                    verbose=True
                    )
# Save data
model.save_weights(f_weights, overwrite=True)
open(f_model, 'w').write(model.to_json())

# ------------------------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------------------------

# predict
pred = model.predict(x_valid)
true = y_valid
dy = (pred-true)/pred
print dy

# ------------------------------------------------------------------------------
# Analyze
# ------------------------------------------------------------------------------

# History
hist = history.history
loss = hist['loss']
val_loss = hist["val_loss"]
epochs = np.arange(nb_epoch)
figsize=(5,3)
fig, axis1 = plt.subplots(figsize=figsize)
plot1_loss = axis1.plot(epochs, loss, 'b', label='loss')
plot1_val_loss = axis1.plot(epochs, val_loss, 'r', label="val loss")
plots = plot1_loss + plot1_val_loss
labs = [l.get_label() for l in plots]
axis1.set_xlabel('Epoch')
axis1.set_ylabel('Loss')
plt.title("Loss History")
plt.tight_layout()
axis1.legend(loc='upper right')





