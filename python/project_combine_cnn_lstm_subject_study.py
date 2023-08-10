from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Activation, Flatten
from keras.utils import to_categorical
import numpy as np
import h5py
import matplotlib.pyplot as plt
from data_utils_subject import get_data
from sklearn import preprocessing

# Load data from all .mat files, combine them, eliminate EOG signals, shuffle and 
# seperate training data, validation data and testing data.
# Also do mean subtraction on x.

for sub in range(9):
    data = get_data('../project_datasets', subject=sub+1, num_validation=30, num_test=30,
                     subtract_mean=True, subtract_axis=1, transpose=True)
    for k in data.keys():
        print('{}: {} '.format(k, data[k].shape))

    #---------------------------------------------------------------------------------        
    num_classes = 4

    # substract data from list
    X_train = data.get('X_train')
    y_train = data.get('y_train')
    X_val = data.get('X_val')
    y_val = data.get('y_val')
    X_test = data.get('X_test')
    y_test = data.get('y_test')
    
    # get data dimension
    N_train, T_train, C_train = data.get('X_train').shape
    N_val, T_val, C_val = data.get('X_val').shape
    N_test, T_test, C_test = data.get('X_test').shape
    
    # add dummy zeros for y classification, convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes) 
    y_test = to_categorical(y_test, num_classes)
    
    #---------------------------------------------------------------------------------
    # construct X_total and y_total based on sub-sampling of X_train and y_train

    # take sub-sampling on the time sequence to reduce dimension for RNN
    sampling = 1

    X_train = X_train.reshape(N_train,int(T_train/sampling), sampling, C_train)[:,:,0,:]
    X_val = X_val.reshape(N_val,int(T_val/sampling), sampling, C_val)[:,:,0,:]
    X_test = X_test.reshape(N_test,int(T_test/sampling), sampling, C_test)[:,:,0,:]
    
    # get new data dimension
    N_train, T_train, C_train = X_train.shape
    N_val, T_val, C_val = X_val.shape
    N_test, T_test, C_test = X_test.shape
    
    print('X_train: ', X_train.shape)
    print('y_train: ', y_train.shape)
    print('X_val: ', X_val.shape)
    print('y_val: ', y_val.shape)
    print('X_test: ', X_test.shape)
    print('y_test: ', y_test.shape)
    
    #---------------------------------------------------------------------------------
    # Expected input batch shape: (batch_size, timesteps, data_dim)
    # Note that we have to provide the full batch_input_shape since the network is stateful.
    # the sample of index i in batch k is the follow-up for the sample i in batch k-1.

    # perhaps should try masking layer

    data_dim = C_train
    timesteps = T_train
    batch_size = 100
    num_epoch = 70

    # make a sequential model
    model = Sequential()

    # add 1-layer cnn
    model.add(Conv1D(20, kernel_size=12, strides=4,
              input_shape=(timesteps, data_dim)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4, strides=4))


    # add 2-layer lstm
    model.add(LSTM(25, return_sequences=True, stateful=False))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(LSTM(15, return_sequences=True, stateful=False))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    # set loss function and optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    #---------------------------------------------------------------------------------
    # train the data with validation
    history = model.fit(X_train, y_train,
                        batch_size=batch_size, 
                        epochs=num_epoch, 
                        shuffle=False,
                        validation_data=(X_val, y_val))

    #---------------------------------------------------------------------------------
    # list all data in history
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'],'o')
    plt.plot(history.history['val_loss'],'o')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    # test set
    print(model.evaluate(X_test,y_test,batch_size=N_test))

data = get_data('../project_datasets', subject=9, num_validation=30, num_test=30,
                 subtract_mean=True, subtract_axis=1, transpose=True)
for k in data.keys():
    print('{}: {} '.format(k, data[k].shape))

#---------------------------------------------------------------------------------        
num_classes = 4

# substract data from list
X_train = data.get('X_train')
y_train = data.get('y_train')
X_val = data.get('X_val')
y_val = data.get('y_val')
X_test = data.get('X_test')
y_test = data.get('y_test')

# get data dimension
N_train, T_train, C_train = data.get('X_train').shape
N_val, T_val, C_val = data.get('X_val').shape
N_test, T_test, C_test = data.get('X_test').shape

# add dummy zeros for y classification, convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes) 
y_test = to_categorical(y_test, num_classes)

#---------------------------------------------------------------------------------
# construct X_total and y_total based on sub-sampling of X_train and y_train

# take sub-sampling on the time sequence to reduce dimension for RNN
sampling = 1

X_train = X_train.reshape(N_train,int(T_train/sampling), sampling, C_train)[:,:,0,:]
X_val = X_val.reshape(N_val,int(T_val/sampling), sampling, C_val)[:,:,0,:]
X_test = X_test.reshape(N_test,int(T_test/sampling), sampling, C_test)[:,:,0,:]

# get new data dimension
N_train, T_train, C_train = X_train.shape
N_val, T_val, C_val = X_val.shape
N_test, T_test, C_test = X_test.shape

print('X_train: ', X_train.shape)
print('y_train: ', y_train.shape)
print('X_val: ', X_val.shape)
print('y_val: ', y_val.shape)
print('X_test: ', X_test.shape)
print('y_test: ', y_test.shape)

#---------------------------------------------------------------------------------
# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.

# perhaps should try masking layer

data_dim = C_train
timesteps = T_train
batch_size = 100
num_epoch = 70

# make a sequential model
model = Sequential()

# add 1-layer cnn
model.add(Conv1D(20, kernel_size=12, strides=4,
          input_shape=(timesteps, data_dim)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=4, strides=4))


# add 2-layer lstm
model.add(LSTM(25, return_sequences=True, stateful=False))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(LSTM(15, return_sequences=True, stateful=False))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

# set loss function and optimizer
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#---------------------------------------------------------------------------------
# train the data with validation
history = model.fit(X_train, y_train,
                    batch_size=batch_size, 
                    epochs=num_epoch, 
                    shuffle=False,
                    validation_data=(X_val, y_val))

#---------------------------------------------------------------------------------
# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'],'o')
plt.plot(history.history['val_loss'],'o')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# test set
print(model.evaluate(X_test,y_test,batch_size=N_test))



