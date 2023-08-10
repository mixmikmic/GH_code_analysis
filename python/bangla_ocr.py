import cv2
import os
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def load_dataset(folder,size=30,randomize=True):
    X_train=[]
    Y_train=[]
    X_test=[]
    Y_test=[]
    for file in os.listdir(folder):
        x=[]
        y=[]
        for f in os.listdir(folder+'/'+file):
            im=cv2.imread(folder+'/'+file+'/'+f,0)
            im=cv2.resize(im,(size,size))
            im=im/255.
            im=np.reshape(im,(im.shape[0],im.shape[1],1))
            x.append(im)
            y.append(int(file))
        X_train+=x[:1500]
        Y_train+=y[:1500]
        X_test+=x[1500:1900]
        Y_test+=y[1500:1900]
        print('loading . .',file)
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    Y_train=np.array(Y_train)
    Y_test=np.array(Y_test)
    return X_train,Y_train,X_test,Y_test



X_train,Y_train,X_test,Y_test=load_dataset('dataset')
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)

input_shape=(30,30,1)
num_classes=10
batch_size=100
epochs=5

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
validation_data=(X_test, Y_test))

model.save('model.h5')

