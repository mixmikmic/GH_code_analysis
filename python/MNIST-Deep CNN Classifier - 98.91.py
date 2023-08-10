import os  
#os.environ['THEANO_FLAGS'] = "device=gpu1"  
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=1"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"
  
#%pylab inline
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score


import theano
from keras.layers import Input, Dense
from keras.layers import Activation, Dropout, Convolution2D, Flatten, MaxPooling2D, Reshape, InputLayer
from keras.models import Sequential, Model
from keras.optimizers import *
from keras.utils.np_utils import *

from sklearn.cross_validation import train_test_split

x_train = np.genfromtxt('x_train.out')
y_train = np.genfromtxt('y_train.out')
vx_train = np.genfromtxt('vx_train.out')
vy_train = np.genfromtxt('vy_train.out')
x_test = np.genfromtxt('x_test.out')
y_test = np.genfromtxt('y_test.out')

print (x_train.shape)
print (y_train.shape)
print (vx_train.shape)
print (vy_train.shape)
print (x_test.shape)
print (y_test.shape)

label_train=to_categorical(y_train)
label_valid=to_categorical(vy_train)
label_test= to_categorical(y_test)

print (x_train.shape)
print (label_train.shape)
print (vx_train.shape)
print (label_valid.shape)
print (x_test.shape)
print (label_test.shape)

# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)

# reshape data

#train_x_temp = x_train.reshape(-1, 28, 28, 1)
#val_x_temp = vx_train.reshape(-1, 28, 28, 1)
#test_x_temp=x_test.reshape(-1, 28, 28, 1)

train_x_temp = x_train.reshape(-1,1, 28, 28)
val_x_temp = vx_train.reshape(-1,1, 28, 28)
test_x_temp=x_test.reshape(-1,1, 28, 28)
print(train_x_temp.shape)
print(val_x_temp.shape)
print(test_x_temp.shape)


# define vars
input_shape = (784,)
input_reshape = (1,28, 28)

conv_num_filters = 5
conv_filter_size = 5

pool_size = (2, 2)

hidden_num_units = 50
output_num_units = 10

epochs = 5
batch_size = 128


# this is our input placeholder
input_img = Input(shape=(input_reshape))
x = Convolution2D(25, 5, 5, border_mode='same',activation='relu')(input_img) 
x = MaxPooling2D(pool_size=pool_size)(x)
x = Convolution2D(25, 5, 5, border_mode='same',activation='relu')(x)
x = MaxPooling2D(pool_size=pool_size)(x)
x = Convolution2D(25, 4, 4, border_mode='same',activation='relu')(x)
x = Flatten()(x)
x = Dense(output_dim=hidden_num_units, activation='relu')(x)
x = Dense(output_dim=output_num_units, activation='softmax')(x)
                  
model = Model(input=input_img, output=x)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

trained_model_conv = model.fit(train_x_temp, label_train, 
                               nb_epoch=epochs, batch_size=batch_size, 
                               validation_data=(val_x_temp, label_valid))

#pred = model.predict_classes(x_test)
# Keras Model nulike Sequential does not support predict_classes
y_pred = model.predict(test_x_temp)

y_pred

new_y_pred=[]
for i in range (len(y_pred)):
    new_y_pred.append([np.argmax(y_pred[i])])

new_y_pred=np.asarray(new_y_pred)
print (new_y_pred.shape)
new_y_pred=to_categorical(new_y_pred)
print (new_y_pred.shape)

new_y_pred

label_test

num=len(label_test)
r=0
w=0
for i in range(num):
        #print ('y_pred ',y_pred[i])
        #print ('labels ',labels[i])
        #without the use of all() returns error truth value of an array with more than one element is ambiguous
        #if y_pred[i].all() == labels[i].all():
        if np.array_equal(new_y_pred[i],label_test[i]):
            r+=1
        else:
            w+=1
print ("tested ",  num, "digits")
print ("correct: ", r, "wrong: ", w, "error rate: ", float(w)*100/(r+w), "%")
print ("got correctly ", float(r)*100/(r+w), "%")



