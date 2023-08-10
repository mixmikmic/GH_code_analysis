import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

image_path = "/home/norman/Desktop/driving_dataset/1000.jpg"

img = plt.imread(image_path)

plt.imshow(img)
plt.show()

hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
plt.imshow(hsv)
plt.show()

rand = random.uniform(0.3,1.6)
hsv[:,:,2] = rand*hsv[:,:,2]
new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
plt.imshow(new_img)
plt.show()

crop_img = img[60::, ::]
plt.imshow(crop_img)
plt.show()

def change_bright(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    rand = random.uniform(0.5,1.)
    hsv[:,:,2] = rand*hsv[:,:,2]
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
   # plt.imshow(new_img)
   # plt.show()
    return new_img

change_bright(img)

def crop_sky(img):
    
    crop_img = img[60::, ::]
   # plt.imshow(crop_img)
   # plt.show()
    return crop_img

crop_sky(img)

data_path="/home/norman/Desktop/driving_dataset/data.txt"

img_paths=[]
steers=[]
with open(data_path) as file:
    for line in file:
        if line.split(',')[0] == "center": continue
        img_paths.append("/home/norman/Desktop/driving_dataset/" + line.split(' ')[0])
        steers.append(line.split(' ')[1].strip())

img_paths, img_valid, steers, steers_valid = train_test_split(img_paths, steers, test_size = 0.10, random_state = 100) 

for i in range(10):
    img=plt.imread(img_paths[i])
    new=crop_sky(change_bright(img))
    plt.imshow(new)
    plt.show()

def gen_batch(batch_size):
    batch_x=np.zeros((batch_size,196,455,3))
    batch_y=np.zeros((batch_size,1))
    pointer=0
    (im_paths, steerss)=shuffle(img_paths, steers)
    while True:
        for i in range(batch_size):
            img=plt.imread(im_paths[pointer])
            steer=steerss[pointer]
            new_img=crop_sky(change_bright(img))
            
            batch_x[i]=new_img
            batch_y[i]=steer
            pointer+=1
            if pointer==len(im_paths)-1: pointer=0
        
        yield batch_x, batch_y

def gen_val_batch(batch_size):
    batch_x=np.zeros((batch_size,196,455,3))
    batch_y=np.zeros((batch_size,1))
    pointer=0
    (im_valid, steer_valid)=shuffle(img_valid, steers_valid)
    while True:
        for i in range(batch_size):
            img=plt.imread(im_valid[pointer])
            steer=steer_valid[pointer]
            new_img=crop_sky(change_bright(img))
            
            batch_x[i]=new_img
            batch_y[i]=steer
            pointer+=1
            if pointer==len(im_valid)-1: pointer=0
        
        yield batch_x, batch_y

import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU, concatenate, GlobalAveragePooling2D, Input
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
import math

generator=gen_batch(16)
val_gen=gen_val_batch(16)
input_shape = (196,455,3)
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))
model.add(Convolution2D(24, (5, 5), padding='valid', strides =(2,2), kernel_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(36, (5, 5), padding='valid', strides =(2,2), kernel_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(48, (5, 5), padding='valid', strides = (2,2), kernel_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), padding='same', strides = (2,2), kernel_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), padding='valid', strides = (2,2), kernel_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(80, kernel_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(40, kernel_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(16, kernel_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(10, kernel_regularizer = l2(0.001)))
model.add(Dense(1, kernel_regularizer = l2(0.001)))
adam = Adam(lr = 0.0001)
model.compile(optimizer= adam, loss='mse')
model.summary()

history= model.fit_generator(generator, steps_per_epoch = int(len(img_paths)/4-10), epochs=10, validation_data=val_gen, validation_steps = 50)

generator=gen_batch(16)
val_gen=gen_val_batch(16)
input_shape = (196,455,3)

def fire(x, squeeze=16, expand=64):
    x = Convolution2D(squeeze, (1,1), padding='valid')(x)
    x = Activation('relu')(x)
    
    left = Convolution2D(expand, (1,1), padding='valid')(x)
    left = Activation('relu')(left)
    
    right = Convolution2D(expand, (3,3), padding='same')(x)
    right = Activation('relu')(right)
    
    x = concatenate([left, right], axis=3)
    return x

img_input=Input(shape=input_shape)

x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid')(img_input)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

x = fire(x, squeeze=16, expand=16)
x = fire(x, squeeze=16, expand=16)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

x = fire(x, squeeze=32, expand=32)
x = fire(x, squeeze=32, expand=32)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

x = fire(x, squeeze=48, expand=48)
x = fire(x, squeeze=48, expand=48)
x = fire(x, squeeze=64, expand=64)
x = fire(x, squeeze=64, expand=64)
x = Dropout(0.5)(x)

x = Convolution2D(5, (1, 1), padding='valid')(x)
x = Activation('relu')(x)
x = Flatten()(x)

x = Dense(1)(x)
out = Activation('linear')(x)

modelsqueeze= Model(img_input, out)

adam = Adam(lr = 0.0001)
modelsqueeze.compile(optimizer= adam, loss='mse')
modelsqueeze.summary()

history= modelsqueeze.fit_generator(generator, steps_per_epoch = int(len(img_paths)/4-10), epochs=10, validation_data=val_gen, validation_steps = 50)


model_json = modelsqueeze.to_json()
with open("modelsqueeze.json", "w") as json_file:
    json_file.write(model_json)

modelsqueeze.save_weights("modelsqueeze.h5")

from keras.models import model_from_json
import json
json_data=open("modelnvidia.json", "r").read()

    
model = model_from_json(json_data)
model.load_weights("modelnvidia.h5")
model.compile(optimizer= adam, loss='mse')

model.summary()

