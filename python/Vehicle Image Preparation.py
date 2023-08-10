import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from glob import glob
from scipy.misc import imread
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
from keras.preprocessing.image import ImageDataGenerator

get_ipython().magic('matplotlib inline')

with open('../data/data.p', 'rb') as f:
    data = pickle.load(f)
    
X_train, y_train = data['train']

print('Length X_train', len(X_train))
print('Length y_train', len(y_train))

cars = X_train[y_train==1]
_ = plt.hist(np.mean(cars[:, 16:48, 16:48], axis=(1,2,3)), bins=20)

print('mean:', np.mean(cars[:, 16:48, 16:48], axis=(1,2,3)).mean())
print('std:', np.mean(cars[:, 16:48, 16:48], axis=(1,2,3)).std())

wht_filter = np.mean(cars[:, 16:48, 16:48], axis=(1,2,3)) > 110

idg = ImageDataGenerator(
    rotation_range=20.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    channel_shift_range=20,
    horizontal_flip=True)

fig, axis = plt.subplots(1,3, figsize=(11,10))
for i, (x,y) in enumerate(idg.flow(cars[wht_filter], np.ones(len(cars[wht_filter])), batch_size=1)):
    if i == 3: break
    axis[i%3].imshow(x[0].astype(np.uint8))

BATCH_SIZE = 100
BATCHES = 25

for i, (x,y) in enumerate(idg.flow(cars[wht_filter], np.ones(len(cars[wht_filter])), batch_size=BATCH_SIZE)):
    if i == BATCHES: break
        
    X_train = np.concatenate((X_train, x.astype(np.uint8)), axis=0)
    y_train = np.concatenate((y_train, y.astype(np.uint8)), axis=0)

cars_new = X_train[y_train==1]

fig, axis = plt.subplots(1,2, figsize=(11,3))
_ = axis[0].hist(np.mean(cars[:, 16:48, 16:48], axis=(1,2,3)), bins=20)
_ = axis[0].set_title('Old distribution')

_ = axis[1].hist(np.mean(cars_new[:, 16:48, 16:48], axis=(1,2,3)), bins=20)
_ = axis[1].set_title('New distribution')

print('Length X_train', len(X_train))
print('Length y_train', len(y_train))

data['train'] = (X_train, y_train)

with open('../data/data_adj.p', 'wb') as f:
    pickle.dump(data, f)

