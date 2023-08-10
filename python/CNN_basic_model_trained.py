### load libraries ####

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation, Conv2D, MaxPooling2D, Flatten

from keras.optimizers import SGD
from keras import backend as K

import pandas as pd
import numpy as np
import matplotlib
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

# image processing 
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imresize
import matplotlib.image as mpimg
import io
import urllib

from keras.callbacks import TensorBoard


############# RUNNING LOCALLY: Read in data FROM FILE ################

# train = pd.read_csv('../data/train_data_with_sampling.csv')
# test = pd.read_csv('../data/test_data.csv')

############# FOR AWS: Read in data FROM GITHUB ################

train = pd.read_csv('https://raw.githubusercontent.com/nikhilaravi/movies/master/data/train_data_with_sampling.csv')
test = pd.read_csv('https://raw.githubusercontent.com/nikhilaravi/movies/master/data/test_data.csv')

############# Convert to test and train set ################

X_train_ids = train[['tmdb_id']].values
y_train = train[['group1', 'group2', 'group3', 'group4', 'group5', 'group6', 'group7']].as_matrix()
X_test_ids = test[['tmdb_id']].values
y_test = test[['group1', 'group2', 'group3', 'group4', 'group5', 'group6', 'group7']].as_matrix()

############## Function to read poster image from github given tmdb id ###################

def loadImageFromId(tmdb_id):
    url = 'https://raw.githubusercontent.com/nikhilaravi/movies/master/posters/' + tmdb_id + '.jpg'
    file = io.BytesIO(urllib.request.urlopen(url).read())
    img = Image.open(file)
    return img

############# Load in posters ################

def load_posters_from_ids(id_array, n, y):
    
    ### Use these three lines to subset the training and testing data when running locally (can't load 7000 images)
    
    # start = np.random.choice(range(len(id_array)-n))
    # end = start + n
    # Y = y[:n]
    Y = y
    posters = []
    ids = []
    errors = 0
    
    for poster in id_array:
        
        # uncomment this line if running on AWS 
        img = loadImageFromId(str(int(poster[0])))
        
        # uncomment this line if running locally
#         img = Image.open('../posters/' + str(int(poster[0])) + '.jpg') 
        
        try:
            # original scale 500 x 741 but need to reduce resolution 
            scaled = img.resize(size=(100, int(741/5)))
            posters.append(np.asarray(scaled))
            ids.append(poster)
            
        except ValueError:
            errors +=1
            continue
            
    posters_array = np.asarray(posters)
    X = np.array(posters_array).astype('float32')
    
    print ('posters shape: ', X.shape)
    print ('errors: ', errors)
    
    # rescale pixel values
    
    return X/255, Y

#### load testing and training data 

X_train, Y_train = load_posters_from_ids(X_train_ids, 500, y_train)
X_test, Y_test = load_posters_from_ids(X_test_ids, 100, y_test)

############# Visualise first 5 images of training set ################

plt.figure(figsize=(50,50))
for i in range(1,6):
    plt.subplot(1, 5, i)
    plt.imshow(X_train[i])

X_train.shape

n_out = 7

####### Callbacks ##########
early_stopping = keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0, patience=0, verbose=0, mode='auto')
tensorboard = TensorBoard(log_dir='./logs/basic_model', histogram_freq=1, write_graph=True, write_images=False)

####### Model ##########
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(X_train.shape[1], X_train.shape[2], 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# output is a 3D feature map (height, width, features)

# add two fully connected output layers

model.add(Flatten()) # 1D feature vector
model.add(Dense(units=64, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=n_out, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
             optimizer='adam')

model.summary()

batch_size = 10
epochs = 20

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1, callbacks = [early_stopping, ])

# model.save_weights('simple_model.h5')

# These are how we measure error - Haming Loss, % exact matches and % at-least-one match
from sklearn.metrics import hamming_loss

def error_measures(ypred, ytest):
    ypred = np.array(ypred)
    ytest = np.array(ytest)
    
    # Hamming loss
    h_loss = hamming_loss(ytest, ypred)

    # Percent exact matches
    y_pred_str = np.array([str(yi) for yi in ypred])
    y_test_str = np.array([str(yi) for yi in ytest])
    percent_exact = np.sum(y_pred_str == y_test_str) * 1. / ytest.shape[0]
    
    # Percent at least one match (at least one of the genres are both 1)
    atleastone_count = 0
    for ind in range(len(ypred)):
        yi_pred = ypred[ind]
        yi_test = ytest[ind]
        for i in range(len(yi_pred)):
            if yi_pred[i] == 1 and yi_test[i] == 1:
                atleastone_count += 1
                break
    percent_atleastone = atleastone_count * 1. / ytest.shape[0]
    
    return h_loss, percent_exact, percent_atleastone

################# Make predictions on the testing set ###################
y_pred_proba = model.predict(X_test, 
                             batch_size=batch_size)

# Predict after casting probabilities to labels
y_pred = []
for yi in y_pred_proba:
    yi_new = np.array([1 if yii >= 0.5 else 0 for yii in yi])
    y_pred.append(yi_new)
y_pred = np.array(y_pred)

X_test.shape

# compute error measures
h_loss, percent_exact, percent_atleastone = error_measures(y_pred, Y_test)
print ('CNN error measures \n=====================\n')
print ('Hamming Loss: ', h_loss)
print ('Percent exact: ', percent_exact)
print ('Percent at least one: ', percent_atleastone)

######## check number in each class in the training set ############
print ('class counts training: ', np.sum(Y_train, axis=0))
print ('class counts testing: ', np.sum(Y_test, axis=0))
print ('class counts prediction: ', np.sum(y_pred, axis=0))



