import nbimporter
from Dataset_Preparation import *
import numpy as np
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn import preprocessing
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json

def loadDatasetFromFile():
    X = np.load('NP-Dataset/X.npy')
    y = np.load('NP-Dataset/y.npy')
    #Train-Test Split. random_state equivalent to seed()
#     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=1)
    indices = sss.split(X,y)
    indices = list(indices)
#     print ("Indices: ",indices[0])
    trainIndices,testIndices = indices[0][0],indices[0][1] 
    trainX = X[trainIndices]
    trainY = y[trainIndices]
    testX = X[testIndices]
    testY = y[testIndices]
    return trainX,testX,trainY,testY
#     return X_train,X_test,y_train,y_test

def oneHotEncoded_y(y):
    
    encoder = preprocessing.LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)
    encoded_y = encoded_y.reshape(-1,1)
    categorical_y = np_utils.to_categorical(encoded_y)
    
    return categorical_y

def normalizedDataset():
    #Reshaping and 
    X_train,X_test,y_train,y_test = loadDatasetFromFile()
    X_train = np.reshape(X_train,(-1,1024))
    X_test = np.reshape(X_test,(-1,1024))
    
    X_train /= 255
    X_test /= 255
    
    y_train = oneHotEncoded_y(y_train)
    y_test = oneHotEncoded_y(y_test)
    
    return X_train,X_test,y_train,y_test

def getProbableClass(x):
    index = np.argmax(x)
    return index

def mal_char_data():
    
    X_train,X_test,y_train,y_test = normalizedDataset()
    
    n_rows = 32
    n_cols = 32
    
    X_train = X_train.reshape(X_train.shape[0],n_rows,n_cols,1)
    X_test = X_test.reshape(X_test.shape[0],n_rows,n_cols,1)

    return X_train,X_test,y_train,y_test

X_train,X_test,y_train,y_test = mal_char_data()

X_train.shape,X_test.shape,y_train.shape,y_test.shape

datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest')

datagen.flow(X_train,y_train)

i = 0
for batch in datagen.flow(X_train, batch_size=1,
                          save_to_dir='NP-Dataset/', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely

