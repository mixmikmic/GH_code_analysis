import numpy as np
np.random.seed(123)    

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten    
from keras.layers import Convolution2D, MaxPooling2D    
from keras.utils import np_utils    

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/Users/av/Desktop/DataMining/myPythonFunctions')

from gradcam import *

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print ('Number of samples in training set=', 
       len(X_train), '\n Pixels =', X_train.shape[1], 'x', X_train.shape[2])

plt.imshow(X_train[0])    # Plot the first sample: sanity check
plt.show()

# STEP 1: Preprocess input data

# Convert data type to float32 and normalize to the range [0,1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255



# STEP 2: Preprocess class labels

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
 

    
# STEP 3: Define model architecture

model = Sequential() 

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) 
model.add(Dense(128, activation='relu')) 

# OUTPUT Dense layer: size=10, corresponding to 10 classes
model.add(Dense(10, activation='softmax'))    



# STEP 4: Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())



# STEP 5: Fit model on training data
model.fit(X_train, Y_train, batch_size=256, epochs=4,  validation_split=0.2, verbose=1)



# STEP 6: Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

# Visualization of grad-CAM from the test set
for i in range(10):

    print('\n Truth=', np.argmax(Y_test[i]))
    gradcam_img(X_test[i], model)

