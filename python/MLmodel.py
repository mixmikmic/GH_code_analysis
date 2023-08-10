from keras.preprocessing import image
from matplotlib.pyplot import imshow
import os
from IPython.display import display 
from PIL import Image
from keras.models import load_model

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#from kt_utils import *
from keras import optimizers

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import glob
import pandas as pd
from sklearn.utils import shuffle

get_ipython().run_line_magic('matplotlib', 'inline')

#load images
filelist = glob.glob(os.path.join('images/a','*.jpg'))
a = np.array([np.array(Image.open(fname)) for fname in filelist])
ay = np.zeros(a.shape[0])

filelist = glob.glob(os.path.join('images/w','*.jpg'))
w = np.array([np.array(Image.open(fname)) for fname in filelist])
wy = np.ones(w.shape[0])

filelist = glob.glob(os.path.join('images/d','*.jpg'))
d = np.array([np.array(Image.open(fname)) for fname in filelist])
dy = np.ones(d.shape[0])*2

filelist = glob.glob(os.path.join('images/q','*.jpg'))
q = np.array([np.array(Image.open(fname)) for fname in filelist])
qy = np.ones(q.shape[0])*3

#randomly split into train and test
aX, aY = shuffle(a, ay, random_state=0)
wX, wY = shuffle(w, wy, random_state=0)
dX, dY = shuffle(d, dy, random_state=0)
qX, qY = shuffle(q, qy, random_state=0)
del (a, ay, w, wy, d, dy, q, qy)

#decrease sizhttp://localhost:8891/notebooks/MLmodel.ipynb#e of w
length_d = int(dX.shape[0] * 1.3)
wX = wX[:length_d]
wY = wY[:length_d]

#randomly split into train and test
allX = np.concatenate((qX, np.concatenate((np.concatenate((aX , wX)) , dX))))
allY = np.concatenate((qY, np.concatenate((np.concatenate((aY , wY)) , dY))))

#one hot encode
allY = pd.get_dummies(allY).values

del (wX, wY)

allX, allY = shuffle(allX, allY, random_state=0)
percent = int(allX.shape[0]*0.17)
testX = allX[:percent]
trainX = allX[percent:]
testY =allY[:percent]
trainY =allY[percent:]

del (allX, allY)

#Show test image
img = Image.fromarray(testX[10], 'RGB')
img.show()
print(testY[10])

def cModel(input_shape):
    
    X_input = Input(input_shape)

    # CONV -> BN -> RELU Block applied to X_input
    X = Conv2D(6, (6,6), strides=(1, 1), name='conv0', padding = 'same')(X_input)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool0')(X)
    
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(16, (5,5), strides=(1, 1), name='conv1', padding = 'valid')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool1')(X)
    
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(8, (3, 3), strides=(1, 1), name='conv2', padding = 'valid')(X)
    X = BatchNormalization(axis=3, name='bn2')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool2')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(units=4, activation='sigmoid', name='fc')(X)
    

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='cModel')

    return model
    
   

endModel = cModel(trainX.shape[1:])

optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
endModel.compile('Adam', 'mean_squared_error', metrics=['accuracy'])
#SGD ~40
#RMSprop ~90
#Adam ~94

endModel.fit(trainX, trainY, epochs=25, batch_size=64)

preds = endModel.evaluate(testX, testY, batch_size=32, verbose=1, sample_weight=None)

print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

results = (endModel.predict(np.expand_dims(testX[8], axis=0)))
#results = (endModel.predict(testX[6:7]))
print(results)
results = np.argmax(results, axis = 1)  
           
print(results[0])

endModel.save('my_model.h5', overwrite=True)

del endModel


endModel = load_model('my_model.h5')
yFit = endModel.predict(np.expand_dims(testX[8], axis=0))
yFit = np.argmax(yFit, axis = 1)  
print()
print(yFit)



