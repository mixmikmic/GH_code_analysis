from keras.layers import Dense, Input
from keras.models import Model

encodingDim = 32
inputImg = Input(shape=(784,))
encodedIP = Dense(encodingDim,activation ='relu')(inputImg)
#relu = REctified Linear Unit .
decodedIP = Dense(784,activation='sigmoid')(encodedIP)

autoencoder = Model(input = inputImg,output = decodedIP)
#autoencoder takes the input image and goves the output decoded img

encoder = Model(input = inputImg,output= encodedIP)
encodedShaped = Input(shape=(encodingDim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(input = encodedShaped, output = decoder_layer(encodedShaped))

autoencoder.compile(optimizer='adam', loss = 'binary_crossentropy')

from keras.datasets import mnist
#load the digits dataset called mnist

import numpy as np

(xtrain,_),(xtest,_) = mnist.load_data()

xtrain = xtrain.astype('float32')/255
xtest = xtest.astype('float32')/255
#normalising xtrain and xtest

print (xtrain.shape)
xtrain = xtrain.reshape(len(xtrain), np.prod(xtrain.shape[1:]))
print (xtrain.shape)
#flattening 28x28 into 784

#//y for xtest
xtest = xtest.reshape(len(xtest),np.prod(xtest.shape[1:]))

autoencoder.fit(xtrain,xtrain,nb_epoch=12,verbose=1,batch_size=256,shuffle=True,
                validation_data=(xtest,xtest))

encodedImgs = encoder.predict(xtest)
decodedImgs = decoder.predict(encodedImgs)

import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
n = 10
plt.figure(figsize = (20,4))
for i in range(n):
    #displaying original
    ax = plt.subplot(2,n,i+1)
    plt.imshow(xtest[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    #displaying predictions
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decodedImgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show() 



