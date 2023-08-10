import os    
#os.environ['THEANO_FLAGS'] = "device=gpu1"  
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=1"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"

from keras.layers import Input, Dense, convolutional,Reshape, Flatten, LeakyReLU, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


img_rows, img_cols = 32, 32

nb_filters_1 = 96
nb_filters_2 = 192
nb_filters_3 = 256
nb_conv = 1
nb_conv_mid = 3
nb_conv_init = 5

def conv(init, nb_filter, row, col, subsample=(1,1), repeat=0):
    c = convolutional.Convolution2D(nb_filter, row, col, border_mode='same', subsample=subsample)(init)
    c = LeakyReLU()(c)

    for i in range(repeat):
        c = convolutional.Convolution2D(nb_filter, row, col, border_mode='same', subsample=subsample)(c)
        c = LeakyReLU()(c)
    return c

init = Input(shape=(3, img_rows, img_cols),)

fork11 = conv(init, nb_filters_1, nb_conv_init, nb_conv_init)
conv_pool1 = conv(fork11, nb_filters_1, nb_conv_init, nb_conv_init)
conv_pool1 = convolutional.MaxPooling2D((2, 2), border_mode='same')(conv_pool1)

fork21 = conv(conv_pool1, nb_filters_2, nb_conv_mid, nb_conv_mid)
conv_pool2 = conv(fork21, nb_filters_2, nb_conv_mid, nb_conv_mid)
conv_pool2 = convolutional.MaxPooling2D((2, 2), border_mode='same')(conv_pool2)

fork31 = conv(conv_pool2, nb_filters_3, nb_conv_mid, nb_conv_mid)
conv_pool3 = conv(fork31, nb_filters_2, nb_conv_mid, nb_conv_mid)

encoded = convolutional.MaxPooling2D((2, 2), border_mode='same')(conv_pool3)

conv_poold13d = conv(encoded, nb_filters_2, nb_conv_mid, nb_conv_mid)
fork13d = conv(conv_poold13d, nb_filters_3, nb_conv_mid, nb_conv_mid)

ups13d = convolutional.UpSampling2D((2, 2))(fork13d)
conv_pool2d = conv(ups13d, nb_filters_2, nb_conv_mid, nb_conv_mid)
fork12d = conv(conv_pool2d, nb_filters_2, nb_conv_mid, nb_conv_mid)

ups12d = convolutional.UpSampling2D((2, 2))(fork12d)
conv_pool1d = conv(ups12d, nb_filters_1, nb_conv_init, nb_conv_init)
ups11d = convolutional.UpSampling2D((2, 2))(conv_pool1d)

decoded = convolutional.Convolution2D(3, 5, 5, activation='sigmoid', border_mode='same')(ups11d)

autoencoder = Model(init, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=['accuracy'])

print (autoencoder.summary())

X_train=np.load('/media/mrafi123/UStore/Dimensionality-Reduction/data/CIFAR10/X_train.npy')
X_test=np.load('/media/mrafi123/UStore/Dimensionality-Reduction/data/CIFAR10/X_test.npy')
y_train=np.load('/media/mrafi123/UStore/Dimensionality-Reduction/data/CIFAR10/y_train.npy')
y_test=np.load('/media/mrafi123/UStore/Dimensionality-Reduction/data/CIFAR10/y_test.npy')

print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)

x_train = X_train.astype('float32') / 255.
x_test = X_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)

x_train = x_train.reshape((len(x_train),3,32,32))
x_test = x_test.reshape((len(x_test),3,32,32))
print (x_train.shape)
print (x_test.shape)

# this model maps an input to its encoded representation
encoding_dim=40
encoded = Flatten()(encoded)
encoded = Dense(encoding_dim, activation='sigmoid')(encoded)
encoder = Model(init, encoded)

print (encoder.summary())

X_train=np.load('/media/mrafi123/UStore/Dimensionality-Reduction/data/CIFAR10/X_train.npy')
X_test=np.load('/media/mrafi123/UStore/Dimensionality-Reduction/data/CIFAR10/X_test.npy')
y_train=np.load('/media/mrafi123/UStore/Dimensionality-Reduction/data/CIFAR10/y_train.npy')
y_test=np.load('/media/mrafi123/UStore/Dimensionality-Reduction/data/CIFAR10/y_test.npy')

print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)

x_train = X_train.astype('float32') / 255.
x_test = X_test.astype('float32') / 255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)

x_train = x_train.reshape((len(x_train),3,32,32))
x_test = x_test.reshape((len(x_test),3,32,32))
print (x_train.shape)
print (x_test.shape)



nfold=5
nb_epoch=1
batch_size=256
random_state =5435

kfold_weights_path = os.path.join('weights_kfold_' + str(nfold) + 
                                  '_epoch_'+str(nb_epoch)+
                                  '_batch_'+str(batch_size)
                                  +'.h5')
print(kfold_weights_path)

kf = KFold(len(x_train), n_folds=nfold, shuffle=True, random_state=random_state)
print(kf)

train_full_encoded_imgs = np.zeros(shape=[x_train.shape[0],encoding_dim])
print(train_full_encoded_imgs.shape)

test_full_encoded_imgs = np.zeros(shape=[x_test.shape[0],encoding_dim])
test_full_decoded_imgs = np.zeros(shape=[x_test.shape[0],3,32,32])
print(test_full_encoded_imgs.shape)
print(test_full_decoded_imgs.shape)

os.path.isfile(kfold_weights_path)

num_fold = 0
restore_from_last_checkpoint=0
for train_index,valid_index in kf:
    s_train,s_valid  = x_train[train_index], x_train[valid_index]
    
    #noise_factor = 0.1
    #s_train_noisy = s_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=s_train.shape) 
    #s_valid_noisy = s_valid + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=s_valid.shape) 

    #s_train_noisy = np.clip(s_train_noisy, 0., 1.)
    #s_valid_noisy = np.clip(s_valid_noisy, 0., 1.)
    #y_train, y_test = y[train_index], y[test_index]
    
    num_fold += 1
    print('\n\nStart KFold number {} from {}'.format(num_fold, nfold))
    print('Split train: ', len(s_train), len(s_train))
    print('Split valid: ', len(s_train), len(s_valid))

    if not os.path.isfile(kfold_weights_path) or restore_from_last_checkpoint == 0:
        callbacks = [
                EarlyStopping(monitor='val_loss', patience=1, verbose=1),
                ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=1),
        ]
        autoencoder.fit(s_train, s_train,
                nb_epoch=nb_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(s_valid, s_valid),
                callbacks=callbacks)
    #if os.path.isfile(kfold_weights_path):
    #    autoencoder.load_weights(kfold_weights_path)
    
    # Store train predictions
    train_encoded_imgs = encoder.predict(x_train,batch_size=batch_size, verbose=1)
    train_full_encoded_imgs = np.add(train_full_encoded_imgs,train_encoded_imgs)

    
    # Store test predictions
    test_encoded_imgs = encoder.predict(x_test,batch_size=batch_size, verbose=1)
    #full_encoded_imgs = np.vstack([full_encoded_imgs,encoded_imgs])
    test_full_encoded_imgs = np.add(test_full_encoded_imgs,test_encoded_imgs)
    #full_encoded_imgs.append(encoded_imgs)
    #print(full_encoded_imgs.shape)
    
    test_decoded_imgs = autoencoder.predict(x_test,batch_size=batch_size, verbose=1)
    #full_decoded_imgs = np.vstack([full_decoded_imgs,decoded_imgs])
    test_full_decoded_imgs = np.add(test_full_decoded_imgs,test_decoded_imgs)



print(test_full_encoded_imgs.shape)
print(test_full_decoded_imgs.shape)
print(test_full_encoded_imgs)
print(test_full_decoded_imgs)

test_res_encoded_imgs = test_full_encoded_imgs/nfold
test_res_decoded_imgs = test_full_decoded_imgs/nfold
print(test_res_encoded_imgs.shape)
print(test_res_decoded_imgs.shape)
print(test_res_encoded_imgs)
print(test_res_decoded_imgs)

train_res_encoded_imgs = train_full_encoded_imgs/nfold
print(train_res_encoded_imgs.shape)
print(train_res_encoded_imgs)

import matplotlib.pyplot as plt
test_res_decoded_imgs=test_res_decoded_imgs.reshape(10000,3,32,32)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    #ax = plt.subplot(1, n, i)
    #Plot the raw original image
    plt.imshow(x_test[i].reshape(32,32,3)) 
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(test_res_decoded_imgs[i].reshape(32,32,3))    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf = clf.fit(train_res_encoded_imgs, np.ravel(y_train))

y_pred = clf.predict(test_res_encoded_imgs)



num=len(test_res_encoded_imgs)
r=0
w=0
y_test = np.ravel(y_test)
for i in range(num):
        #print ('y_pred ',y_pred[i])
        #print ('labels ',labels[i])
        #without the use of all() returns error truth value of an array with more than one element is ambiguous
        #if y_pred[i].all() == labels[i].all():
        if np.array_equal(y_pred[i],y_test[i]):
            r+=1
        else:
            w+=1
print ("tested ",  num, "digits")
print ("correct: ", r, "wrong: ", w, "error rate: ", float(w)*100/(r+w), "%")
print ("got correctly ", float(r)*100/(r+w), "%")

y_pred

y_test



