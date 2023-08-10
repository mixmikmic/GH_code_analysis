get_ipython().run_line_magic('matplotlib', 'inline')
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import keras
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D,        UpSampling2D, Lambda, Activation, merge, AveragePooling2D,         GlobalAveragePooling2D
from keras.layers.core import Dropout, Reshape, Flatten
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import Initializer
from keras.regularizers import l2
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.datasets import mnist, cifar10
from keras.utils import to_categorical
import pydot
import graphviz
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
from scipy.misc import imsave
import bcolz
import math

'''LOAD CIFAR10'''
(x_all, y_all), (x_test, y_test) = cifar10.load_data()
x_all = x_all/255.0
x_test = x_test/255.0

x_train = x_all[:40000]
y_train = y_all[:40000]
x_validate = x_all[40000:]
y_validate = y_all[40000:]

num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_validate = to_categorical(y_validate, num_classes)
y_test = to_categorical(y_test, num_classes)

input_shape = x_train.shape
x_train.shape, y_train.shape

classifier_inp = Input(x_train.shape[1:])
conv1 = Conv2D(32, (3, 3), padding='same')(classifier_inp)
conv1 = Activation('relu')(conv1)
conv2 = Conv2D(32, (3, 3))(conv1)
conv2 = Activation('relu')(conv2)
conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv2 = Dropout(0.25)(conv2)

conv3 = Conv2D(64, (3, 3), padding='same')(conv2)
conv3 = Activation('relu')(conv3)
conv4 = Conv2D(64, (3, 3))(conv3)
conv4 = Activation('relu')(conv4)
conv4 = MaxPooling2D(pool_size=(2, 2))(conv4)
conv4 = Dropout(0.25)(conv4)

classifier = Flatten()(conv4)
classifier = Dense(512)(classifier)
classifier = Activation('relu')(classifier)
classifier = Dropout(0.5)(classifier)
classifier = Dense(num_classes)(classifier)
classifier = Activation('softmax')(classifier)

classifier = Model(classifier_inp,classifier)

classifier.compile(Adam(),
                   loss = 'categorical_crossentropy',
                   metrics=['accuracy'])

classifier.optimizer = Adam(0.00001)
classifier.fit(x_train, y_train,
          batch_size = 512,
          epochs = 3000,
          verbose = 1,
          validation_data = (x_validate, y_validate))

classifier_accuracy = classifier.evaluate(x_test, y_test, verbose=0)
print('Classifier accuracy: %.2f%%' % (classifier_accuracy[1] * 100.0))

'''Make our classifier not trainable'''
classifier.trainable = False
for layer in classifier.layers:
    layer.trainable = False

'''Prepare perception target - Functional'''
percept = Model(classifier_inp,conv2)

percept_train = percept.predict(x_train)
percept_validate = percept.predict(x_validate)

percept_train.shape, percept_validate.shape

'''Prepare adversarial target'''
y0_train = np.zeros((x_train.shape[0],10))
y0_train[:,0] = 1.0
y0_validate = np.zeros((x_validate.shape[0],10))
y0_validate[:,0] = 1.0

y1_train = np.zeros((x_train.shape[0],10))
y1_train[:,1] = 1.0
y1_validate = np.zeros((x_validate.shape[0],10))
y1_validate[:,1] = 1.0

y2_train = np.zeros((x_train.shape[0],10))
y2_train[:,2] = 1.0
y2_validate = np.zeros((x_validate.shape[0],10))
y2_validate[:,2] = 1.0

y3_train = np.zeros((x_train.shape[0],10))
y3_train[:,3] = 1.0
y3_validate = np.zeros((x_validate.shape[0],10))
y3_validate[:,3] = 1.0

y4_train = np.zeros((x_train.shape[0],10))
y4_train[:,4] = 1.0
y4_validate = np.zeros((x_validate.shape[0],10))
y4_validate[:,4] = 1.0

y5_train = np.zeros((x_train.shape[0],10))
y5_train[:,5] = 1.0
y5_validate = np.zeros((x_validate.shape[0],10))
y5_validate[:,5] = 1.0

y6_train = np.zeros((x_train.shape[0],10))
y6_train[:,6] = 1.0
y6_validate = np.zeros((x_validate.shape[0],10))
y6_validate[:,6] = 1.0

y7_train = np.zeros((x_train.shape[0],10))
y7_train[:,7] = 1.0
y7_validate = np.zeros((x_validate.shape[0],10))
y7_validate[:,7] = 1.0

y8_train = np.zeros((x_train.shape[0],10))
y8_train[:,8] = 1.0
y8_validate = np.zeros((x_validate.shape[0],10))
y8_validate[:,8] = 1.0

y9_train = np.zeros((x_train.shape[0],10))
y9_train[:,9] = 1.0
y9_validate = np.zeros((x_validate.shape[0],10))
y9_validate[:,9] = 1.0

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, (size,size), strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = Activation('relu')(x)
    return x

def residual_block(blockInput, num_filters=64):
    x = convolution_block(blockInput, num_filters, 3)
    x = convolution_block(x, num_filters, 3, activation=False)
    x = merge([x, blockInput], mode='sum')
    return x

def upsampling_block(x, filters, size):
    x = UpSampling2D()(x)
    x = Conv2D(filters, (size,size), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

classifier.trainable = False
for layer in classifier.layers:
    layer.trainable = False
    
percept.trainable = False
for layer in percept.layers:
    percept.trainable = False

'''NETWORK #0'''
inp0 = Input(x_train.shape[1:])
x0 = convolution_block(inp0, 64, 9)
x0 = residual_block(x0)
x0 = residual_block(x0)
x0 = residual_block(x0)
x0 = residual_block(x0)
out0 = Conv2D(3, (9,9), activation = 'sigmoid', padding='same')(x0)
percept0 = percept(out0)
y0 = classifier(out0)

model0 = Model(inp0, [percept0, y0])

model0.summary()

plot_model(model0, to_file='model0.png')

model0.compile(Adam(), 
            loss = {'model_2': 'mean_squared_error', 'model_1': 'categorical_crossentropy'},
            loss_weights = {'model_2': 250.0, 'model_1': 1.0}) #model_2 = perception, model_1 = classifier

model0.optimizer = Adam(0.00001)
model0.loss_weights = {'model_2': 250.0, 'model_1': 1.0} #model_2 = perception, model_1 = classifier
model0.fit(x_train, {'model_2': percept_train, 'model_1':y0_train}, 
        validation_data = (x_validate, {'model_2': percept_validate,'model_1': y0_validate}),
        epochs = 2000,
        batch_size = 512,
        shuffle = True,
        #callbacks = [model0_loss]
       )

generator = Model(inp0, out0)

x_fake = generator.predict(x_test)
x_fake.shape

'''Lets see some of the generated examples'''
num = 10
fig,axis = plt.subplots(nrows = num, ncols=2, figsize=(10,30))
for i in range(num):
    axis[i,0].imshow(x_test[i])
    axis[i,0].get_xaxis().set_visible(False)
    axis[i,0].get_yaxis().set_visible(False)
    axis[i,1].imshow(x_fake[i])
    axis[i,1].get_xaxis().set_visible(False)
    axis[i,1].get_yaxis().set_visible(False)

classifier_accuracy_test = classifier.evaluate(x_test, y_test, verbose=0)
print('Classifier accuracy on test set: %.2f%%' % (classifier_accuracy_test[1] * 100.0))

classifier_accuracy_fake = classifier.evaluate(x_fake, y_test, verbose=0)
print('Classifier accuracy on fake test set: %.2f%%' % (classifier_accuracy_fake[1] * 100.0))

labels = ['airplane', 'automobile', 'bird', 'cat',
         'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

fake_predictions = classifier.predict(x_fake)
fig,axis = plt.subplots(nrows = num, ncols=2, figsize=(10,30))
for i in range(num):
    axis[i,0].imshow(x_test[i])
    axis[i,0].get_xaxis().set_visible(False)
    axis[i,0].get_yaxis().set_visible(False)
    axis[i,1].imshow(x_fake[i,])
    axis[i,1].get_xaxis().set_visible(False)
    axis[i,1].get_yaxis().set_visible(False)
    title = 'Prediction: ' + str(np.argmax(fake_predictions[i]))            + ' - ' + labels[np.argmax(fake_predictions[i])]            + ' (' +            '%.2f%%' % (np.amax(fake_predictions[i])*100.0)            + ')'
    axis[i,1].set_title(title, fontsize = 20)

generator.save('generator_model.h5')
classifier.save('mnist_classifier_model.h5')

