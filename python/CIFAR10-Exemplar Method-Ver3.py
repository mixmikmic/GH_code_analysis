import os    
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda,floatX=float32"

import numpy as np
from sklearn.utils import shuffle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import math

from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, UpSampling3D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.layers.core import Flatten
from keras.constraints import maxnorm

import keras.regularizers

from keras.optimizers import SGD, Adam
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


img_rows = 32
img_cols = 32

#test for gpu
from theano import function, config, shared, tensor
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')



# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')






split = 50
divided_input = np.split(x_train, split)
X_train = divided_input[0]

divided_output = np.split(y_train, split)
Y_train = divided_output[0]

unique, counts = numpy.unique(Y_train, return_counts=True)
print (dict(zip(unique, counts)))

print(X_train.shape)
print(Y_train.shape)
print (len(X_train))

Y_train_Surr = np.arange(len(X_train))

Y_train_Surr = Y_train_Surr.reshape(len(X_train),1)
Y_train_Surr.shape

X_train = X_train.astype('float32')
X_train /= 255
print(X_train.shape)

print('Using real-time data augmentation.')
# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        shear_range=0.3, #Float. Shear Intensity (Shear angle in counter-clockwise direction as radians)
        zoom_range=0.3 , #Float or [lower, upper]. Range for random zoom. If a float,  [lower, upper] = [1-zoom_range, 1+zoom_range].
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images



i=0
generated_images = np.empty(shape=[0,3,32,32])
generated_labels = np.empty(shape=[0,1])


for i in range(50):
    
    for X_batch,Y_batch in datagen.flow(X_train,Y_train_Surr,batch_size=len(X_train)):
        generated_images = np.concatenate((generated_images, X_batch))
        generated_labels = np.concatenate((generated_labels, Y_batch))
        break  # otherwise the generator would loop indefinitely

    i+=1
    if i >= 50:
        break

print (X_train.shape)
print (Y_train_Surr.shape)

print (X_batch.shape)
print (Y_batch.shape)
print (generated_images.shape)
print (generated_labels.shape)



image = X_train.reshape(X_train.shape[0], 3, 32, 32).transpose(0,2,3,1)
print (image.shape)
#print (image[0])
image = image*127.5+127.5
#print (image[0])
Image.fromarray(image[0].astype(np.uint8)).save("actual_image.png")

unique, counts = numpy.unique(generated_labels, return_counts=True)
print (dict(zip(unique, counts)))

num_classes = len(X_train)
generated_labels = np_utils.to_categorical(generated_labels, num_classes)

print (generated_images.shape)
print (generated_labels.shape)

from sklearn.model_selection import train_test_split
generated_images_train, generated_images_val, generated_labels_train, generated_labels_val = train_test_split(
    generated_images,generated_labels,test_size=0.1, random_state=42)

print (generated_images_train.shape)
print (generated_labels_train.shape)
print (generated_images_val.shape)
print (generated_labels_val.shape)



# Architecture is changed from ver-2 and 
# no of surrogate classes increased
def cnn_model():
    model = Sequential()
    model.add(Convolution2D(64, 5, 5, border_mode='same',subsample=(2, 2),input_shape=(3, 32, 32)))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(128, 5, 5, border_mode='same',subsample=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(256, 5, 5, border_mode='same',subsample=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(512, 5, 5, border_mode='same',subsample=(4, 4)))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model


model2 = cnn_model()
modeladam = cnn_model()
modelada = cnn_model()

lr = 0.01

sgd = SGD(lr=lr, decay=1e-5, momentum=0.8, nesterov=True)

model2.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])

modeladam.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

modelada.compile(loss='categorical_crossentropy',
          optimizer='adadelta',
          metrics=['accuracy'])

modeladam.summary()

nb_epochs=2
batch_size = 128
kfold_weights_path = os.path.join('weights_' +  'CIFAR10-Exemplar-Ver5.0' +
                                  '_epoch_'+str(nb_epochs)+
                                  '_batch_'+str(batch_size)
                                  +'.h5')
print(kfold_weights_path)

os.path.isfile(kfold_weights_path)

# Some transfer learning
if os.path.isfile(kfold_weights_path):
    print ('Loading already stored weights...')
    modeladam.load_weights(kfold_weights_path)
else:
    print ('Training for the first time...')
    

print (generated_images_train.shape)
print (generated_labels_train.shape)
print (generated_images_val.shape)
print (generated_labels_val.shape)

image = generated_images_train.reshape(generated_images_train.shape[0], 3, 32, 32).transpose(0,2,3,1)
print (image.shape)
#print (image[0])
image = image*127.5+127.5
#print (image[0])
Image.fromarray(image[1].astype(np.uint8)).save("generated_image.png")

get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt

n = 36000  # how many digits we will display
rows=20
columns=10
fig, axs = plt.subplots(rows,columns,figsize=(20, 22))

plt.gray()
fig.subplots_adjust(hspace = .5, wspace=.001)

#for r in range(rows):
r=0
c=0

image = generated_images_train.reshape(generated_images_train.shape[0], 3, 32, 32).transpose(0,2,3,1)
image = image*127.5+127.5


for i in range(n):
    if np.argmax(generated_labels_train[i]) == 15:
        axs[r, c].imshow(Image.fromarray(image[i].astype(np.uint8)))
        axs[r, c].set_title(np.argmax(generated_labels_train[i]))
        axs[r, c].get_xaxis().set_visible(False)
        axs[r, c].get_yaxis().set_visible(False)
        #print (r,c)
        c=c+1
        if c==10:
            #print ('breaking to next row')
            r=r+1
            c=0

plt.show()

nb_epochs=50

callbacks = [
                EarlyStopping(monitor='val_loss', patience=2, verbose=1),
                ModelCheckpoint(kfold_weights_path, monitor='val_loss', 
                                save_best_only=True, 
                                verbose=1),
            ]
modeladam.fit(generated_images_train, generated_labels_train,
            nb_epoch=nb_epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(generated_images_val, generated_labels_val),
            callbacks=callbacks
            )

# Now that we have generated a weights file
# we need a method to do some transfer learning
# train the last few layers to predict the original 10 classes

feature_layers = [
    Convolution2D(64, 5, 5, border_mode='same', subsample=(2, 2),input_shape=(3, 32, 32), activation='relu'),
    Dropout(0.5),
    Convolution2D(128, 5, 5, border_mode='same', subsample=(2,2) , activation='relu'),
    Dropout(0.5),
    Convolution2D(256, 5, 5, border_mode='same', subsample=(2,2) , activation='relu'),
    Dropout(0.5),
    Convolution2D(512, 5, 5, border_mode='same', subsample=(4,4) , activation='relu'),
    Dropout(0.5),
    Flatten()
]

classification_layers = [
    Dense(512, W_regularizer=keras.regularizers.l2(0.01), name='fc_layer1'),
    Activation('sigmoid'),
    Dense(10, activation='softmax', W_regularizer=keras.regularizers.l2(0.01), name='fc_layer2')
]


#print model.summary()

model = Sequential(feature_layers + classification_layers)

model.load_weights(kfold_weights_path, by_name=True)

for l in feature_layers:
    print (l)
    l.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

print('Model Compilation successful')

lr = 0.1

sgd = SGD(lr=lr, decay=1e-5, momentum=0.8, nesterov=True)

model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])

model.summary()

# No point in traiing all 50K images
# beats the purpose of semi-supervised learning
# reduce the 50 K to 5 K samples with original labels
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

split = 10
divided_input = np.split(x_train, split)
x_train_semisup = divided_input[0]

divided_output = np.split(y_train, split)
y_train_semisup = divided_output[0]
print (x_train_semisup.shape)
print (y_train_semisup.shape)
unique, counts = numpy.unique(y_train_semisup, return_counts=True)
print (dict(zip(unique, counts)))

x_train_semisup = x_train_semisup.astype('float32')
x_train_semisup /= 255
print(x_train_semisup.shape)



num_real_classes =10
# Convert class vectors to binary class matrices.
y_train_semisup = np_utils.to_categorical(y_train_semisup, num_real_classes)
y_test = np_utils.to_categorical(y_test, num_real_classes)
print(y_train_semisup.shape)
print(y_test.shape)

model.fit(x_train_semisup, y_train_semisup, batch_size=batch_size, nb_epoch=50,
          verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



