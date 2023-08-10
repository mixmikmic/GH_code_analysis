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
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, UpSampling3D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
from keras.datasets import cifar10
from keras.utils import np_utils

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

def Generator():
    # bulid the generator model, it is a model made up of UpSample and Convolution
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=2048, init='normal'))
    model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(Dense(2048))
    #model.add(BatchNormalization())
    model.add(Activation('tanh'))
    #model.add(Reshape((2, 2, 512), input_shape=(2048,))) commented by me next line added by me
    model.add(Reshape((512, 2, 2), input_shape=(2048,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(256, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(128, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(3, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    print ('Generator model...')
    print (model.summary())
    return model


def Discriminator():
    model = Sequential()
    #model.add(Convolution2D(64, 5, 5, border_mode='same',subsample=(2, 2), input_shape=(32, 32, 3))) commented by me next line added by me
    model.add(Convolution2D(64, 5, 5, border_mode='same',subsample=(2, 2), input_shape=(3, 32, 32)))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    #model.add(Activation('tanh'))
    model.add(Convolution2D(128, 5, 5, border_mode='same', subsample=(2,2)))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    #model.add(Activation('tanh'))
    model.add(Convolution2D(256, 5, 5, border_mode='same', subsample=(2,2)))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    #model.add(Activation('tanh'))
    model.add(Convolution2D(512, 5, 5, border_mode='same', subsample=(4,4)))
    model.add(LeakyReLU(0.2))
    #model.add(Activation('tanh'))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    print ('Discriminator model...')

    print (model.summary())
    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model
# Note that you will have to change the output_shape depending on the backend used.

def combine_images(generated_images):
    num = generated_images.shape[0]
    #print ('num ',num)
    width = int(math.sqrt(num))
    #print ('width ',width)
    height = int(math.ceil(float(num)/width))
    #print ('height ',height)
    #print ('generated images shape before reshape ',generated_images.shape)
    ## added the below line to overcome the poor quality of image generated issue with RGB channel 
    generated_images = generated_images.reshape(generated_images.shape[0], 3, 32, 32).transpose(0,2,3,1)

    generated_images = generated_images.reshape((generated_images.shape[0], 32, 32, 3) )
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1], 3),dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1], 0] =             img[:, :, 0]
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1], 1] =             img[:, :, 1]
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1], 2] =             img[:, :, 2]
    return image


def train(BATCH_SIZE, epoch_num):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    #print X_train.dtype
    #X_train = (X_train.astype(np.float32) - 127.5)/127.5
    #print X_train.dtype
    print ('downloaded X_train shape ',X_train.shape)
    print ('downloaded X_test shape ', X_test.shape)
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    #X_train /= 255
    X_train = (X_train - 127.5)/127.5
    X_test /= 255
    X_train = X_train.reshape((X_train.shape[0], ) + X_train.shape[1:])
    #X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    print ('X_train shape ',X_train.shape)
    print ('X_test shape ', X_test.shape)
    discriminator = Discriminator()
    print ('Discriminator initialized...')
    generator = Generator()
    print ('Generator initialized...')

    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
    print ('generator_containing_discriminator initialized...')
    
    #d_optim = SGD(lr=0.0005, momentum=0.5, nesterov=True)
    #g_optim = SGD(lr=0.0005, momentum=0.5, nesterov=True)
    d_optim = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)
    g_optim = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)

    generator.compile(loss='binary_crossentropy', optimizer="Adam")
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    noise = np.zeros((BATCH_SIZE, 100))
    print ('noise shape ',noise.shape)
    #for epoch in range(100):

    for epoch in range(epoch_num):
    
        batches_num = int(X_train.shape[0]/BATCH_SIZE)
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        # load weights on first try (i.e. if process failed previously and we are attempting to recapture lost data)
        
        if epoch == 0:
            if os.path.exists('generator_cifar') and os.path.exists('discriminator_cifar'):
                print ("Loading saves weights..")
                generator.load_weights('generator_cifar')
                discriminator.load_weights('discriminator_cifar')
                print ("Finished loading")
            else:
                pass
        
        for index in range(batches_num):
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
                #print 'noise', noise.dtype
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(noise, [0.9] * BATCH_SIZE)
            
            #filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
            #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            #callbacks_list = [checkpoint]
            
            discriminator.trainable = True
            #print("epoch %d/%d batch %d/%d g_loss : %f" % (epoch+1, epoch_num,index, batches_num, g_loss))

            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = generator.predict(noise, verbose=0)
            
            if index % 64 == 0:
                gen_image = combine_images(generated_images)
                gen_image = gen_image.reshape(384, 352, 3)
                gen_image = gen_image*127.5+127.5
                Image.fromarray(gen_image.astype(np.uint8)).save('images/gen_image'+
                    str(epoch)+"_"+str(index)+".png")

                org_image = combine_images(image_batch)
                org_image = org_image.reshape(384, 352, 3)
                org_image = org_image*127.5+127.5
                Image.fromarray(org_image.astype(np.uint8)).save('images/org_image'+
                    str(epoch)+"_"+str(index)+".png")

                #print image_batch.shape, generated_images.shape

            X = np.concatenate((image_batch, generated_images))           
            y = [0.9] * BATCH_SIZE + [0.0] * BATCH_SIZE
            #y = np.array(y)
            #print 'y ', y.shape
            d_loss = discriminator.train_on_batch(X, y)
            #print("epoch %d/%d batch %d/%d d_loss : %f" % (epoch+1, epoch_num, index, batches_num, d_loss))
            #for i in range(BATCH_SIZE):
            #    noise[i, :] = np.random.uniform(-1, 1, 100)
            #discriminator.trainable = False
            '''g_loss = discriminator_on_generator.train_on_batch(
                noise, np.array([1.0] * BATCH_SIZE))
            discriminator.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            '''
            if index % 50 == 0:
                generator.save_weights('generator_cifar', True)
                discriminator.save_weights('discriminator_cifar', True)

'''
def generate(BATCH_SIZE, nice=False):
    generator = Generator()
    generator.compile(loss='binary_crossentropy', optimizer="Adam")
    generator.load_weights('generator_cifar')
    if nice:
        discriminator = Discriminator()
        discriminator.compile(loss='binary_crossentropy', optimizer="Adam")
        discriminator.load_weights('discriminator_cifar')
        noise = np.zeros((BATCH_SIZE*20, 100))
        for i in range(BATCH_SIZE*20):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        print generated_images.shape
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE, ) +
                           (generated_images.shape[1:3]) + (1,), dtype=np.float32)
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.zeros((BATCH_SIZE, 100))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        image = combine_images(generated_images)
    #image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")
'''

def generate(BATCH_SIZE):
    generator = Generator()
    d_optim = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)
    g_optim = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)
    generator.compile(loss='binary_crossentropy', optimizer=g_optim)
    generator.load_weights('generator_cifar')
    noise = np.zeros((BATCH_SIZE, 100))
    for i in range(BATCH_SIZE):
    	noise[i, :] = np.random.uniform(-1, 1, 100)
    generated_images = generator.predict(noise, verbose=1)
    print ('after generator predict generated images shape',generated_images.shape)
    image = combine_images(generated_images)

    image = image.reshape(384, 352, 3)

    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save("generated_image.png")

    #clr_img = Image.fromarray(image,'RGB')   
    #clr_img.save("generated_image_clr.png")
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.add_argument("--epoch_num",type=int,default=100)
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args



if not os.path.exists('images'):
    os.mkdir('images')


#Training the model - python cifar_gan.py --mode train --batch_size 128 --epoch_num 200
train(BATCH_SIZE=128, epoch_num=2)

generate(BATCH_SIZE=128)

#from __future__ import print_function
#import numpy as np
#import keras
#from keras.utils import to_categorical
#from keras.utils import np_utils
from keras.datasets import cifar10
##from keras.layers.core import Activation
#from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Convolution2D, MaxPooling2D
#from keras.layers.advanced_activations import LeakyReLU
import keras.regularizers






batch_size = 128
num_classes = 10
#epochs = 5

#mnist image dimensionality
img_rows = 32
img_cols = 32

from collections import Counter
#loading the mnist dataInit
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

divided_input = np.array_split(X_train, 50)
X_train = divided_input[0]


divided_output = np.array_split(Y_train, 50)
Y_train = divided_output[0]

unique, counts = numpy.unique(Y_train, return_counts=True)
print (dict(zip(unique, counts)))


divided_inputtest = np.array_split(X_test, 2)
X_test = divided_inputtest[0]
divided_outputtest = np.array_split(Y_test, 2)
Y_test = divided_outputtest[0]

unique, counts = numpy.unique(Y_test, return_counts=True)
print (dict(zip(unique, counts)))

#reshaping for input to network
#X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
#X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)

X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)

#input_shape = (img_rows, img_cols, 3)
input_shape = (3, img_rows, img_cols)

#making data float datatype
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#normalizing the data
X_train /= 255
X_test /= 255

print('x_train shape:', X_train.shape)
print('x_test shape:', X_test.shape)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#convert class vectors to one hot encoded vectors
Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)

print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)


feature_layers = [
    Convolution2D(64, 5, 5, border_mode='same',subsample=(2, 2), input_shape=(3,32,32)),
    LeakyReLU(0.2),
    Dropout(0.5),
    Convolution2D(128, 5, 5, border_mode='same', subsample=(2,2)),
    LeakyReLU(0.2),
    Dropout(0.5),
    Convolution2D(256, 5, 5, border_mode='same', subsample=(2,2)),
    LeakyReLU(0.2),
    Dropout(0.5),
    Convolution2D(512, 5, 5, border_mode='same', subsample=(4,4)),
    LeakyReLU(0.2),
    Dropout(0.5),
    Flatten()
]

classification_layers = [
    Dense(512, W_regularizer=keras.regularizers.l2(0.01), name='fc_layer1'),
    Activation('relu'),
    Dense(num_classes, activation='softmax', W_regularizer=keras.regularizers.l2(0.01), name='fc_layer2')
]
'''

classification_layers = [
    Dense(512, W_regularizer=keras.regularizers.l2(0.01), name='fc_layer1'),
    Activation('relu'),
    Dense(num_classes, activation='softmax', name='fc_layer2')
]
'''

model = Sequential(feature_layers + classification_layers)
# different backend has different image dim order, so we need to judge first.

'''
input_shape = (28,28,1)
model.add(Convolution2D(64, 5, 5, border_mode='same',subsample=(2, 2), input_shape=input_shape))
#model.add(LeakyReLU(0.02))
model.add(Activation('tanh'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, 5, 5, border_mode='same', subsample=(2,2)))
#model.add(LeakyReLU(0.02))
#model.add(BatchNormalization())
model.add(Activation('tanh'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024))
#model.add(LeakyReLU(0.02))
#model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dense(num_classes, activation='softmax'))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))
'''
#print model.summary()

model.load_weights('discriminator_cifar', by_name=True)

for l in feature_layers:
    l.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

print('Model Compilation successful')




model.summary()

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=50,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



