from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

import keras
from six.moves import cPickle as pickle

import random

from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
get_ipython().magic('matplotlib inline')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

train_dataset = mnist.train.images - 0.5
train_labels = mnist.train.labels
valid_dataset = mnist.validation.images -0.5
valid_labels = mnist.validation.labels
test_dataset = mnist.test.images - 0.5
test_labels = mnist.test.labels

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def show_imagelist_as_grid(img_list, nrow, ncol):
    fig = plt.figure(figsize=(5,5))
    grid = AxesGrid(fig, 111, nrows_ncols=(nrow, ncol), axes_pad=0.05, label_mode="1")
    for i in range(nrow*ncol):
        im = grid[i].imshow(img_list[i], interpolation="none", cmap='gray', vmin=-0.5, vmax=0.5)
    plt.draw()
    plt.show()
    

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

dataset_mean = np.mean(train_dataset)
dataset_std = np.std(train_dataset)
print("mean and std: ", dataset_mean, dataset_std)
#show_imagelist_as_grid(train_dataset[:16].reshape(-1,image_size,image_size),4,4)

# parameters
batch_size = 24
num_steps = 2000
dummy_size = 512
stddev=0.05
dropout_prob_gen = 0.6
dropout_prob_discr = 0.6
lrelu_alpha = 0.3

num_discr_layer1 = dummy_size//2
num_discr_layer2 = dummy_size//2
num_discr_layer3 = dummy_size//2

# generator: 3 hidden layers
num_gen_input_size = dummy_size
num_gen_layer1 = dummy_size*2
num_gen_layer2 = dummy_size*2
num_gen_layer3 = dummy_size*2

def gen_model():
    model = Sequential()
    model.add(Dense(num_gen_layer1, input_shape=(num_gen_input_size,), init='normal'))
    model.add(Dropout(dropout_prob_gen))
    model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(Dense(num_gen_layer2, init='normal'))
    model.add(Dropout(dropout_prob_gen))
    model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(Dense(num_gen_layer3, init='normal'))
    model.add(Dropout(dropout_prob_gen))
    model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(Dense(image_size*image_size, init='normal'))
    model.add(Activation('tanh'))
    return model

def discr_model():
    model = Sequential()
    model.add(Dense(num_discr_layer1, input_shape=(image_size*image_size,), init='normal'))
    model.add(Dropout(dropout_prob_discr))
    model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(Dense(num_discr_layer2, init='normal'))
    model.add(Dropout(dropout_prob_discr))
    model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(Dense(num_discr_layer3, init='normal'))
    model.add(Dropout(dropout_prob_discr))
    model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(Dense(1, init='normal'))
    model.add(Activation('sigmoid'))
    return model

def discr_on_gen_model(gen, discr):
    model = Sequential()
    model.add(gen)
    discr.trainable = False
    model.add(discr)
    return model
    
discriminator = discr_model();
generator = gen_model();
discriminator_on_generator = discr_on_gen_model(generator, discriminator);

discr_optimizer = SGD(lr=0.005, momentum=0.9, nesterov=True)
gen_optimizer = SGD(lr=0.005, momentum=0.9, nesterov=True)

generator.compile(loss='binary_crossentropy', optimizer="SGD")
discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=gen_optimizer)
discriminator.trainable = True # we had made it untrainable when we added to the discriminator_on_generator
discriminator.compile(loss='binary_crossentropy', optimizer=discr_optimizer)

# training
print_step = 100

gen_loss_total = 0.0
gen_trained = 0
discr_loss_total = 0.0
discr_trained = 0

batch_1s = np.zeros(batch_size)+1
batch_0s = np.zeros(batch_size)


print("Starting training.")
for step in range(num_steps):

    #train discriminator
    offset = (round(random.uniform(0, 100000)) + step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_noise = np.random.normal(loc=dataset_mean, scale=dataset_std, size=(batch_size, num_gen_input_size))
    generated_images = generator.predict(batch_noise, verbose=0)
    # batch_X = np.concatenate((batch_data, generated_images))
    # batch_Y = np.concatenate((np.zeros(batch_size)+1, np.zeros(batch_size)))
    discriminator.trainable = True # just to make sure it's still trainable
    ld1 = discriminator.train_on_batch(batch_data, batch_1s)
    ld2 = discriminator.train_on_batch(generated_images, batch_0s)
    discr_loss_total += (ld1+ld2)/2
    discr_trained += 1
    
    #train generator
    batch_noise = np.random.normal(loc=dataset_mean, scale=dataset_std, size=(batch_size, num_gen_input_size))    
    discriminator.trainable = False # make sure we train only the generator
    lg = discriminator_on_generator.train_on_batch(batch_noise, np.zeros(batch_size) + 1)
    discriminator.trainable = True # just to make sure it's still trainable
    gen_loss_total += lg
    gen_trained += 1
        
    if (step % print_step == print_step-1):
        if discr_trained == 0:
            discr_trained = 1
        print('Minibatch loss before step %d: discriminator %f, generator: %f' % (step+1, discr_loss_total/discr_trained, gen_loss_total/gen_trained))
        gen_loss_total = 0.0
        discr_loss_total = 0.0
        gen_trained = 0
        discr_trained = 0

batch_noise = np.random.normal(loc=dataset_mean, scale=dataset_std, size=(batch_size, num_gen_input_size))    
generated_images = generator.predict(batch_noise, verbose=0)
show_imagelist_as_grid(generated_images.reshape(-1,image_size,image_size), 4,4)

