# Arda Mavi
# Unsupervised Classification With Autoencoder

# Import
import keras
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Getting Dataset:
from get_dataset import get_dataset
X_train, X_test, Y_train, Y_test = get_dataset()

# About Dataset:
img_size = X_train.shape[1] # 64
print('Training shape:', X_train.shape)
print(X_train.shape[0], 'sample,',X_train.shape[1] ,'x',X_train.shape[2] ,'size RGB image.\n')
print('Test shape:', X_test.shape)
print(X_test.shape[0], 'sample,',X_test.shape[1] ,'x',X_test.shape[2] ,'size RGB image.\n')

print('Examples:')
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # Display some data:
    ax = plt.subplot(1, n, i)
    plt.imshow(X_train[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# Deep Learning Model:
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense
from keras.models import Model

input_img = Input(shape=(64, 64, 3))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='rmsprop', loss='mse')

autoencoder.summary()

# Checkpoints:
from keras.callbacks import ModelCheckpoint, TensorBoard
checkpoints = []
#checkpoints.append(TensorBoard(log_dir='/Checkpoints/logs'))

# Creates live data:
# For better yield. The duration of the training is extended.

from keras.preprocessing.image import ImageDataGenerator
generated_data = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, rotation_range=0,  width_shift_range=0.1, height_shift_range=0.1, horizontal_flip = True, vertical_flip = False)
generated_data.fit(X_train)

model.fit_generator(generated_data.flow(X_train, X_train, batch_size=batch_size), steps_per_epoch=X.shape[0], epochs=epochs, validation_data=(X_test, X_test), callbacks=checkpoints)

# Training Model:
epochs = 4
batch_size = 1
autoencoder.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, X_test), shuffle=True, callbacks=checkpoints)

decoded_imgs = autoencoder.predict(X_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # Display original:
    ax = plt.subplot(2, n, i)
    plt.imshow(X_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction:
    ax = plt.subplot(2, n, i+n)
    plt.imshow(decoded_imgs[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# Describe the number of classes:
num_class = 2

from keras import backend as K

# Custom classifier function:
def classifier_func(x):
    return x-x+K.one_hot(K.argmax(x, axis=1), num_classes=num_class)

# Deep Learning Model:
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Activation, Lambda, Flatten, concatenate, Reshape
from keras.models import Model

input_img = Input(shape=(64, 64, 3))

layer_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
layer_1 = MaxPooling2D((2, 2))(layer_1)

layer_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(layer_1)
layer_2 = MaxPooling2D((2, 2))(layer_2)

layer_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(layer_2)
layer_3 = MaxPooling2D((2, 2))(layer_3)

flat_1 = Flatten()(layer_3)

fc_1 = Dense(256)(flat_1)
fc_1 = Activation('relu')(fc_1)

fc_2 = Dense(128)(fc_1)
fc_2 = Activation('relu')(fc_2)

fc_3 = Dense(num_class)(fc_2)
act_class = Lambda(classifier_func, output_shape=(num_class,))(fc_3)

# Merge Layers:

merge_1 = concatenate([act_class, fc_2])

#Decoder:
fc_4 = Dense(256)(merge_1)
fc_4 = Activation('relu')(fc_4)

fc_5 = Dense(4096)(fc_4)
fc_5 = Activation('relu')(fc_5)

reshape_1 = Reshape((8, 8, 64))(fc_5)

layer_4 = UpSampling2D((2, 2))(reshape_1)
layer_4 = Conv2D(64, (3, 3), activation='relu', padding='same')(layer_4)

layer_5 = UpSampling2D((2, 2))(layer_4)
layer_5 = Conv2D(64, (3, 3), activation='relu', padding='same')(layer_5)

layer_6 = UpSampling2D((2, 2))(layer_5)
layer_6 = Conv2D(32, (3, 3), activation='relu', padding='same')(layer_6)

layer_7 = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(layer_6)

autoencoder = Model(input_img, layer_7)
autoencoder.compile(optimizer='rmsprop', loss='mse')

autoencoder.summary()

# Creates live data:
# For better yield. The duration of the training is extended.

from keras.preprocessing.image import ImageDataGenerator
generated_data = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, rotation_range=0,  width_shift_range=0.1, height_shift_range=0.1, horizontal_flip = True, vertical_flip = False)
generated_data.fit(X_train)

model.fit_generator(generated_data.flow(X_train, X_train, batch_size=batch_size), steps_per_epoch=X.shape[0], epochs=epochs, validation_data=(X_test, X_test), callbacks=checkpoints)

# Training Model:
epochs = 4
batch_size = 5
autoencoder.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, X_test), shuffle=True, callbacks=checkpoints)

decoded_imgs = autoencoder.predict(X_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(X_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# Split autoencoder:
encoder = Model(input_img, act_class)
encoder.summary()

encode = encoder.predict(X_train)

class_dict = np.zeros((num_class, num_class))
for i, sample in enumerate(Y_train):
    class_dict[np.argmax(encode[i], axis=0)][np.argmax(sample)] += 1
    
print(class_dict)
    
neuron_class = np.zeros((num_class))
for i in range(num_class):
    neuron_class[i] = np.argmax(class_dict[i], axis=0)

print(neuron_class)

# Getting class as string:
def cat_dog(model_output):
    if model_output == 0:
        return "Cat"
    else:
        return "Dog"

encode = encoder.predict(X_test)

predicted = np.argmax(encode, axis=1)
for i, sample in enumerate(predicted):
    predicted[i] = neuron_class[predicted[i]]

comparison = predicted == np.argmax(Y_test, axis=1)
loss = 1 - np.sum(comparison.astype(int))/Y_test.shape[0]

print('Loss:', loss)
print('Examples:')
for i in range(10):
    plt.imshow(X_test[i])
    plt.axis('off')
    plt.show()
    neuron = np.argmax(encode[i], axis=0)
    print('Class:',  cat_dog(np.argmax(Y_test[i], axis=0)), '- Model\'s Output Class:', cat_dog(neuron_class[neuron]),'\n'*2,'-'*40)

