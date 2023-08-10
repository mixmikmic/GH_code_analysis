# Import
import keras
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Getting Dataset:
from get_dataset import get_dataset
X_train, X_test, Y_train, Y_test = get_dataset()

# About Dataset:

img_size = X_train.shape[1] # 64
channel_size = X_train.shape[3] # 1: Grayscale, 3: RGB

print('Training shape:', X_train.shape)
print(X_train.shape[0], 'sample,',X_train.shape[1] ,'x',X_train.shape[2] ,'size grayscale image.\n')
print('Test shape:', X_test.shape)
print(X_test.shape[0], 'sample,',X_test.shape[1] ,'x',X_test.shape[2] ,'size grayscale image.\n')

print('Examples:')
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # Display some data:
    ax = plt.subplot(1, n, i)
    plt.imshow(X_train[i].reshape(img_size, img_size))
    plt.gray()
    plt.axis('off')

# Deep Learning Model:
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Activation, Lambda, Flatten, concatenate, Reshape
from keras.models import Model

input_img = Input(shape=(img_size, img_size, channel_size))

layer_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
layer_1 = MaxPooling2D((2, 2))(layer_1)

layer_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(layer_1)
layer_2 = MaxPooling2D((2, 2))(layer_2)

layer_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(layer_2)
layer_3 = MaxPooling2D((2, 2))(layer_3)

flat_1 = Flatten()(layer_3)

fc_1 = Dense(256)(flat_1)
fc_1 = Activation('relu')(fc_1)

fc_2 = Dense(128)(fc_1)
fc_2 = Activation('relu')(fc_2)

#Decoder:

fc_3 = Dense(256)(fc_2)
fc_3 = Activation('relu')(fc_3)

fc_4 = Dense(16384)(fc_3)
fc_4 = Activation('relu')(fc_4)

reshape_1 = Reshape((8, 8, 256))(fc_4)

layer_4 = UpSampling2D((2, 2))(reshape_1)
layer_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(layer_4)

layer_5 = UpSampling2D((2, 2))(layer_4)
layer_5 = Conv2D(128, (3, 3), activation='relu', padding='same')(layer_5)

layer_6 = UpSampling2D((2, 2))(layer_5)
layer_6 = Conv2D(64, (3, 3), activation='relu', padding='same')(layer_6)

layer_7 = Conv2D(channel_size, (3, 3), activation='sigmoid', padding='same')(layer_6)

autoencoder = Model(input_img, layer_7)
autoencoder.compile(optimizer='rmsprop', loss='mse')

autoencoder.summary()

# Checkpoints:
from keras.callbacks import ModelCheckpoint, TensorBoard
checkpoints = []
#checkpoints.append(TensorBoard(log_dir='/Checkpoints/logs'))

# Getting saved mode:

autoencoder.load_weights('Data/Model/weights.h5')

decoded_imgs = autoencoder.predict(X_test[0:11])

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(X_test[i].reshape(64, 64))
    plt.gray()
    plt.axis('off')

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(64, 64))
    plt.gray()
    plt.axis('off')

# Split autoencoder:
encoder = Model(input_img, fc_2)
encoder.summary()

num_summary = 128

# Deep Learning Model:
from keras.layers import Input, Dense, Activation, Dropout
from keras.models import Model

sn_inputs = Input(shape=(2*num_summary,))

sn_fc_1 = Dense(512)(sn_inputs)
sn_fc_1 = Activation('relu')(sn_fc_1)

sn_drp_1 = Dropout(0.5)(sn_fc_1)

sn_fc_2 = Dense(256)(sn_drp_1)
sn_fc_2 = Activation('relu')(sn_fc_2)

sn_drp_2 = Dropout(0.5)(sn_fc_2)

sn_fc_3 = Dense(64)(sn_drp_2)
sn_fc_3 = Activation('relu')(sn_fc_3)

sn_drp_3 = Dropout(0.5)(sn_fc_3)

sn_fc_4 = Dense(1)(sn_drp_3)
sn_similarity_output = Activation('sigmoid')(sn_fc_4)

similarity_net = Model(sn_inputs, sn_similarity_output)
similarity_net.compile(optimizer='adadelta', loss='mse')

similarity_net.summary()

from keras.layers import Input, concatenate

encoder.trainable = False

dis_input_img = Input(shape=(img_size, img_size, channel_size))
dis_encoder_out = encoder(dis_input_img)

dis_input_img_2 = Input(shape=(img_size, img_size, channel_size))
dis_encoder_out_2 = encoder(dis_input_img_2)

dis_cont_1 = concatenate([dis_encoder_out, dis_encoder_out_2])

dis_output = similarity_net(dis_cont_1)
discriminator = Model([dis_input_img, dis_input_img_2], dis_output)
discriminator.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
discriminator.summary()

from os import listdir
from get_dataset import get_img

dataset_path = 'Data/Train_Data/'

data_samples = []

labels = listdir(dataset_path)
for label in range(0,10):
    datas_path = dataset_path+'/{0}'.format(label)
    img = get_img(datas_path+'/'+listdir(datas_path)[5])
    data_samples.append(img)
data_samples = 1 - np.array(data_samples).astype('float32')/255.
data_samples = data_samples.reshape(data_samples.shape[0], img_size, img_size, channel_size)

from keras.preprocessing.image import ImageDataGenerator

X_train_sets = []
X_train_sets_2 = []
Y_train_sets = []

datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2) # For include left hand data add: 'horizontal_flip = True'
datagen.fit(X_train)

for X_batch, X_batch_2 in datagen.flow(X_train, X_train, batch_size=X_train.shape[0], seed=599):
    for i in range(0, X_train.shape[0]):
        X_train_sets.append(X_batch[i])
        X_train_sets_2.append(X_batch_2[i])
        Y_train_sets.append(1)
    break

datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2) # For include left hand data add: 'horizontal_flip = True'
datagen.fit(data_samples)    

len_truths = len(Y_train_sets)
print('Length of truths:', len_truths)

for X_batch, X_batch_2 in datagen.flow(data_samples, data_samples, batch_size=10, seed=599):
    for i in range(0, 10):
        if i%2==0:
            X_train_sets.append(X_batch[i])
            X_train_sets_2.append(X_batch_2[i + 1 if i < 9 else 0])
        else:
            X_train_sets.append(X_batch_2[i + 1 if i < 9 else 0])
            X_train_sets_2.append(X_batch[i])
        Y_train_sets.append(0)
    if len(Y_train_sets)-len_truths >= len_truths:
        break

print('Length of faults:', len(Y_train_sets)-len_truths)

X_train_sets_np = np.array(X_train_sets)
X_train_sets_2_np = np.array(X_train_sets_2)
Y_train_sets_np = np.array(Y_train_sets)

X_train_sets = np.concatenate((X_train_sets_np, X_train_sets_2_np), axis=0)
X_train_sets_2 = np.concatenate((X_train_sets_2_np, X_train_sets_np), axis=0)
Y_train_sets = np.concatenate((Y_train_sets_np, Y_train_sets_np), axis=0)

print('Length of data:', Y_train_sets.shape[0])

# Getting saved mode:

discriminator.load_weights('Data/Model/weights_discriminator.h5')

index = 8
one_simple = X_test[index].reshape(1, img_size, img_size, channel_size)

plt.gray()
plt.imshow(one_simple.reshape(img_size, img_size))
plt.axis('off')
plt.show()

shift = 5
for i in [shift, -1*shift,]:
    for j in [1, 2]:
        noise_img = np.roll(one_simple, i, axis=j)

        plt.imshow(noise_img.reshape(img_size, img_size))
        plt.axis('off')
        plt.show()

        print(discriminator.predict([one_simple, noise_img])[0][0])

plt.gray()
index = 8
one_simple = X_test[index].reshape(1, img_size, img_size, channel_size)

plt.gray()
plt.imshow(one_simple.reshape(img_size, img_size))
plt.axis('off')
plt.show()

for i in [1,2]:
    for j in [1,-1]:
        noise_image = X_train[index + j*i].reshape(1, img_size, img_size, 1)

        plt.imshow(noise_image.reshape(img_size, img_size))
        plt.axis('off')
        plt.show()

        print(discriminator.predict([one_simple, noise_image])[0][0])

print(discriminator.predict([X_test[1].reshape(1,64,64,1), X_test[9].reshape(1,64,64,1)])[0][0])

plt.axis('off')
plt.imshow(X_test[1].reshape(64, 64))
plt.show()

plt.axis('off')
plt.imshow(X_test[9].reshape(64, 64))
plt.show()

from os import listdir
from get_dataset import get_img

dataset_path = 'Data/Train_Data/'

data_samples = []

labels = listdir(dataset_path)
for label in range(0,10):
    datas_path = dataset_path+'/{0}'.format(label)
    img = get_img(datas_path+'/'+listdir(datas_path)[5])
    data_samples.append(img)
data_samples = 1 - np.array(data_samples).astype('float32')/255.
data_samples = data_samples.reshape(data_samples.shape[0], img_size, img_size, channel_size)

for i, img in enumerate(data_samples):
    print('{0}:'.format(i))
    plt.gray()
    plt.imshow(img.reshape(img_size, img_size))
    plt.axis('off')
    plt.show()

from keras import backend as K

class_code = encoder.predict(data_samples)

encode = encoder.predict(X_test)

models_y_test = []
for i in encode:
    results = []
    for j in class_code:
        sim_y = similarity_net.predict(np.concatenate((i, j), axis=0).reshape(1, 256))
        results.append(sim_y[0][0])
    models_y_test.append(np.argmax(np.array(results).reshape(10), axis=0))
    
models_y_test = np.array(models_y_test)

num_Y_test = np.argmax(Y_test, axis=1)

comparison = models_y_test == num_Y_test
loss = 1 - np.sum(comparison.astype(int)) / num_Y_test.shape[0]

print('Loss:', loss)
print('Examples:')
for i in range(10,14):
    plt.imshow(X_test[i].reshape(64, 64))
    plt.gray()
    plt.axis('off')
    plt.show()
    print('Class:',  num_Y_test[i], '- Model\'s Output Class:', models_y_test[i],'\n'*2,'-'*40)

