import numpy as np
import matplotlib.pyplot as plt
import svhn
import graphics
import keras_utils
from keras.utils import np_utils
import preprocessing
from sklearn.cross_validation import train_test_split

get_ipython().magic('matplotlib inline')

# define some constants
max_digits = 7
image_size = (54,128)
checkpoint_path = '../checkpoints/model.hdf5'
resume_training = True

# print the keras version used
import keras
print "Keras version : {}".format(keras.__version__)

# read the h5py data file (takes time)
rawdata = svhn.read_process_h5('../inputs/train/digitStruct.mat')

nsamples = 33402

def generateData(data, n=1000):
    '''
    generates flattened SVHN dataset
    '''
    Ximg_flat = []
    Xidx_flat = []
    ycount_flat = []
    ycoord_flat = []
    ylabel_flat = []
    
    for datapoint in np.random.choice(data, size=n, replace=False):
        img,_ = svhn.createImageData(datapoint, image_size, '../inputs/train/')
        for i in range(0,datapoint['length']):
            Ximg_flat.append(img)
            Xidx_flat.append(i)
            ycount_flat.append(datapoint['length'])
            ylabel_flat.append(datapoint['labels'][i])
            
    ylabel_flat = [0 if y==10 else int(y) for y in ylabel_flat]
    return np.array(Ximg_flat), np.array(Xidx_flat), np.array(ycount_flat), np.array(ylabel_flat)

Ximg, Xidx, ycount, ylabel = generateData(rawdata, nsamples)
Xidx = np_utils.to_categorical(Xidx, max_digits)
ycount = np_utils.to_categorical(ycount, max_digits)
ylabel = np_utils.to_categorical(ylabel, 10)

Ximg, Ximg_val, Xidx, Xidx_val, ycount, ycount_val, ylabel, ylabel_val = train_test_split(Ximg, Xidx, ycount, ylabel, test_size=0.05)

def standardize(img):
    s = img - np.mean(img, axis=(2,0,1), keepdims=True)
    s /= (np.std(s, axis=(2,0,1), keepdims=True) + 1e-7)
    return s
Xs_val = np.array([standardize(x) for x in Ximg_val])

# use custom version of keras image augmentation system
# also normalize the input image
datagen = preprocessing.SVHNImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.15,
    shear_range=0.4,
    zoom_range=[1.0,1.6],
    samplewise_center=True,
    samplewise_std_normalization=True
    )

datagen.fit(Ximg)

from keras.layers import Input, Dense, Merge, Flatten, Dropout, merge
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization

# define vision model
image_in_vision = Input(shape=(image_size[0],image_size[1],3))
x = BatchNormalization(axis=3)(image_in_vision)
x = Convolution2D(32, 3, 3, activation='tanh')(x)
x = BatchNormalization(axis=3)(x)
x = Convolution2D(32, 3, 3, activation='relu')(x)
x = BatchNormalization(axis=3)(x)
x = Dropout(0.2)(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Convolution2D(64, 3, 3, activation='relu')(x)
x = BatchNormalization(axis=3)(x)
x = Convolution2D(64, 3, 3, activation='relu')(x)
x = BatchNormalization(axis=3)(x)
x = Dropout(0.2)(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Convolution2D(128, 3, 3, activation='relu')(x)
x = BatchNormalization(axis=3)(x)
x = Convolution2D(128, 3, 3, activation='relu')(x)
x = BatchNormalization(axis=3)(x)
x = Dropout(0.2)(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(1024, activation='relu')(x)
h = BatchNormalization()(x)
vision_model = Model(input=image_in_vision, output=h, name='vision')

# define counter model
h_in_counter = Input(shape=(1024,))
yc = Dense(256, activation='relu')(h_in_counter)
yc = BatchNormalization()(yc)
yc = Dropout(0.2)(yc)
yc = Dense(max_digits, activation='softmax')(yc)
counter_model = Model(input=h_in_counter, output=yc, name='counter')

# define detector model
h_in_detector = Input(shape=(1024,))
idx_in_detector = Input(shape=(max_digits,))
yl = merge([h_in_detector, idx_in_detector], mode='concat') 
yl = Dense(512, activation='relu')(yl)
yl = BatchNormalization()(yl)
yl = Dense(512, activation='relu')(yl)
yl = BatchNormalization()(yl)
yl = Dropout(0.2)(yl)
yl = Dense(10, activation='softmax')(yl)

detector_model = Model(input=[h_in_detector, idx_in_detector], output=yl, name='detector')

# Combine the three models to construct training graph
# We're defining the training graph as a composite made of macro models. 
# This will enable us to easily retrieve these macro components and restructure them during inference.
Ximg_in = Input(shape=(image_size[0], image_size[1], 3), name='train_input_img')
Xidx_in = Input(shape=(max_digits,), name='train_input_idx')
h = vision_model(Ximg_in)
yc = counter_model(h)
yl = detector_model([h, Xidx_in])

train_graph = Model(input=[Ximg_in, Xidx_in], output=[yc, yl])
train_graph.compile(optimizer='adamax', loss=['categorical_crossentropy','categorical_crossentropy'], metrics=['accuracy'])

# save the entire training graph
with open('../checkpoints/model.yaml','w') as model_def:
    model_def.write(train_graph.to_yaml())

# resume training if instructed and weights file exists
import os.path
if resume_training and os.path.isfile(checkpoint_path):
    train_graph.load_weights(checkpoint_path)

# define checkpoint callback
checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', mode='min', save_best_only=True)
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')

history = train_graph.fit_generator(datagen.flow(Ximg, Xidx, ycount, ylabel, batch_size=64),
                          nb_epoch=50, samples_per_epoch=len(Xidx),
                          validation_data=([Xs_val, Xidx_val],[ycount_val, ylabel_val]),
                          callbacks=[checkpoint, earlystop])

plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
ctl_plot = plt.plot(history.history['counter_loss'], 'r', label='training')
cvl_plot = plt.plot(history.history['val_counter_loss'], 'g', label='validation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('counter loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['counter_acc'], 'r', label='training')
plt.plot(history.history['val_counter_acc'], 'g', label='validation')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('counter accuracy')
plt.legend(loc=4)

plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.plot(history.history['detector_loss'], 'r', label='training')
plt.plot(history.history['val_detector_loss'], 'g', label='validation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('detector loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['detector_acc'], 'r', label='training')
plt.plot(history.history['val_detector_acc'], 'g', label='validation')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('detector accuracy')
plt.legend(loc=4)



