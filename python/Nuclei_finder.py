'''
NUCLEI FINDER

Initial setup

Ref:
https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277

'''

import random
import numpy as np
import matplotlib.pyplot as plt

# Image parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

seed = 64

# Get train and test IDs
import os

train_ids = next(os.walk('stage1_train/'))[1]
test_ids = next(os.walk('stage1_test/'))[1]

print('Number of training ids: {}'.format(len(train_ids)))
print('Number of test ids: {}'.format(len(test_ids)))

'''
Get the data
'''

import sys
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize

from skimage.io import imshow

import tensorflow as tf

from keras import backend as K

from keras.models import load_model

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = 'stage1_train/' + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = 'stage1_test/' + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')


# Check if training data looks all right

ix = random.randint(0, len(train_ids))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()


# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

'''
THIS IS THE BENCHMARK MODEL
'''


#Build model


from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate


'''
From kernel mentioned above

Loosely based on U-Net: Convolutional Networks for Biomedical Image Segmentation 
https://arxiv.org/pdf/1505.04597.pdf

and very similar to this repo 
https://github.com/jocicmarko/ultrasound-nerve-segmentation
from the Kaggle Ultrasound Nerve Segmentation competition.
'''


# Build U-Net model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()

from keras.callbacks import EarlyStopping, ModelCheckpoint


# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50, 
                    callbacks=[earlystopper, checkpointer])


# Predict on train, val and test
model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))
    


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()


# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()

'''
Encode results for submission
'''

import pandas as pd
from skimage.morphology import label

# Run-length encoding 
# from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
        
'''
Iterate over the test IDs and generate run-length encodings for each seperate mask identified by skimage
'''
        
new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))
    

'''
Create submission DataFrame
'''

sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('nuclei_finder_bench1_deraso.csv', index=False)

'''
RefineNet inspired architecture
''' 
    
#Build model


from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import add


inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

# The contracting path is similar to the one in the benchmark model, one level shorter
# in order to have 3 RefinNets modules in total:
c1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.3) (c4)
c4 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)

# Multi-path Refinement:

# RefineNet 1:
# Residual Conv Unit (Only using 1 Conv layer instead of 2)
rcu5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
rcu5 = Dropout(0.3) (rcu5)
rcu5 = add([c4, rcu5])
# In the bottom RefineNet module the multi-resolution fusion only has one input:
f5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (rcu5)
f5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (f5)
# Chained Residual Pooling:
p5_1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same') (f5)
p5_1 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p5_1)
p5_1 = Dropout(0.3) (p5_1)
c5 = add([f5, p5_1])
p5_2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same') (p5_1)
p5_2 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p5_2)
c5 = add([c5, p5_2])

# RefineNet 2:
# Residual Conv Unit
rcu6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)
rcu6 = Dropout(0.2) (rcu6)
rcu6 = add([c5, rcu6])
# Multi-resolution fusion, `im` refers to input map from corrsponding contracting path level:
f6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (rcu6)
f6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (f6)
im6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
im6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (im6)
mrf6 = add([im6, f6])
# Chained Residual Pooling:
p6_1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same') (mrf6)
p6_1 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p6_1)
p6_1 = Dropout(0.2) (p6_1)
c6 = add([mrf6, p6_1])
p6_2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same') (p6_1)
p6_2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p6_2)
c6 = add([c6, p6_2])

# RefineNet 3:
# Residual Conv Unit
rcu7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)
rcu7 = Dropout(0.2) (rcu7)
rcu7 = add([c6, rcu7])
# Multi-resolution fusion
f7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (rcu7)
f7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (f7)
im7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
im7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (im7)
mrf7 = add([im7, f7])
# Chained Residual Pooling:
p7_1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same') (mrf7)
p7_1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p7_1)
p7_1 = Dropout(0.2) (p7_1)
c7 = add([mrf7, p7_1])
p7_2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same') (p7_1)
p7_2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p7_2)
c7 = add([c7, p7_2])

# Output
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)
c8 = Dropout(0.2) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)
outputs = Conv2D(1, (1, 1), activation='sigmoid') (c8)

model2 = Model(inputs=[inputs], outputs=[outputs])
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model2.summary()

from keras.callbacks import EarlyStopping, ModelCheckpoint

# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-2.h5', verbose=1, save_best_only=True)
results2 = model2.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50, 
                     callbacks=[earlystopper, checkpointer])

# Predict on train, val and test
model2 = load_model('model-dsbowl2018-2.h5', custom_objects={'mean_iou': mean_iou})
preds2_train = model2.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds2_val = model2.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds2_test = model2.predict(X_test, verbose=1)

# Threshold predictions
preds2_train_t = (preds2_train > 0.5).astype(np.uint8)
preds2_val_t = (preds2_val > 0.5).astype(np.uint8)
preds2_test_t = (preds2_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds2_test_upsampled = []
for i in range(len(preds2_test)):
    preds2_test_upsampled.append(resize(np.squeeze(preds2_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))

# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds2_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds2_train_t[ix]))
plt.show()

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds2_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds2_val_t[ix]))
plt.show()

'''
Encode results for submission
'''

import pandas as pd
from skimage.morphology import label

# Run-length encoding 
# from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
        
'''
Iterate over the test IDs and generate run-length encodings for each seperate mask identified by skimage
'''
        
new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds2_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))
    

'''
Create submission DataFrame
'''

sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('nuclei_finder_RefineNet_deraso.csv', index=False)

'''
PSPNet inspired architecture
'''

# 

'''
For this model we'll use transfer learning.

Refs: 

https://towardsdatascience.com/transfer-learning-using-keras-d804b2e04ef8 
https://keras.io/applications/

Models with corresponding pre-trained weights available in Keras Applications are pre-trained on ImageNet.
Since we have a small dataset very different from the original dataset, in order to make use of Transfer Learning
we would like ideally to only use the features extracted only on the earliest layers. For this we have to make
use of the h5 file of the pre-trained weights of the model we will use as base.

The PSPNet module can the be added on top of the earliest layers of the base model  

'''

from keras.applications.vgg19 import VGG19

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

'''
h5 weight file for VGG19:

https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5
(548 MB)
'''
import h5py

weights_path = 'vgg19_weights.h5'
f = h5py.File(weights_path)

# Let's check the layers 
print("\n".join(f.keys()))


'''
Since we only need the first layers, let's build a base model up to `block2_pool` where we can store the weights
preserving the original layer names.

In order for the base model to fit the VGG19 model we use for transfer learning let's first check this model. 
'''

VGG19model = VGG19(weights = "imagenet", include_top=False, input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

VGG19model.summary()

main_input = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name = 'main_input')

# Block 1
b1 = Conv2D(64, (3, 3), activation='elu', padding='same', name='block1_conv1')(main_input)
b1 = Conv2D(64, (3, 3), activation='elu', padding='same', name='block1_conv2')(b1)
p1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(b1)

# Block 2
b2 = Conv2D(128, (3, 3), activation='elu', padding='same', name='block2_conv1')(p1)
b2 = Conv2D(128, (3, 3), activation='elu', padding='same', name='block2_conv2')(b2)
p2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(b2)

base_model = Model(inputs = main_input, outputs = p2)

'''
Set weights on base model
'''
# Layer dictionary
layer_dict = dict([(layer.name, layer) for layer in base_model.layers])

# List of layer names in base model:
layer_names = [layer.name for layer in base_model.layers]

for i in layer_dict.keys():
    if i != 'main_input':
        weight_names = f[i].attrs["weight_names"]
        weights = [f[i][j] for j in weight_names]
        index = layer_names.index(i)
        base_model.layers[index].set_weights(weights)

'''
Now let's build the PSPNet Module that goes on top of the base model.
'''

# The output for the base model has dimensions (None, 32, 32, 128)
# Now let's build the expanding path to get the feature map


transfer = base_model.output 
expand1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (transfer)
expand1 = concatenate([expand1, b2])
c4 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (expand1)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)

expand2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c4)
expand2 = concatenate([expand2, b1])
c5 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (expand2)
c5 = Dropout(0.2) (c5)
c5 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)


# PSPNet module
p6 = MaxPooling2D((2, 2), strides=(2, 2)) (c5)
c6_1 = Conv2D(64, (1, 1), activation='elu', padding='same') (p6)
c6_2 = Conv2D(64, (2, 2), activation='elu', padding='same') (p6)
c6_3 = Conv2D(64, (3, 3), activation='elu', padding='same') (p6)
c6_4 = Conv2D(64, (6, 6), activation='elu', padding='same') (p6)
c6_1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6_1)
c6_2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6_2)
c6_3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6_3)
c6_4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6_4)

c6 = concatenate([c5, c6_4, c6_3, c6_2, c6_1])

# final Conv output layers
co = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)
co = Dropout(0.2) (co)
co = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (co)
co = Dropout(0.2) (co)
final_output = Conv2D(1, (1, 1), activation='sigmoid') (co)

model3 = Model(inputs=[main_input], outputs=[final_output])

# For training freeze layers where VGG19 weights are loaded
for layer in base_model.layers:
    layer.trainable = False

model3.summary()


model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

from keras.callbacks import EarlyStopping, ModelCheckpoint

# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-3.h5', verbose=1, save_best_only=True)
results3 = model3.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50, 
                 callbacks=[earlystopper, checkpointer])

# Predict on train, val and test
from keras.models import load_model

model3 = load_model('model-dsbowl2018-3.h5', custom_objects={'mean_iou': mean_iou})
preds3_train = model3.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds3_val = model3.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds3_test = model3.predict(X_test, verbose=1)

# Threshold predictions
preds3_train_t = (preds3_train > 0.5).astype(np.uint8)
preds3_val_t = (preds3_val > 0.5).astype(np.uint8)
preds3_test_t = (preds3_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds3_test_upsampled = []
for i in range(len(preds3_test)):
    preds3_test_upsampled.append(resize(np.squeeze(preds3_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))

# Check on some random training sample

ix = random.randint(0, len(preds3_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds3_train_t[ix]))
plt.show()

# Check on some random validation sample
ix = random.randint(0, len(preds3_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds3_val_t[ix]))
plt.show()

'''
Encode results for submission
'''

import pandas as pd
from skimage.morphology import label

# Run-length encoding 
# from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
        
'''
Iterate over the test IDs and generate run-length encodings for each seperate mask identified by skimage
'''
        
new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds3_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))
    

'''
Create submission DataFrame
'''

sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('nuclei_finder_PSPNet_deraso.csv', index=False)

# From my ML_DL_notes

# break training set into training and validation sets
(x_train, x_valid) = X_train[67:], X_train[:67]
(y_train, y_valid) = Y_train[67:], Y_train[:67]

# print number of training, and validation images
print x_train.shape[0], 'train samples'
print x_valid.shape[0], 'validation samples'

from keras.preprocessing.image import ImageDataGenerator

datagen_args = dict(rotation_range=90, 
                    width_shift_range=0.5, 
                    height_shift_range=0.5, 
                    fill_mode='reflect', 
                    horizontal_flip=True, 
                    vertical_flip=True)

xt_datagen = ImageDataGenerator(**datagen_args)
yt_datagen = ImageDataGenerator(**datagen_args)

xv_datagen = ImageDataGenerator(**datagen_args)
yv_datagen = ImageDataGenerator(**datagen_args)

xt_datagen.fit(x_train, seed=seed)
yt_datagen.fit(y_train, seed=seed)

xv_datagen.fit(x_valid, seed=seed)
yv_datagen.fit(y_valid, seed=seed)

'''
Visualize original and augmented images
'''

# take subsets 
x_train_subset = x_train[:6]
y_train_subset = y_train[:6]

x_valid_subset = x_valid[:3]
y_valid_subset = y_valid[:3]

# visualize subset of training data
fig = plt.figure(figsize=(20,2))
for i in range(0, len(x_train_subset)):
    ax = fig.add_subplot(1, 6, i+1)
    ax.imshow(x_train_subset[i])
fig.suptitle('Subset of Original Training Images', fontsize=16)
plt.show()

fig = plt.figure(figsize=(20,2))
for i in range(0, len(y_train_subset)):
    ax = fig.add_subplot(1, 6, i+1)
    ax.imshow(np.squeeze(y_train_subset[i]))
fig.suptitle('Subset of Original Training Masks', fontsize=16)
plt.show()

# visualize augmented training images
fig = plt.figure(figsize=(20,2))
for x_batch in xt_datagen.flow(x_train_subset, batch_size=6, seed=seed):
    for i in range(0, 6):
        ax = fig.add_subplot(1, 6, i+1)
        ax.imshow(x_batch[i].astype('uint8'))
    fig.suptitle('Augmented Training Images', fontsize=16)
    plt.show()
    break;

fig = plt.figure(figsize=(20,2))
for y_batch in yt_datagen.flow(y_train_subset, batch_size=6, seed=seed):
    for i in range(0, 6):
        ax = fig.add_subplot(1, 6, i+1)
        ax.imshow(np.squeeze(y_batch[i]))
    fig.suptitle('Augmented Training Masks', fontsize=16)
    plt.show()
    break;

# visualize subset of validation data
fig = plt.figure(figsize=(20,2))
for i in range(0, len(x_valid_subset)):
    ax = fig.add_subplot(1, 3, i+1)
    ax.imshow(x_valid_subset[i])
fig.suptitle('Subset of Original Validation Images', fontsize=16)
plt.show()

fig = plt.figure(figsize=(20,2))
for i in range(0, len(y_valid_subset)):
    ax = fig.add_subplot(1, 3, i+1)
    ax.imshow(np.squeeze(y_valid_subset[i]))
fig.suptitle('Subset of Original Validation Masks', fontsize=16)
plt.show()

# visualize augmented images
fig = plt.figure(figsize=(20,2))
for x_batch in xv_datagen.flow(x_valid_subset, batch_size=3, seed=seed):
    for i in range(0, 3):
        ax = fig.add_subplot(1, 3, i+1)
        ax.imshow(x_batch[i].astype('uint8'))
    fig.suptitle('Augmented Validation Images', fontsize=16)
    plt.show()
    break;

fig = plt.figure(figsize=(20,2))
for y_batch in yv_datagen.flow(y_valid_subset, batch_size=3, seed=seed):
    for i in range(0, 3):
        ax = fig.add_subplot(1, 3, i+1)
        ax.imshow(np.squeeze(y_batch[i]))
    fig.suptitle('Augmented Validation Masks', fontsize=16)
    plt.show()
    break;

# Create train images and masks lists to avoid passing in `fit_generator` masks as labels. Use `fit` instead.

# Ref: 
# https://github.com/keras-team/keras/issues/3059
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# iterations over original dataset:
overall_i = 3

xtl_aug = []
ytl_aug = []

ct1 = 0
for batch in xt_datagen.flow(x_train, batch_size=1, seed=seed): 
    ct1 += 1
    xtl_aug.append(batch.astype('uint8'))
    if ct1 == (x_train.shape[0])*overall_i: 
        break;

ct2 = 0
for batch in yt_datagen.flow(y_train, batch_size=1, seed=seed): 
    ct2 += 1
    ytl_aug.append(batch.astype('bool'))
    if ct2 == (y_train.shape[0])*overall_i:
        break;

xvl_aug = []
yvl_aug = []

cv1 = 0
for batch in xv_datagen.flow(x_valid, batch_size=1, seed=seed):
    cv1 += 1
    xvl_aug.append(batch.astype('uint8'))
    if cv1 == (x_valid.shape[0])*overall_i:
        break;

cv2 = 0
for batch in yv_datagen.flow(y_valid, batch_size=1, seed=seed): 
    cv2 += 1
    yvl_aug.append(batch.astype('bool'))
    if cv2 == (y_valid.shape[0])*overall_i:
        break;

# Shape of nested list for xtl_aug is: (1812, 1, 128, 128, 3), must be (1812, 128, 128, 3). Same for the rest.

xt_aug = [im3d for xdim in xtl_aug for im3d in xdim]
yt_aug = [im3d for xdim in ytl_aug for im3d in xdim]
xv_aug = [im3d for xdim in xvl_aug for im3d in xdim]
yv_aug = [im3d for xdim in yvl_aug for im3d in xdim]


print('Augmented training images dimensions: {}, {}, {}, {}'.format(len(xt_aug), 
                                                                    len(xt_aug[0]), 
                                                                    len(xt_aug[0][0]),
                                                                    len(xt_aug[0][0][0])))
print('Augmented training masks dimensions: {}, {}, {}, {}'.format(len(yt_aug), 
                                                                   len(yt_aug[0]), 
                                                                   len(yt_aug[0][0]),
                                                                   len(yt_aug[0][0][0])))
print('Augmented validation images dimensions: {}, {}, {}, {}'.format(len(xv_aug), 
                                                                      len(xv_aug[0]), 
                                                                      len(xv_aug[0][0]),
                                                                      len(xv_aug[0][0][0])))
print('Augmented validation masks dimensions: {}, {}, {}, {}'.format(len(yv_aug), 
                                                                     len(yv_aug[0]), 
                                                                     len(yv_aug[0][0]),
                                                                     len(yv_aug[0][0][0])))

# Let's add batch normalization layers 
from keras.layers import BatchNormalization

transfer = base_model.output 
expand1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (transfer)
expand1 = BatchNormalization() (expand1)
expand1 = concatenate([expand1, b2])
c4 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (expand1)
c4 = BatchNormalization() (c4)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)

expand2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c4)
expand2 = BatchNormalization() (expand2)
expand2 = concatenate([expand2, b1])
c5 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (expand2)
c5 = BatchNormalization() (c5)
c5 = Dropout(0.2) (c5)
c5 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)


# PSPNet module
p6 = MaxPooling2D((2, 2), strides=(2, 2)) (c5)
c6_1 = Conv2D(64, (1, 1), activation='elu', padding='same') (p6)
c6_2 = Conv2D(64, (2, 2), activation='elu', padding='same') (p6)
c6_3 = Conv2D(64, (3, 3), activation='elu', padding='same') (p6)
c6_4 = Conv2D(64, (6, 6), activation='elu', padding='same') (p6)
c6_1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6_1)
c6_2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6_2)
c6_3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6_3)
c6_4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6_4)

c6 = concatenate([c5, c6_4, c6_3, c6_2, c6_1])

# final Conv output layers
co1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)
co1 = BatchNormalization() (co1)
co1 = Dropout(0.2) (co1)
co2 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (co1)
co2 = BatchNormalization() (co2)
co2 = Dropout(0.2) (co2)
final_output = Conv2D(1, (1, 1), activation='sigmoid') (co2)

modelf = Model(inputs=[main_input], outputs=[final_output])

# For training freeze layers where VGG19 weights are loaded
for layer in base_model.layers:
    layer.trainable = False

modelf.summary()

modelf.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

# Fit model
earlystopper = EarlyStopping(patience=10, verbose=1) # let's increase patience for this last try
checkpointer = ModelCheckpoint('model-dsbowl2018-f.h5', verbose=1, save_best_only=True)

# Train on augmented set of images/masks
results_final = modelf.fit(np.asarray(xt_aug), np.asarray(yt_aug), 
                           validation_data=(np.asarray(xv_aug), np.asarray(yv_aug)), 
                           batch_size=32, epochs=50, 
                           callbacks=[earlystopper, checkpointer])

# Predict on original train, val and test
model_final = load_model('model-dsbowl2018-f.h5', custom_objects={'mean_iou': mean_iou})
predsf_train = model_final.predict(x_train, verbose=1)
predsf_val = model_final.predict(x_valid, verbose=1)
predsf_test = model_final.predict(X_test, verbose=1)

# Threshold predictions
predsf_train_t = (predsf_train > 0.5).astype(np.uint8)
predsf_val_t = (predsf_val > 0.5).astype(np.uint8)
predsf_test_t = (predsf_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
predsf_test_upsampled = []
for i in range(len(predsf_test)):
    predsf_test_upsampled.append(resize(np.squeeze(predsf_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))

# Check on some random training sample
ix = random.randint(0, len(predsf_train_t))
imshow(x_train[ix])
plt.show()
imshow(np.squeeze(y_train[ix]))
plt.show()
imshow(np.squeeze(predsf_train_t[ix]))
plt.show()

# Check on some random validation sample
ix = random.randint(0, len(predsf_val_t))
imshow(x_valid[ix])
plt.show()
imshow(np.squeeze(y_valid[ix]))
plt.show()
imshow(np.squeeze(predsf_val_t[ix]))
plt.show()

'''
Encode results for submission
'''

import pandas as pd
from skimage.morphology import label

# Run-length encoding 
# from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
        
'''
Iterate over the test IDs and generate run-length encodings for each seperate mask identified by skimage
'''
        
new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(predsf_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))
    

'''
Create submission DataFrame
'''

sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('nuclei_finder_FINAL_deraso.csv', index=False)

