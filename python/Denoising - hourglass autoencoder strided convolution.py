get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import PIL

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import AveragePooling2D
from keras.layers import UpSampling2D
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Add
from keras.optimizers import Adam

encoding_size = 257
perturbation_max = 40

preprocess = lambda x : x / 127 - 1
deprocess  = lambda x :((x + 1) * 127).astype(np.uint8)

img = np.array(PIL.Image.open('dog.png'))
preproc_img = preprocess(img)
image_shape = img.shape
plt.imshow(deprocess(preproc_img))

corruption = np.random.randint(-perturbation_max, perturbation_max, size = image_shape)
corrupted_img = (img + corruption).clip(0, 255)
preproc_corrupted_img = preprocess(corrupted_img)
plt.imshow(deprocess(preproc_corrupted_img))

model_input = Input(shape = image_shape)
conv1       = Convolution2D(32, 3, padding = 'same', activation = 'relu')(model_input)
conv2       = Convolution2D(32, 3, padding = 'same', activation = 'relu')(conv1)
#pool1       = AveragePooling2D()(conv2)
strided_conv1 = Convolution2D(32, 3, strides = (2, 2), padding = 'same')(conv2)
conv3       = Convolution2D(64, 3, padding = 'same', activation = 'relu')(strided_conv1)
conv4       = Convolution2D(64, 3, padding = 'same', activation = 'relu')(conv3)
#pool2       = AveragePooling2D()(conv4)
strided_conv2 = Convolution2D(64, 3, strides = (2, 2), padding = 'same')(conv4)
conv5       = Convolution2D(128, 3, padding = 'same', activation = 'relu')(strided_conv2)
conv6       = Convolution2D(128, 3, padding = 'same', activation = 'relu')(conv5)
flatten     = Flatten()(conv6)
encoding    = Dense(encoding_size, activation = 'relu')(flatten)
dense2      = Dense(192, activation = 'relu')(encoding)
reshape     = Reshape((8, 8, 3))(dense2)
upsample2   = UpSampling2D(size = (4, 4))(reshape)
conv11      = Convolution2D(128, 3, padding = 'same', activation = 'relu')(upsample2)
conv12      = Convolution2D(128, 3, padding = 'same', activation = 'relu')(conv11)
add1        = Add()([conv12, conv6])
upsample3   = UpSampling2D()(add1)
conv13      = Convolution2D(64, 3, padding = 'same', activation = 'relu')(upsample3)
conv14      = Convolution2D(64, 3, padding = 'same', activation = 'relu')(conv13)
# add2        = Add()([conv14, conv4])
upsample3   = UpSampling2D()(conv14)
conv15      = Convolution2D(8, 3, padding = 'same', activation = 'relu')(upsample3)
conv16      = Convolution2D(3, 3, padding = 'same', activation = 'tanh')(conv15)

autoencoder = Model(model_input, conv16)

autoencoder.summary()

autoencoder.compile(Adam(1e-3), loss = 'mse')

base_image = np.random.random(size = (1,) + image_shape) * 2 - 1
corrupted_img_batch = np.expand_dims(preproc_corrupted_img, 0)
fit_params = {
    'x': base_image,
    'y': corrupted_img_batch,
    'epochs': 50,
    'batch_size': 1,
    'verbose': 0
}

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(base_image)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(base_image)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(base_image)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(base_image)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(base_image)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(base_image)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(base_image)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(base_image)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(base_image)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(base_image)
plt.imshow(deprocess(img_pred[0]))

# autoencoder.fit(**fit_params)
# img_pred = autoencoder.predict(corrupted_img_batch)
# plt.imshow(deprocess(img_pred[0]))

plt.figure(figsize = (18, 12))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.subplot(1, 3, 2)
plt.imshow(deprocess(preproc_corrupted_img))
plt.subplot(1, 3, 3)
plt.imshow(deprocess(img_pred[0]))

