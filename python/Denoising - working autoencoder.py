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
from keras.optimizers import Adam

encoding_size = 128
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

encoder = Sequential([
    Convolution2D(32, 3, padding = 'same', input_shape = image_shape, activation = 'relu'),
    Convolution2D(32, 3, padding = 'same', activation = 'relu'),
    AveragePooling2D(),
    Convolution2D(64, 3, padding = 'same', activation = 'relu'),
    Convolution2D(64, 3, padding = 'same', activation = 'relu'),   
    AveragePooling2D(),
    Convolution2D(128, 3, padding = 'same', activation = 'relu'),
    Convolution2D(128, 3, padding = 'same', activation = 'relu'),
    Flatten(),
    Dense(encoding_size, activation = 'tanh')
])

decoder = Sequential([
    Dense(192, input_shape = (encoding_size,), activation = 'relu'),
    Reshape((8, 8, 3)),
    Convolution2D(128, 3, padding = 'same', activation = 'relu'),
    Convolution2D(128, 3, padding = 'same', activation = 'relu'),
    UpSampling2D(),
    Convolution2D(64, 3, padding = 'same', activation = 'relu'),
    Convolution2D(64, 3, padding = 'same', activation = 'relu'),
    UpSampling2D(),
    Convolution2D(32, 3, padding = 'same', activation = 'relu'),
    Convolution2D(32, 3, padding = 'same', activation = 'relu'),
    UpSampling2D(),    
    Convolution2D(16, 3, padding = 'same', activation = 'relu'),
    Convolution2D(16, 3, padding = 'same', activation = 'relu'),
    UpSampling2D(),
    Convolution2D(8, 3, padding = 'same', activation = 'relu'),
    Convolution2D(3, 3, padding = 'same', activation = 'tanh')
])

autoencoder = Sequential([
    encoder, 
    decoder
])
autoencoder.compile(Adam(1e-2), loss = 'mse')

base_image = np.random.random(size = (1,) + image_shape) * 2 - 1
corrupted_img_batch = np.expand_dims(preproc_corrupted_img, 0)
fit_params = {
    'x': base_image,
    'y': corrupted_img_batch,
    'epochs': 100,
    'batch_size': 1,
    'verbose': 0
}

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(corrupted_img_batch)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(corrupted_img_batch)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(corrupted_img_batch)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(corrupted_img_batch)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(corrupted_img_batch)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(corrupted_img_batch)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(corrupted_img_batch)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(corrupted_img_batch)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(corrupted_img_batch)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(corrupted_img_batch)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(corrupted_img_batch)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(corrupted_img_batch)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(corrupted_img_batch)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(corrupted_img_batch)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(corrupted_img_batch)
plt.imshow(deprocess(img_pred[0]))

autoencoder.fit(**fit_params)
img_pred = autoencoder.predict(corrupted_img_batch)
plt.imshow(deprocess(img_pred[0]))

plt.figure(figsize = (18, 12))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.subplot(1, 3, 2)
plt.imshow(deprocess(preproc_corrupted_img))
plt.subplot(1, 3, 3)
plt.imshow(deprocess(img_pred[0]))

