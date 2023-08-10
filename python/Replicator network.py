get_ipython().run_line_magic('matplotlib', 'inline')
import keras
import keras.backend     as K
import matplotlib.pyplot as plt
import numpy             as np

from keras.models         import Sequential
from keras.layers         import Convolution2D
from keras.layers         import MaxPooling2D
from keras.layers         import Activation
from keras.layers         import Dense
from keras.layers         import Flatten
from keras.layers         import Reshape
from keras.layers         import UpSampling2D
from keras.optimizers     import Adam
from keras.utils.np_utils import to_categorical
from keras.datasets       import mnist

path_to_fashion_mnist = '../../datasets/fashion-mnist/'

import sys
sys.path.insert(0, path_to_fashion_mnist)
from utils import mnist_reader

anomalies_number = 400
encoding_size    = 200
batch_size       = 64

X_train_fashion, _ = mnist_reader.load_mnist(path_to_fashion_mnist + 'data/fashion', kind = 'train')
X_test_fashion, _  = mnist_reader.load_mnist(path_to_fashion_mnist + 'data/fashion', kind = 't10k')
X_fashion          = np.concatenate((X_train_fashion, X_test_fashion)).reshape(-1, 28, 28, 1)

plt.imshow(X_fashion[35].squeeze(), cmap = 'gray')

(X_train_digits, _), (X_test_digits, _) = mnist.load_data()
X_digits                                = np.concatenate((X_train_digits, X_test_digits)).reshape(-1, 28, 28, 1)

plt.imshow(X_digits[35].squeeze(), cmap = 'gray')

anomalies = np.random.permutation(X_fashion)[:400]

plt.imshow(anomalies[39].squeeze(), cmap = 'gray')

X = np.concatenate((X_digits, anomalies))
X = np.random.permutation(X)
X = (X - X.mean()) / X.std()

encoder = Sequential([
    Convolution2D(64, (3, 3), padding = 'same', input_shape = (28, 28, 1), activation = 'relu'),
    Convolution2D(64, (3, 3), padding = 'same', activation = 'relu'),
    MaxPooling2D(),
    Convolution2D(128, (3, 3), padding = 'same', activation = 'relu'),
    Convolution2D(128, (3, 3), padding = 'same', activation = 'relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation = 'relu'),
    Dense(encoding_size, activation = 'tanh'),
])

decoder = Sequential([
    Dense(49, input_shape = (encoding_size,), activation = 'relu'),
    Reshape((7, 7, 1)),
    Convolution2D(128, (3, 3), padding = 'same', activation = 'relu'),
    Convolution2D(128, (3, 3), padding = 'same', activation = 'relu'),
    UpSampling2D(),
    Convolution2D(64, (3, 3), padding = 'same', activation = 'relu'),
    Convolution2D(64, (3, 3), padding = 'same', activation = 'relu'),
    UpSampling2D(),
    Convolution2D(1, (3, 3), padding = 'same')
])

autoencoder = Sequential([
    encoder,
    decoder
])
autoencoder.compile(optimizer = Adam(1e-4), loss = 'mse')

fit_params = {
    'x'               : X,
    'y'               : X,
    'batch_size'      : batch_size,
    'epochs'          : 30,
    'validation_split': 0.15
}

autoencoder.fit(**fit_params)

digit_img = np.expand_dims(X_digits[2929], 0)
digit_img_ = autoencoder.predict(digit_img)
plt.subplot(1, 2, 1)
plt.imshow(digit_img[0].squeeze(), cmap = 'gray')
plt.subplot(1, 2, 2)
plt.imshow(digit_img_[0].squeeze(), cmap = 'gray')

fashion_img = np.expand_dims(X_fashion[22], 0)
fashion_img_ = autoencoder.predict(fashion_img)
plt.subplot(1, 2, 1)
plt.imshow(fashion_img[0].squeeze(), cmap = 'gray')
plt.subplot(1, 2, 2)
plt.imshow(fashion_img_[0].squeeze(), cmap = 'gray')

autoencoder.save_weights('../models/autoencoder_%dD.h5' % encoding_size)

autoencoder.load_weights('../models/autoencoder_%dD.h5' % encoding_size)

X_ = autoencoder.predict(X, batch_size = 2 * batch_size)

pixel_mse = ((X - X_) ** 2).squeeze()
image_mse = pixel_mse.reshape(pixel_mse.shape[0], -1).mean(axis = 1)

plt.hist(image_mse, bins = 30)

reconstruction_loss_sort_idx = image_mse.argsort()

def create_visualization(X, n, loss_idx):
    plt.figure(figsize = (15, 15))
    X_high_loss = X[loss_idx[-n**2:]]
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            plt.subplot(n, n, idx + 1)
            img = X_high_loss[idx].squeeze()
            plt.imshow(img, cmap = 'gray')
            plt.axis('off')

create_visualization(X, 25, reconstruction_loss_sort_idx)

