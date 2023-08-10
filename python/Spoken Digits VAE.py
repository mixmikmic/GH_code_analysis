get_ipython().magic('matplotlib inline')
import IPython.display

import gzip
import cPickle as pickle
import pandas as pd
import random
import seaborn
import librosa
import numpy as np
from matplotlib import pyplot as plt

from librosa.display import waveplot, specshow
seaborn.set(style='ticks')

dbfile ='../SpokenDigitDB.pkl.gz'
with gzip.open(dbfile, 'rb') as ifile:
    df = pickle.load(ifile)
    print('File loaded as '+ dbfile)    

df.info()

mag = df.Magnitude
mgs = [np.shape(x)[1] for x in mag]
maxlen = np.max(mgs)
print('Maximum length is: {} '.format(maxlen))
plt.hist(mgs,50)

# Padding & Truncating
maxlen = 84
# pad    = lambda a, n: a[:,0: n] if a.shape[1] > n else np.hstack((a, np.zeros([a.shape[0],n - a.shape[1]])))
pad    = lambda a, n: a[:,0: n] if a.shape[1] > n else np.hstack((a, np.min(a[:])*np.ones([a.shape[0],n - a.shape[1]])))

df.Magnitude = df.Magnitude.apply(pad,args=(maxlen,))  # MaxLen Truncation Voodoo :D
df.Phase     = df.Phase.apply(pad,args=(maxlen,))

print(np.unique([np.shape(x)[1] for x in df.Magnitude]))
print(np.unique([np.shape(x)[1] for x in df.Phase]))

seaborn.set(style='white')

# Plot K Random Examples
k  = 5
sr = 8000

sidx = random.sample(range(len(df)),k)
sidx = np.append(sidx,[sidx,sidx])    

for i,j in enumerate(sidx):
    if i<k:
        plt.subplot(3,k,i+1)
        waveplot(df.Wave[j],sr=sr)
        plt.title('Digit:{1}'.format(j,df.Class[j]))
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.gca().get_xaxis().set_visible(False)

    elif (i>=k and i<2*k):
        plt.subplot(3,k,i+1)
        specshow(df.Magnitude[j],sr=sr)
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        
    else:
        plt.subplot(3,k,i+1)
        specshow(df.Phase[j],sr=sr)
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])        

# np.max(df.Magnitude[j])
# np.max(df.Phase[j])

# Play back an example!
j = sidx[1]
IPython.display.Audio(data=df.Wave[j], rate=sr)

# Imports
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard

from keras import metrics
# from keras.datasets import mnist
from sklearn.utils import shuffle
from scipy.stats import norm

chns, rows, cols = 1, 64, 84
filters = 8

batch_size = 10
latent_dim = 2
middle_dim = 128

epochs = 25
epsilon_std = 1.0
img_size = (rows,cols,chns)

# x = Input(input_shape=(batch_size,)+img_size)
x = Input(shape=img_size)

# Encoder
conv1 = Conv2D(filters,(3,5),padding='same',activation='relu',strides=(2,3))(x)
conv2 = Conv2D(filters,(3,3),padding='same',activation='relu',strides=(2,2))(conv1)
conv  = Flatten()(conv2)
encoded = Dense(middle_dim,activation='relu')(conv)

# Latent Distribution
z_mean = Dense(latent_dim)(encoded)
z_lvar = Dense(latent_dim)(encoded)

# Gaussian Sampler
def sampling(args):
    z_mean, z_lvar = args
    bsize = K.shape(z_mean)[0]
    epsilon = K.random_normal(shape=(bsize, latent_dim),
                              mean=0.0, stddev=epsilon_std)
    return z_mean + K.exp(z_lvar / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_lvar])

## Intermediate Shared Layer - Decoder
# Instantiate Layer Objects
decode_in = Dense(middle_dim, activation='relu')
upsample  = Dense(filters*16*14, activation='relu')
ureshape  = Reshape((16,14,filters))
trconv1   = Conv2DTranspose(filters,3,padding='same',
                          activation='relu',strides=2)
trconv2   = Conv2DTranspose(filters,(3,5),padding='same',
                          activation='relu',strides=(2,3))
decode_ou = Conv2D(chns,2,padding='same',activation='relu')

# Stack Decoder Onto Encoder
in_decode = decode_in(z)
up_decode = upsample(in_decode)
re_decode = ureshape(up_decode)
c1_decode = trconv1(re_decode)
c2_decode = trconv2(c1_decode)
ou_decode = decode_ou(c2_decode)

# # Intermediate Shared Layer - Decoder
# decode_h = Dense(middle_dim, activation='relu')(z)

# upsample = Dense(filters*16*14, activation='relu')(decode_h)
# ureshape = Reshape((16,14,filters))(upsample)

# trconv1  = Conv2DTranspose(filters,3,padding='same',
#                           activation='relu',strides=2)(ureshape)
# trconv2  = Conv2DTranspose(filters,(3,5),padding='same',
#                           activation='relu',strides=(2,3))(trconv1)

# decoded  = Conv2D(chns,2,padding='same',activation='relu')(trconv2)

def vae_loss(x, decoded):
    x = K.flatten(x)
    decoded = K.flatten(decoded)
    gen_loss = rows * cols * metrics.binary_crossentropy(x, decoded)
    kl_loss = - 0.5 * K.mean(1 + z_lvar - K.square(z_mean) - K.exp(z_lvar), axis=-1)
    return K.mean(gen_loss + kl_loss)

# Model
vae = Model(x,ou_decode)
vae.compile(optimizer='rmsprop', loss=vae_loss)
vae.summary()

# Encoder
encoder = Model(x,z_mean)

# Generator
dd_in = Input(shape=(latent_dim,))
dd    = decode_in(dd_in)
dd    = upsample(dd)
dd    = ureshape(dd)
dd    = trconv1(dd)
dd    = trconv2(dd)
dd_ou = decode_ou(dd)
generator = Model(dd_in,dd_ou) 

# Get Training Data
# x_data = df.Magnitude.values
# x_data = np.dstack(x_data)
# x_data = x_data.transpose(2,0,1)
# x_data = x_data[...,None]         # add singleton class
# x_data = shuffle(x_data)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Train Scaler
x_data = df.Magnitude.values
normsc = np.hstack(x_data)
scaler = MinMaxScaler().fit(normsc.T)

# Transform Data using Scaler
x_data = [scaler.transform(arr.T).T for arr in df.Magnitude.values]
x_data = np.dstack(x_data).transpose(2,0,1)

# Add Singleton & Shuffle
x_data = x_data[...,None]         # Add singleton class
# x_data = shuffle(x_data)

vae.fit(x_data,x_data,
        shuffle=True,
        epochs=20,
        batch_size=50,
        callbacks=[TensorBoard(log_dir='/tmp/vautoeconder')])

## 2D Scatter Plot of Latent Encodings
# Encode to Latent Space
x_encoded = encoder.predict(x_data, batch_size=50)
y_data    = df.Class.values

# Use PCA Decomposition
from matplotlib.mlab import PCA
# xpca = PCA(x_encoded).Y
xpca = x_encoded

plt.figure(figsize=(4, 4))
plt.scatter(xpca[:, 0], xpca[:, 1], c=y_data,cmap='viridis')
plt.colorbar()
plt.show()

## Display 2D Manifold
n = 8   
figure = np.zeros((rows * n, cols * n))

grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))


# 10 Latent Dimensions
# for i, yi in enumerate(grid_x):
#     for j, xi in enumerate(grid_y):
#         z_sample  = np.random.uniform(size=10)
#         z_sample  = z_sample[...,None].T 
#         x_decoded = generator.predict(z_sample)
#         digit = x_decoded.squeeze()
#         figure[i * rows: (i + 1) * rows,
#                j * cols: (j + 1) * cols] = digit

## 2 Latent Dimensions
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded.squeeze()
        figure[i * rows: (i + 1) * rows,
               j * cols: (j + 1) * cols] = digit
        
plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='inferno')
plt.show()

k  = 500
xt = x_data[k]
xx = vae.predict(xt[None,...])

plt.subplot(211)
plt.title('Original Spectrogram')
ss=xt.squeeze()
specshow(ss,sr=sr)

plt.subplot(212)
plt.title('VAE Generated Spectrogram')
ss=xx.squeeze()
specshow(ss,sr=sr)



