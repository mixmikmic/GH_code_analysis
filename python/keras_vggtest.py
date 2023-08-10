import scipy.io
import scipy.misc
import tensorflow as tf
import numpy as np
import time
#from IPython.display import Image

from PIL import Image

from keras import backend
backend.set_image_dim_ordering('th')

from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

from keras.utils.vis_utils import plot_model

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))

model.load_weights("./vgg19_weights.h5")

model.summary()

plot_model(model, show_shapes=True, to_file='model.png')

img_model=Image.open('model.png')

plt.figure(figsize=(30, 60))
plt.axis("off")
plt.imshow(img_model)
plt.show()

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

synsets = []
with open("./synset_words.txt", "r") as f:
    synsets += f.readlines()
synsets = [x.replace("\n","") for x in synsets]

fileList=['img/elephant.jpg','img/bus.jpg']

for actfile in fileList:
    
    # We are going to classify this image:
    print(actfile)
    # Load it
    im=Image.open(actfile).resize((224, 224), Image.ANTIALIAS)
    
    # Plot it
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(im)
    plt.show()
    
    # Preporcess the image
    im = np.array(im).astype(np.float32)

    # scale the image, according to the format used in training
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    
    # Now we can do the prediction
    out = model.predict(im)
    
    # Print 10 class labels with the highest  probabilities
    for index in np.argsort(out)[0][::-1][:10]:
        print("%01.4f - %s" % (out[0][index], synsets[index].replace("\n","")))
        
    print('*********')
        

    

from read_activations import get_activations

activations = get_activations(model, im, print_shape_only=True) 

len(activations)

activations[0][0,0,:,:].shape

plt.imshow(activations[0][0,2,:,:])

activations[1].shape

activations[1][0,1,:,:].shape

plt.imshow(activations[1][0,23,:,:])

plt.imshow(activations[5][0,47,:,:])

plt.imshow(activations[11][0,47,:,:])

print(activations[23].shape)
print(activations[23][0,147,:,:].shape)
plt.imshow(activations[23][0,147,:,:])

print(activations[35].shape)
print(activations[35][0,47,:,:].shape)
plt.imshow(activations[35][0,47,:,:])

mtx=activations[42][0]
mtx.shape
mtx2=mtx.reshape(25,40)
fig=plt.imshow(mtx2)
plt.colorbar(fig)





