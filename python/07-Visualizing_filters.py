from __future__ import print_function

from scipy.misc import imsave
import numpy as np
import time
from keras import backend as K

from keras import applications

# build the VGG16 network
model = applications.VGG16(include_top=False,
                           weights='imagenet')

model.input

get_ipython().run_line_magic('run', '__initremote__.py')

from keras.models import load_model
model = load_model('datagen_115.h5')

weights = model.get_weights()

class_names_list = ['airplane',
                    'automobile',
                    'bird',
                    'cat',
                    'deer',
                    'dog',
                    'frog',
                    'horse',
                    'ship',
                    'truck']

y_train = y_train.tolist()

for y in range(len(y_train)):
    y_train[y] = y_train[y][0]
    
y_train

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
def show_unique_images(images, labels, class_names=class_names_list):
    unique_labels = []
    unique_indices = []
    
    fig = plt.figure(figsize=(6,3))
    
    n = 0
    for i in range(len(labels)):
        if labels[i] not in unique_labels:
            #image = images[i].reshape(3,32,32).transpose(1,2,0)
            image = images[i]
            plt.subplot(2,5,n+1)
            n += 1
            plt.imshow(image, interpolation="nearest")
            plt.title(class_names[labels[i]])
            unique_labels.append(labels[i])
            unique_indices.append(i)
    plt.show()
    
    return unique_indices

indices = show_unique_images(x_train, y_train)

img_width = 128
img_height = 128

model.layers

layer_dict = dict([(layer.name, layer) for layer in model.layers])
layer_dict

input_img = model.input

input_img

#layer_name = 'conv2d_29'
layer_name = 'block5_conv1'

filter_index = 0

layer_output = layer_dict[layer_name].output
#Utilizes backend function K.mean() to find the mean of a Tensor
loss = K.mean(layer_output[:, :, :, filter_index])
grads = K.gradients(loss, input_img)[0]

layer_output

grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
iterate = K.function([input_img], [loss, grads])

def deprocess_image(x):
    '''Returns the image to a 0 to 255 range int'''
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1
    
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

# we start from a gray image with some noise
input_img_data = np.random.random((1, 3, img_width, img_height)) * 20 + 128.
# run gradient ascent for 20 steps
for i in range(20):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step

img = input_img_data[0]
img = deprocess_image(img)
#imsave('%s_filter_%d.png' % (layer_name, filter_index), img)
plt.imshow(img, )

