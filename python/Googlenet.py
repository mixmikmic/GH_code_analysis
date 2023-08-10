from googlenet import *
import numpy as np

img = imresize(imread('cat_pictures/cat1.jpg', mode='RGB'), (224, 224)).astype(np.float32)
img[:, :, 0] -= 123.68
img[:, :, 1] -= 116.779
img[:, :, 2] -= 103.939
img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
img = img.transpose((2, 0, 1))
img = np.expand_dims(img, axis=0)

# Test pretrained model
model = create_googlenet('googlenet_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
out = model.predict(img) # note: the model has three outputs
print(np.argmax(out[2]))

synset_words = open('synset_words.txt', 'rb')
class_names = []
for line in synset_words:
    class_names.append(line[9:-1].decode('UTF-8'))
synset_words.close()

out = model.predict(img) # note: the model has three outputs
print(np.argmax(out[2]))
print(class_names[np.argmax(out[2])])

import json
parsed = json.loads(model.to_json())

#print(json.dumps(parsed, indent=4, sort_keys=True))
parsed['config']['layers']

## Now doing a viz!

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np




def get_max_location(result, filter_number):
    a = result[0][filter_number]
    return np.unravel_index(a.argmax(), a.shape)

# Not sure 'bout this calculation at all.
def get_max_patch(result, filter_number, image):
    max_location = get_max_location(result, filter_number)
    width_ratio = 2#image.shape[1] / result.shape[3]
    height_ratio = 2#image.shape[0] / result.shape[2]
    patch_top = int(max_location[0] * height_ratio-4)
    patch_left = int(max_location[1] * width_ratio-4)
    return image[patch_top:patch_top+8, patch_left:patch_left+8]

def get_max_value(result, filter_number):
    max_location = get_max_location(result, filter_number)
    return result[0][filter_number][max_location[0], max_location[1]]


def max_patches(dataset, predictions=np.array([]), filter_number=0, n=9):
    # Note: no more than 1 patch per image... not mathematically sound, but works well enough.
    if predictions.shape[0] != dataset.shape[0]:
        predictions = model.predict_on_batch(dataset)

    maxes = [-get_max_value([p], filter_number) for p in predictions]
    order = np.argsort(maxes)
    
    max_patches = [
        get_max_patch(np.array([predictions[max_image]]), filter_number, dataset[max_image][0])
        for max_image in order[:n]
    ]
    return max_patches



#from keras.models import Sequential, load_model
#Sequential().fork
from keras import backend as K
LAYER = 3
FILTER = 0

get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[LAYER].output])
layer_output = get_3rd_layer_output([img])[0]
original_image = mpimg.imread("cat.jpg")


plt.figure(figsize=(10, 10))


filter_shape = layer_output[FILTER][0].shape
squished_image_shape = img[0,:,:][0].shape
original_image_shape = original_image.shape

y_filter, x_filter = get_max_location(layer_output, FILTER)

## Draw the filter's output
plt.subplot(1,3,1)
plt.imshow(layer_output[0][0])
plt.plot(x_filter, y_filter, 'go')

## Draw the squished image:
x_squished = x_filter*squished_image_shape[1]/filter_shape[1]
y_squished = y_filter*squished_image_shape[0]/filter_shape[0]
plt.subplot(1,3,2)
plt.imshow(img[0,:,:][0], cmap='gray')
plt.plot(x_squished, y_squished, 'go')


## Draw the original image
plt.subplot(1,3,3)
# Compute percentage
x_original = x_filter*original_image_shape[1]/filter_shape[1]
y_original = y_filter*original_image_shape[0]/filter_shape[0]
plt.imshow(original_image)
plt.plot(x_original, y_original, 'go')


print('\nJoy...</sarcasm> reading the image in different ways gets different shapes:',
      mpimg.imread("cat.jpg").shape,
      img.shape)


plt.figure()
patch_size = 30 //2
plt.imshow(original_image[y_original-patch_size:y_original+patch_size,
                          x_original-patch_size:x_original+patch_size])

## Multipurpose max patch functions

from keras import backend as K
import keras

# Get a list of layers that can be used in max_patch
def get_convolutional_layers(model):
    legal_classes = (keras.layers.Convolution2D, keras.layers.convolutional.ZeroPadding2D)
    return [ layer for layer in model.layers if isinstance(layer, legal_classes)]


# Tested/built for Theano... should be tensorflow compatible
def max_patch(model, data, images=None, layer=None, layer_number=-1, filter_number=0, number_of_patches=9, patch_size=(8,8)):
    
    # images are unpreprocessed data
    if images == None:
        images = data
    
    # Layer is an optional argument
    if layer == None:
        layer = model.layers[layer_number]
    
    # Make sure the layer is a convolutional layer
    if not isinstance(layer, (keras.layers.Convolution2D, keras.layers.convolutional.ZeroPadding2D)):
        print('Hey! Your layer is of class {:}. Are you sure you want to get a 2D max patch of it?'.format(layer.__class__))
    
    # Has shape (1), where each element is the layer's output.
    # A typical layer's output is (1, filters, width, height)
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [layer.output])
    
    # List of np.arrays(shape=(width, height))
    outputs = [get_layer_output([inputs, 0])[0][0][filter_number] for inputs in data]
    
    # Get the maximum values
    maxes = [output.argmax() for output in outputs]
    
    # The indices of the images with the n highest maxes
    image_indices = np.argsort(maxes)[:number_of_patches]
    
    max_outputs = [ outputs[index] for index in image_indices]
    
    # Maximum locations in each 'image'
    # list of (x, y) locations... (technically, (x,y,z,q) locations are fine too)
    max_locations = [np.unravel_index(output.argmax(), output.shape) for output in max_outputs]

    
    # Works for multidimensional input
    # Get the location of the centers as fractions (between 0 and 1)
    # List of (index, (x,y)) where 0 < x < 1
    #fractional_centers = []    
    #for index in range(len(outputs)):
    #    fractions = [loc/total for loc, total in zip(max_locations[index], outputs[index].shape)]
    #    fractional_centers.append(tuple(fractions))    
    
    
    
    # Works only for 2D images
    def patch_from_location(image, max_location, patch_size):
        x = int(max_locations[0][1]/outputs[0].shape[1]*images[0].shape[1])
        y = int(max_locations[0][0]/outputs[0].shape[0]*images[0].shape[0])
        top = y-patch_size[0]//2
        left = x-patch_size[1]//2
        return image[top:top+patch_size[0],
                     left:left+patch_size[1]]
    
    patches = [patch_from_location(images[index], max_locations[index], patch_size)
            for index in range(len(images))]
    
    return patches




get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# GoogLeNet preprocesses images and messes with their dimensions, which makes it a pain to use matplotlib.
# This allows us to create patches that are easy to plot
import matplotlib.image as mpimg

original_image = mpimg.imread("cat.jpg")

data = [img]*2 # input to the neural net
images = [original_image]*2 # Used for generating patches - we cannot call plt.imshow(data[0])

layers = get_convolutional_layers(model) # Get a list of the convolutional layers

patches = max_patch(model, data, images, layer=layers[8], filter_number=34, patch_size=(40,50))

for patch in patches:
    plt.figure()
    plt.imshow(patch)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# GoogLeNet preprocesses images and messes with their dimensions, which makes it a pain to use matplotlib.
# This allows us to create patches that are easy to plot
import matplotlib.image as mpimg

def preprocessImage(number):
    img = imresize(imread('cat_pictures/cat{:}.jpg'.format(number), mode='RGB'), (224, 224)).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


notebook_path = "/Users/ezradavis/Desktop/Ezra's Folder/school/WPI/Fourth Year/mqp/mqpproject"
print('If you\'re not Ezra, make sure that the notebook path is appropriate. Sadly __file__ does not exist for ipynb')

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(notebook_path, fpath))
 
load_src("max_patch", "max_patch.py")
from max_patch import *

data = [preprocessImage(i+1) for i in range(5)]
images = [imresize(mpimg.imread("cat_pictures/cat{:}.jpg".format(i+1)), (224, 224)) for i in range(5)]


layers = get_convolutional_layers(model) # Get a list of the convolutional layers

patches = max_patch(model, data, images,
                    layer=layers[8], filter_number=18, patch_size=(50,50), number_of_patches=5)

for patch in patches:
    plt.figure()
    plt.imshow(patch)

model.layers[0].output_shape



