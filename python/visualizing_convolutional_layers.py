import numpy as np
import keras.backend as K
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

model = VGG16(include_top=False)
model.summary()

def make_random_image(img_height=128, img_width=128, mean=127, std=10):
    return np.random.normal(loc=mean, scale=std, size=(img_height, img_width, 3))

random_img = make_random_image()

plt.imshow(random_img)
plt.xticks([])
plt.yticks([])
plt.show()

# find a layer object given a model and layer name
# alternatively, we could construct a dictionary
# of layer name to layer object.
def find_layer(model, layer_name):
    for layer in model.layers:
        if layer.name == layer_name:
            return layer
    return None

# convert result data into 0-255 image data
def as_image(x):
    # normalize data
    x -= x.mean()
    x /= (x.std() + 1e-5)
    # set the std to 0.1 and the mean to 0.5
    x *= 0.1
    x += 0.5
    # scale data and clip between 0 and 255 like an image data
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def layer_image(model, layer_name, filter_index, input_img, steps=20, step_size=1.0):
    layer = find_layer(model, layer_name)
    
    # we want to maximize the mean activation of the filter of the layer
    activation = K.mean(layer.output[:, :, :, filter_index])
    
    # the gradients of the activations of the filter of the layer
    grads = K.gradients(activation, model.input)[0]
    
    # normalize the gradients to avoid very small/large gradients
    grads /= K.sqrt(K.mean(K.square(grads))) + 1e-5
    
    # calculate the mean activation and the gradients which depend on the mean activation
    calculate = K.function([model.input], [activation, grads])
        
    # adjust input image suitable for the calculate function
    input_img = np.copy(input_img)    # make a copy to preserve the original
    input_img = np.float64(input_img) # make sure it's float type
    input_data = input_img.reshape((1, *input_img.shape)) # reshape to one record image data

    # maximize the activation using the gradient ascent
    # (nudge the image data with the gradients)
    for i in range(steps):
        _, grads_value = calculate([input_data])
        input_data += grads_value * step_size
    result = input_data[0]
    
    return as_image(result)

result = layer_image(model, layer_name='block4_conv1', filter_index=0, input_img=random_img)

plt.figure(figsize=(15,5))
plt.imshow(result)
plt.xticks([])
plt.yticks([])
plt.show()

def show_filters(layer_name, input_img):
    print(layer_name)
    plt.figure(figsize=(25,5))
    for i in range(20):
        result = layer_image(model, layer_name, filter_index=i, input_img=input_img)    
        plt.subplot(2, 10, i+1)
        plt.imshow(result)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()

show_filters('block1_conv1', random_img)
show_filters('block1_conv2', random_img)

show_filters('block2_conv1', random_img)
show_filters('block2_conv2', random_img)

show_filters('block3_conv1', random_img)
show_filters('block3_conv2', random_img)
show_filters('block3_conv3', random_img)

show_filters('block4_conv1', random_img)
show_filters('block4_conv2', random_img)
show_filters('block4_conv3', random_img)

show_filters('block5_conv1', random_img)
show_filters('block5_conv2', random_img)
show_filters('block5_conv3', random_img)

cat_img = plt.imread('../images/cat.835.jpg') # the image source is the reference [4]

result = layer_image(model, layer_name='block5_conv3', filter_index=0, input_img=cat_img, steps=100)

plt.figure(figsize=(25,10))
plt.subplot(121)
plt.imshow(cat_img)
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(result)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

