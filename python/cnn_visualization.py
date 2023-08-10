import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import skimage.filters
import keras
from keras.applications import vgg16
from keras import backend as K

# Allow graph embeding in notebook
get_ipython().magic('matplotlib inline')

def tensor_summary(tensor):
    """Display shape, min, and max values of a tensor."""
    print("shape: {}  min: {}  max: {}".format(tensor.shape, tensor.min(), tensor.max()))

    
def normalize(image):
    """Takes a tensor of 3 dimensions (height, width, colors) and normalizes it's values
    to be between 0 and 1 so it's suitable for displaying as an image."""
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min() + 1e-5)


def display_images(images, titles=None, cols=5, interpolation=None, cmap="Greys_r"):
    """
    images: A list of images. I can be either:
        - A list of Numpy arrays. Each array represents an image.
        - A list of lists of Numpy arrays. In this case, the images in
          the inner lists are concatentated to make one image.
    """
    titles = titles or [""] * len(images)
    rows = math.ceil(len(images) / cols)
    height_ratio = 1.2 * (rows/cols) * (0.5 if type(images[0]) is not np.ndarray else 1)
    plt.figure(figsize=(11, 11 * height_ratio))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.axis("off")
        # Is image a list? If so, merge them into one image.
        if type(image) is not np.ndarray:
            image = [normalize(g) for g in image]
            image = np.concatenate(image, axis=1)
        else:
            image = normalize(image)
        plt.title(title, fontsize=9)
        plt.imshow(image, cmap=cmap, interpolation=interpolation)
        i += 1

# Build a VGG16 Convolutional Network pre-trained on ImageNet
model = vgg16.VGG16(weights='imagenet')
model.summary()

# Pick a random image from the Web.
# Make sure it's 224x224 image since VGG16 expects this size.
image = skimage.io.imread("http://lorempixel.com/224/224/animals/")
tensor_summary(image)
display_images([image], cols=2)

# Convert image to float
x = image.astype(np.float32)
# Make it a batch of one. The model expects a batch, not a single image
x = x[np.newaxis,...]
# Preprocess image. Convert RGB to BGR and subtract the ImageNet mean.
x = vgg16.preprocess_input(x)

# Classify the image
predictions = model.predict(x)
# We'll get a 1000 values. Print the first 10.
print(predictions[0][:10])

# Find the largest confidence value. This corresponds to the label index.
label_index = np.argmax(predictions)
print("label index: ", label_index)
# Display the top 5 classes
vgg16.decode_predictions(predictions)

# List of ImageNet classes. Print a subset of the 1000
# The previous call to decode_predictions() sets the value
# of CLASS_INDEX, so call that function first.
imagenet_classes = keras.applications.imagenet_utils.CLASS_INDEX
for i in range(10):
    print(i, imagenet_classes[str(i)][1])

step = 56
heatmap_x = []
for row in range(0, image.shape[0], step):
    for col in range(0, image.shape[1], step):
        new_image = image.copy()
        # Add a square patch. Using a bright color here to make it easier to see.
        new_image[row:row+step, col:col+step, :] = [250,128,128]
        heatmap_x.append(new_image)
heatmap_x = np.stack(heatmap_x)
heatmap_x.shape

display_images(heatmap_x[:28], cols=8)

heatmap_y = model.predict(vgg16.preprocess_input(heatmap_x.astype(np.float32)))
tensor_summary(heatmap_y)

probs = heatmap_y[:, label_index]
tensor_summary(probs)

heatmap = (probs.max() - probs) / (probs.max()-probs.min())
heatmap = np.reshape(heatmap, (4, 4))
tensor_summary(heatmap)

tensor_summary(heatmap)
_ = plt.imshow(heatmap, cmap=plt.cm.Reds)

def apply_mask(image, mask):
    # Resize mask to match image size
    mask = skimage.transform.resize(normalize(mask), image.shape[:2])[:,:,np.newaxis].copy()
    # Apply mask to image
    image_heatmap = image * mask
    tensor_summary(image_heatmap)
    display_images([image_heatmap], cols=2)


# Apply mask to image
apply_mask(image, heatmap**2)

weights = model.get_layer("block1_conv1").get_weights()[0]
weights.shape, weights.min(), weights.max()

display_images([weights[:,:,::-1,i] for i in range(64)], cols=16, interpolation="none")

def read_layer(model, x, layer_name):
    """Return the activation values for the specifid layer"""
    # Create Keras function to read the output of a specific layer
    get_layer_output = K.function([model.layers[0].input], [model.get_layer(layer_name).output])
    outputs = get_layer_output([x])[0]
    tensor_summary(outputs)
    return outputs[0]
    
def view_layer(model, x, layer_name, cols=5):
    outputs = read_layer(model, x, layer_name)
    display_images([outputs[:,:,i] for i in range(10)], cols=cols)

view_layer(model, x, "block1_conv1")

view_layer(model, x, "block1_conv2")

view_layer(model, x, "block2_conv1")

view_layer(model, x, "block3_conv1")

view_layer(model, x, "block4_conv1")

view_layer(model, x, "block5_conv3")

a = read_layer(model, x, "block5_conv3")
apply_mask(image, a[:,:,0])

# The last layer is a 1D 1000-vector. Visualize as a bar chart.
a = read_layer(model, x, "predictions")
_ = plt.plot(a)

def build_backprop(model, loss):
    # Gradient of the input image with respect to the loss function
    gradients = K.gradients(loss, model.input)[0]
    # Normalize the gradients
    gradients /= (K.sqrt(K.mean(K.square(gradients))) + 1e-5)
    # Keras function to calculate the gradients and loss
    return K.function([model.input], [loss, gradients])

# Loss function that optimizes one class
loss_function = K.mean(model.get_layer("predictions").output[:,label_index])

# Backprop function
backprop = build_backprop(model, loss_function)

# Calculate gradients on the input image
loss, grads = backprop([x])
tensor_summary(grads)

# Visualize the gradients
grad_image = normalize(grads)
display_images([grad_image[0], image], cols=2)

# Start with a random image
random_image = np.random.random((1, 224, 224, 3))
display_images(random_image)

# Iteratively apply gradient ascent
for i in range(50):
    loss, grads = backprop([random_image])
    
    # Multiply gradients by the learning rate and add to the image
    # Optionally, apply a gaussian filter to the gradients to smooth
    # out the generated image. This gives better results.
    # The first line, which is commented out, is the native method
    # and the following line uses the filter. Try with both to
    # see the difference.
    #
    # random_image += grads * .1
    random_image += skimage.filters.gaussian(np.clip(grads, -1, 1), 2) 

    # Print loss value
    if i % 10 == 0:
        print('Loss:', loss)

tensor_summary(random_image)
display_images(random_image[...,::-1], cols=2)

# Classify the image
predictions = model.predict(random_image)
vgg16.decode_predictions(predictions)

# Loss function that optimizes one class
loss_function = K.mean(model.get_layer("block1_conv1").output[:,:,:,0])

# Backprop function
backprop = build_backprop(model, loss_function)

# Start with a random image
random_image = np.random.random((1, 224, 224, 3)) 

# Iteratively apply gradient ascent
for i in range(50):
    loss, grads = backprop([random_image])
    grads = np.nan_to_num(grads)  # In case gradients are NaN
    # Apply gradients with or without a gaussian filter
    # random_image += grads * .1
    random_image += skimage.filters.gaussian(np.clip(grads, -1, 1), 2) 
    if i % 10 == 0:
        print('Loss:', loss)

tensor_summary(random_image)
display_images(random_image[...,::-1], cols=2)

# Read activiations all all layers
activations = []
for layer in model.layers[1:]:
    # Extract activations
    get_layer_output = K.function([model.input], [layer.output])
    activations.append( (layer.name, get_layer_output([x])[0]) )
    tensor_summary(activations[-1][1])

# Build the loss function
loss_function = 0
for l, a in activations:
    print(l)
    loss_function += K.mean(K.sqrt((model.get_layer(l).output - a)**2))
loss_function = -loss_function

# Backprop function
backprop = build_backprop(model, loss_function)

# Start with a random image
random_image = np.random.random((1, 224, 224, 3)) * 10

for i in range(50):
    loss, grads = backprop([random_image])
    grads = np.nan_to_num(grads)
    #random_image += grads * .1
    random_image += skimage.filters.gaussian(np.clip(grads, -1, 1), 2) 
    if i % 10 == 0:
        print('Loss:', loss)

tensor_summary(random_image)
display_images(random_image[...,::-1], cols=2)



