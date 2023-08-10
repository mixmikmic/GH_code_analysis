# Lets get some essential; imports out of the way.
import numpy as np
import utils
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from keras import backend as K

from vgg16 import VGG16

# Load a pretrained vggnet but make all activations linear
vggnet = VGG16(activation='linear')
print vggnet.summary()

# Load a simple test image.
img = utils.load_img("../images/volcano.jpg", target_size=(224, 224))
plt.imshow(img)

# Lets take the first filter and see what it is computing.
conv1 = vggnet.layers[1]

# Input to the vggnet is float32.
# Also input layer expects a batch.
img_batch = np.array([img], dtype=np.float32)

# Lets build a function to compute conv1 output
out_fn = K.function([vggnet.input], [conv1.output])
output = out_fn([img_batch])[0]
print output.shape

#lets see what the first filter computes.
out = output[0, :, :, 0].copy()
plt.imshow(out, cmap='gray')
plt.colorbar()

out = output[0, :, :, 0].copy()
out[out < 0] = 0
plt.imshow(out, cmap='gray')
plt.colorbar()

out = output[0, :, :, 0].copy()

# Lets remove positive values and flip negative to positive values.
out[out > 0] = 0
out = -out
plt.imshow(out, cmap='gray')
plt.colorbar()

# get_weights returns a list with two weights. The conv filter weights and the bias weights. 
filters = conv1.get_weights()[0]
print filters.shape
print filters[:, :, :, 0]

# Stitch all 3 images into one
stitched = utils.stitch_images(filters[:, :, :, 0], margin=1)
plt.figure(figsize=(10,5))
plt.imshow(stitched, cmap='gray')
plt.colorbar()

# ReLU.
out = output.copy()
out[out < 0] = 0

# Remove the first axis.
out = np.squeeze(out)

# We want to move axis=2 (64) to 0 so that we can treat `out` as a slices of images with (h, w) values.
out = np.moveaxis(out, 2, 0)

# Generate an 8 X 8 image mosaic to see everything side by side.
stitched = utils.stitch_images(out, cols=8, margin=0)
plt.figure(figsize=(20,20))
plt.imshow(stitched, cmap='gray')
plt.colorbar()

#This is the filter output in question
out = output[0, :, :, 31].copy()
plt.imshow(out, cmap='gray')
plt.colorbar()

# ReLU
out[out < 0] = 0

# Normalize to [0, 1]
out /= np.max(out)

# threshold at 0.5
out[out < 0.5] = 0
plt.imshow(out, cmap='gray')
plt.colorbar()

# Threshold to top 20% of the activation to see what this 31st filter is *really* trying to detect.
out[out < 0] = 0
out /= np.max(out)
out[out < 0.8] = 0
plt.imshow(out, cmap='gray')
plt.colorbar()

