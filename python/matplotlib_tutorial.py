import numpy as np
import matplotlib.pyplot as plt

# compute x and y coordinates for points on a sine curve
x = np.arange(0, 3*np.pi, 0.1)
y = np.sin(x)

# plot the points using matplotlib
plt.plot(x, y)
plt.show() # plt.show() must be called to make the graphics appear

# compute x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3*np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('X-axis label')
plt.ylabel('Y-axix label')
plt.title('Sine and Cosine')
plt.legend(['sine, cosine'])
plt.show()

# setup a subplot grid that has a height 2 and width 1 and set the first such subplot as active
plt.subplot(2,1,1)

# make the first plot
plt.plot(x, y_sin)
plt.title('sine')

# set the second subplot as active and make the second subplot
plt.subplot(2,1,2)
plt.plot(x, y_cos)
plt.title('cosine')

# show the figure
plt.show()

from scipy.misc import imread, imresize

img = imread('./bajrangbali')
img_tinted = img * [1, 0.55, 0.5]

# show the original image
plt.subplot(1,2,1)
plt.imshow(img)

# show the tinted image
plt.subplot(1,2,2)

# a slight gotcha with imshow is that it might give strange results if presented with data that is nor uint8.
# to work around this, we explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))
plt.show()

