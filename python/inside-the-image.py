from skimage import data, io

image = data.coins()

io.imshow(image)
io.show()

# image type
print("image type",type(image))

# image dimension/shape
print("shape of the image", image.shape)

# This image has 2-dimensional [x, y]

# show the value of pixel [303, 384]
image[303,384]

# These index values are out of bounds.
# Because index of numpy array starts from 0

# value of index [0,0]
print("The value of index[0,0]", image[0,0])

# this last index of this image is [302, 383]
print("The value of index [302,383]",image[302,383])

# assign shape of the image to variable
row, col = image.shape

# Because index starts from 0, then we have to delete 1 value out
row = row - 1
col = col - 1

print("shape of the image", image.shape)
print("Number of row", row)
print("Number of col", col)

# you can assign the shape of the image to variable with this method
row = image.shape[0]-1
col = image.shape[1]-1

print("shape of the image", image.shape)
print("Number of row", row)
print("Number of col", col)

# show value of each pixel
# [row, col]
# row = 0 to 5
# col = 0 to 10
image[0:5,0:10]

# show minimum and maximum value 

print("minimum value", image.min())
print("maximum value", image.max())

from skimage import data, io

image = data.coffee()

io.imshow(image)
io.show()

# image dimension/shape
print("shape of the image", image.shape)

# This image has 3-dimensional [row, col, dim]

# dim[0] = RED value
# dim[1] = GREEN value
# dim[2] = BLUW value

# assign shape of the image to variable
row, col, dim = image.shape

# Because index starts from 0, then we have to delete 1 value out
row = row - 1
col = col - 1
dim = dim - 1

print("shape of the image", image.shape)
print("Number of row", row)
print("Number of col", col)
print("Number of dimension", dim)

# you can assign the shape of the image to variable with this method
row = image.shape[0]-1
col = image.shape[1]-1
dim = image.shape[2]-1

print("shape of the image", image.shape)
print("Number of row", row)
print("Number of col", col)
print("Number of dimension", dim)

# show value of each pixel
# [row, col, dim]
# row = 0 to 5
# col = 0 to 10
# dim = 0
image[0:5,0:10,0]

# show value of each pixel
# [row, col, dim]
# row = 0 to 5
# col = 0 to 10
# dim = 1
image[0:5,0:10,1]

# show value of each pixel
# [row, col, dim]
# row = 0 to 5
# col = 0 to 10
# dim = 2
image[0:5,0:10,2]

from skimage import data, io

# read image from directory
image = io.imread('image/bw-cat-image.png')

io.imshow(image)
io.show()

# image dimension/shape
print("shape of the image", image.shape)

# This image has 2-dimensional [row, col]

# you can assign the shape of the image to variable with this method
row = image.shape[0]-1
col = image.shape[1]-1

print("shape of the image", image.shape)
print("Number of row", row)
print("Number of col", col)

# show value of each pixel
# [row, col]

image[75:90,80:90]

#0 = low intensity = black
# 255 /or/ 1 = high intensity = white

print("min value", image.min())
print("max value", image.max())

# change value 255 in the array to 1

# Replace all elements of numPy that are greater than specific value
image[image == 255] = 1

image[75:90,80:90]

# Maybe you can get the problem when show the image 
io.imshow(image)
io.show()

import matplotlib.pyplot as plt
import numpy as np

bw_img = np.asarray(image)
plt.imshow(bw_img, cmap='gray')
plt.show()

