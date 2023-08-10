import numpy as np

import bokeh.io
bokeh.io.output_notebook()

import bokeh.plotting
import bokeh.layouts

import sklearn.datasets

N = 8
img = np.zeros((N,N), dtype=np.uint32)  # create an NxN image, each pixel with 32-bits
view = img.view(dtype=np.uint8).reshape((N, N, 4))  # break each of the pixels into 4 8-bit definitions

# create a column of red pixels
view[:, 0, 0] = 255 

# create a column of blue pixels
view[:, 2, 2] = 255

# create a column of green pixels
view[:, 4, 1] = 255

# create a column of white pixels
view[:, 6, :] = 255

# make all pixels fully visible (no alpha)
view[:, :, 3] = 255

p = bokeh.plotting.figure(width=400, height=400, x_range=(0,10), y_range=(0,10))

# must give a vector of images
p.image_rgba(image=[img], x=0, y=0, dw=10, dh=10)

bokeh.plotting.show(p)

N = 8
img = np.zeros((N,N), dtype=np.uint32)  # create an NxN image, each pixel with 32-bits
view = img.view(dtype=np.uint8).reshape((N, N, 4))  # break each of the pixels into 4 8-bit definitions

view[:, :, 3] = 255  # make all pixels fully visible (no alpha)

# setup the columns
view[:, 0, 0] = 255  # create a column of red pixels
view[:, 2, 2] = 255  # create a column of blue pixels
view[:, 4, 1] = 255  # create a column of green pixels
view[:, 6, :] = 255  # create a column of white pixels

# setup the rows
view[0, :, 0] = 255  # create a column of red pixels
view[2, :, 2] = 255  # create a column of blue pixels
view[4, :, 1] = 255  # create a column of green pixels
view[6, :, :] = 255  # create a column of white pixels

p = bokeh.plotting.figure(width=400, height=400, x_range=(0,10), y_range=(0,10))

# must give a vector of images
p.image_rgba(image=[img], x=0, y=0, dw=10, dh=10)

bokeh.plotting.show(p)

def rgba_from_4bit(img_4):
    n, m = img_4.shape
    img_rgba = np.empty((n, m), dtype=np.uint32)
    view = img_rgba.view(dtype=np.uint8).reshape((n, m, 4))
    view[:, :, 3] = 255  # set all alpha values to fully visible
    rgba = 255 - img_4[:, :] / 16 * 255
    
    # rgba is upside-down, hence the ::-1
    view[:, :, 0] = view[:, :, 1] = view[:, :, 2] = rgba[::-1]
    
    return img_rgba

digits = sklearn.datasets.load_digits()

# some information about the data
digits.DESCR

p = bokeh.plotting.figure(width=110, height=100, x_range=(0, 8), y_range=(0, 8),
                          tools='', title='Training: {}'.format(digits.target[0]))
p.xaxis.visible = p.yaxis.visible = False

p.image_rgba(image=[rgba_from_4bit(digits.images[0])], x=0, y=0, dw=8, dh=8)

bokeh.plotting.show(p)

images_and_labels = list(zip(digits.images, digits.target))

n_plot = 4
plots = []
w = 80
h = 80

for index, (image, label) in enumerate(images_and_labels[:n_plot]):
    img_rgba = rgba_from_4bit(image)
    p = bokeh.plotting.figure(width=w, height=h, tools='', 
                              x_range=(0, 8), y_range=(0, 8),
                              title='Training: {}'.format(label))
    p.xaxis.visible = p.yaxis.visible = False
    p.image_rgba(image=[img_rgba], x=0, y=0, dw=8, dh=8)
    plots.append(p)

grid = bokeh.layouts.gridplot([plots])
bokeh.plotting.show(grid)

import matplotlib.pyplot as plt

for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
plt.show()



