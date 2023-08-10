import numpy
import os
from scipy.misc import toimage
get_ipython().magic('matplotlib inline')
from IPython.display import Image
feature_path = "../../featuremaps"

os.listdir(feature_path)

data = numpy.load('../../featuremaps/data.npy')

image = data[0][0].astype(numpy.uint8)
image = numpy.abs(image)
image = 100*image#/numpy.max(image)
image = numpy.clip(image, a_min=5, a_max=255)
high = 3600
low = 5200
print "Total: ", numpy.min(image), numpy.max(image)
print " High: ", numpy.min(image[:,:high,:]), numpy.max(image[:,:high,:]) 
print " Low: ", numpy.min(image[:,low:,:]), numpy.max(image[:,low:,:]) 

toimage(image[:,high:low,:]).save('../../featuremaps/data.png')

Image('../../featuremaps/data.png')

data = numpy.load('../../featuremaps/block1_conv1.npy')[0]
print data.shape

border=2
n_wide=8
n_tall = len(data)/n_wide
width=(n_wide)*(data[0].shape[1])+border*(n_wide+1)
height= (n_tall)*(data[0].shape[2])+border*(n_tall+1)
big_image = numpy.ndarray(shape=(3,width,height ), dtype=numpy.uint8)
big_image = big_image+255
print "Big Image Shape: ", big_image.shape
print "Small Image Shape:", data[0].shape

for index, image in enumerate(data):
    image = numpy.abs(image)
    image = 255*image#/numpy.max(image)
    image = numpy.clip(image,0,255)
    toimage(image.astype(numpy.uint8)).save('../../featuremaps/b1c1_{}.png'.format(index))
    x_position = index%n_wide
    y_position = index/n_wide
    
    left_edge = border*(x_position+1)+image.shape[1]*x_position
    upper_edge = border*(y_position+1)+image.shape[2]*y_position
    big_image[:,
              left_edge:left_edge+image.shape[1],
              upper_edge:upper_edge+image.shape[2]] = image
toimage(big_image).save('../../featuremaps/b1c1.png')

Image('../../featuremaps/b1c1.png')

data = numpy.load('../../featuremaps/block2_conv1.npy')[0].astype(numpy.uint8)

border=2
n_wide=8
n_tall = len(data)/n_wide
width=(n_wide)*(data[0].shape[1])+border*(n_wide+1)
height= (n_tall)*(data[0].shape[2])+border*(n_tall+1)
big_image = numpy.ndarray(shape=(3,width,height ), dtype=numpy.uint8)
big_image = big_image+255
print "Big Image Shape: ", big_image.shape
print "Small Image Shape:", data[0].shape

for index, image in enumerate(data):
    image = numpy.abs(image)
    image = 255*image#/numpy.max(image)
    image = numpy.clip(image,0,255)
    toimage(image.astype(numpy.uint8)).save('../../featuremaps/b2c1_{}.png'.format(index))
    x_position = index%n_wide
    y_position = index/n_wide
    
    left_edge = border*(x_position+1)+image.shape[1]*x_position
    upper_edge = border*(y_position+1)+image.shape[2]*y_position
    big_image[:,
              left_edge:left_edge+image.shape[1],
              upper_edge:upper_edge+image.shape[2]] = image
toimage(big_image).save('../../featuremaps/b2c1.png')

Image('../../featuremaps/b2c1.png')

data = numpy.load('../../featuremaps/block3_conv1.npy')[0]
print data.shape

border=2
n_wide=8
n_tall = len(data)/n_wide
width=(n_wide)*(data[0].shape[1])+border*(n_wide+1)
height= (n_tall)*(data[0].shape[2])+border*(n_tall+1)
big_image = numpy.ndarray(shape=(3,width,height ), dtype=numpy.uint8)
big_image = big_image+255
print "Big Image Shape: ", big_image.shape
print "Small Image Shape:", data[0].shape

for index, image in enumerate(data):
    image = numpy.abs(image)
    image = 255*image#/numpy.max(image)
    image = numpy.clip(image,0,255)
    buff = numpy.ndarray(shape=(3, image.shape[1], image.shape[2]))
    for i in range(3): buff[i]=image[0]
    toimage(buff.astype(numpy.uint8)).save('../../featuremaps/b3c1_{}.png'.format(index))
    x_position = index%n_wide
    y_position = index/n_wide
    
    left_edge = border*(x_position+1)+image.shape[1]*x_position
    upper_edge = border*(y_position+1)+image.shape[2]*y_position
    big_image[:,
              left_edge:left_edge+image.shape[1],
              upper_edge:upper_edge+image.shape[2]] = buff
toimage(big_image).save('../../featuremaps/b3c1.png')

Image('../../featuremaps/b3c1.png')

data = numpy.load('../../featuremaps/block4_conv1.npy')[0]
print data.shape

border=2
n_wide=8
n_tall = len(data)/n_wide
width=(n_wide)*(data[0].shape[1])+border*(n_wide+1)
height= (n_tall)*(data[0].shape[2])+border*(n_tall+1)
big_image = numpy.ndarray(shape=(3,width,height ), dtype=numpy.uint8)
big_image = big_image+255
print "Big Image Shape: ", big_image.shape
print "Small Image Shape:", data[0].shape

for index, image in enumerate(data):
    image = numpy.abs(image)
    image = 255*image#/numpy.max(image)
    image = numpy.clip(image,0,255)
    buff = numpy.ndarray(shape=(3, image.shape[1], image.shape[2]))
    for i in range(3): buff[i]=image[0]
    toimage(buff.astype(numpy.uint8)).save('../../featuremaps/b4c1_{}.png'.format(index))
    x_position = index%n_wide
    y_position = index/n_wide
    
    left_edge = border*(x_position+1)+image.shape[1]*x_position
    upper_edge = border*(y_position+1)+image.shape[2]*y_position
    big_image[:,
              left_edge:left_edge+image.shape[1],
              upper_edge:upper_edge+image.shape[2]] = buff
toimage(big_image).save('../../featuremaps/b4c1.png')

Image('../../featuremaps/b4c1.png')

data = numpy.load('../../featuremaps/block5_conv1.npy')[0]
print data.shape

border=2
n_wide=8
n_tall = len(data)/n_wide
width=(n_wide)*(data[0].shape[1])+border*(n_wide+1)
height= (n_tall)*(data[0].shape[2])+border*(n_tall+1)
big_image = numpy.ndarray(shape=(3,width,height ), dtype=numpy.uint8)
big_image = big_image+255
print "Big Image Shape: ", big_image.shape
print "Small Image Shape:", data[0].shape

for index, image in enumerate(data):
    image = numpy.abs(image)
    image = 255*image#/numpy.max(image)
    image = numpy.clip(image,0,255)
    buff = numpy.ndarray(shape=(3, image.shape[1], image.shape[2]))
    for i in range(3): buff[i]=image[0]
    toimage(buff.astype(numpy.uint8)).save('../../featuremaps/b5c1_{}.png'.format(index))
    x_position = index%n_wide
    y_position = index/n_wide
    
    left_edge = border*(x_position+1)+image.shape[1]*x_position
    upper_edge = border*(y_position+1)+image.shape[2]*y_position
    big_image[:,
              left_edge:left_edge+image.shape[1],
              upper_edge:upper_edge+image.shape[2]] = buff
toimage(big_image).save('../../featuremaps/b5c1.png')

Image('../../featuremaps/b5c1.png')



