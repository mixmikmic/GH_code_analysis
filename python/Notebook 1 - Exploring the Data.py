# numerics
import numpy as np
import itertools

# images
from scipy.misc import *
#imresize, imread, imshow
import matplotlib.pylab as plt

# dealing with tar files
import tarfile, gzip

# extracting data about faces/people
import csv

# nice stuff
import os, re
from pprint import pprint

get_ipython().magic('matplotlib inline')

print("Building test/train lists...")

# skip row 0, which is the header
with open('data/pairsDevTrain.txt', 'r') as csvfile:
    trainrows = list(csv.reader(csvfile, delimiter='\t'))[1:]
with open('data/pairsDevTest.txt', 'r') as csvfile:
    testrows = list(csv.reader(csvfile, delimiter='\t'))[1:]

print("Done.")

print("Length of training data set: %d"%(len(trainrows)))

pprint(trainrows[:10])

pprint(trainrows[-10:])

print("Length of test data set: %d"%(len(testrows)))

pprint(testrows[:10])

pprint(testrows[-10:])

with open('data/lfw-names.txt', 'r') as csvfile:
    allrows = list(csv.reader(csvfile, delimiter='\t'))[1:]

for row in allrows:
    if('Bush' in row[0]):
        print(row)

def load_image(tgz_file, basename, name, number):
    
    # images of people are stored in the tar files in the following format:
    # 
    # <basename>/<name>/<name>_<number 04d>.jpg
    #
    # where number comes from the second or third column in the text file
    
    filename = "{0}/{1}/{1}_{2:04d}.jpg".format(basename, name, int(number))
    tgz = tarfile.open(tgz_file)
    return imread(tgz.extractfile(filename))

# From the tarfile of all images of George W. Bush,
tgz = "data/lfw-bush.tgz"

# Load the fifth image:
z = load_image(tgz,"lfw","George_W_Bush",5)

print("Shape of image: W x H x RGB")
print(np.shape(z))

# To show the image in color, 
# convert data z to numpy unsigned 8-bit integer 
# (8 bits = 2^8 = 256 = 0 to 255
plt.imshow(np.uint8(z))

fig = plt.figure(figsize=(14,6))
[ax1, ax2, ax3] = [fig.add_subplot(1,3,i+1) for i in range(3)]

ax1.imshow(z[:,:,0],cmap="gray")
ax2.imshow(z[:,:,1],cmap="gray")
ax3.imshow(z[:,:,2],cmap="gray")
plt.show()

def extract_features(z):
    features = np.array([z[:,:,0],z[:,:,1],z[:,:,2]])
    return features

features = extract_features(z)
print(np.shape(features))

print(features)

# Loading "same person" row
pprint(trainrows[0])

tgz = "data/lfw.tgz"
prefix = "lfw"

def load_one_person(row):
    name = row[0]
    
    imgnum1 = row[1]
    img1 = load_image(tgz, prefix, name, imgnum1)
    
    imgnum2 = row[2]
    img2 = load_image(tgz, prefix, name, imgnum2)
    
    return img1, img2

img1, img2 = load_one_person(trainrows[0])

fig = plt.figure()
ax1, ax2 = [fig.add_subplot(1,2,i+1) for i in range(2)]

ax1.imshow(img1)
ax2.imshow(img2)
plt.show()

# Loading "different person" row
pprint(trainrows[-3])

tgz = "data/lfw.tgz"
prefix = "lfw"

def load_two_persons(row):
    name1 = row[0]
    imgnum1 = row[1]
    img1 = load_image(tgz, prefix, name1, imgnum1)
    
    name2 = row[2]
    imgnum2 = row[3]
    img2 = load_image(tgz, prefix, name2, imgnum2)
    
    return img1, img2

img1, img2 = load_two_persons(trainrows[-3])

fig = plt.figure()
ax1, ax2 = [fig.add_subplot(1,2,i+1) for i in range(2)]

ax1.imshow(img1)
ax2.imshow(img2)
plt.show()

print(np.shape(img1))

print(np.shape(np.ravel(img1)))

