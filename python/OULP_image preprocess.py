import os

import numpy as np
import tensorflow as tf

from tensorflow_vgg import vgg16
from tensorflow_vgg import utils

get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
from scipy.ndimage import imread
from scipy.misc import imsave

import cv2
img = imread('00000018.png')
print(type(img))
# plt.imshow(img, cmap='gray')
stacked_img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

print(type(stacked_img))
# stacked_img.reshape((224, 224,3))
print(stacked_img.shape)
plt.imshow(stacked_img)
cv2.imwrite('00000018_3d.png', stacked_img)

import math
math.sqrt(4096)

from PIL import Image 
img1 = Image.open('00000018.png')
img1 = img1.resize((224, 224))
img1 = np.array(img1)
print(img1.shape)
plt.imshow(img1)
print(type(img1))

from PIL import Image 
data_dir = 'OULP-C1V2_Pack/OULP-C1V2_NormalizedSilhouette(88x128)/Seq01'
contents = os.listdir(data_dir)
sequences = [each for each in contents if os.path.isdir(data_dir + '/' + each)]
sequences

for each in sequences:
    print("Starting {} images".format(each))
    sequence_path = data_dir + '/' + each
    if each == '0000000':
        files = os.listdir(sequence_path)
        files.remove('.DS_Store')
    else:
        files = os.listdir(sequence_path)
    for ii, file in enumerate(files, 1):
        img = Image.open(os.path.join(sequence_path, file))
#         img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img = img.resize((224, 224))
        img = np.array(img)
        cv2.imwrite('{}/{}'.format(sequence_path, file), img)

img = Image.open('OULP-C1V2_Pack/OULP-C1V2_NormalizedSilhouette(88x128)/Seq00/0000000/00000018.png')

img = img.resize((224, 224))
img = np.array(img)

plt.imshow(img)
# img.shape



