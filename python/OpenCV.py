import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve 

get_ipython().magic('matplotlib inline')

plt.axis('off')

# download the Statue of Liberty
urlretrieve('https://upload.wikimedia.org/wikipedia/commons/a/a1/Statue_of_Liberty_7.jpg', 'liberty.jpg')

# and read in the image
img = cv2.imread('liberty.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(img.shape)
plt.imshow(img)

# we can get a single pixel
print(img[100, 100])
# and edit it
img[100, 100, 0] = 10
print(img[100, 100])

# crop the image
statue = img[50:1200, 500:1000]
plt.imshow(statue)

# add noise to the image, making a copy in the process
im = np.zeros((2022, 1464, 3), np.uint8)
cv2.randn(im,(0),(99))
img_2 = im + img
plt.imshow(img_2)

# easily resize the image
resized = cv2.resize(img, (0,0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
print(resized.shape)
plt.imshow(resized)

# flip the image
flipped = cv2.flip(img, 1)
plt.imshow(flipped)

# convert the image from RGB space to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# and apply a mask
lower = np.array([50, 0, 0])
upper = np.array([220, 200, 200])

# Threshold the HSV image to get only statue
mask = cv2.inRange(hsv, lower, upper)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(img, img, mask= mask)

plt.imshow(res)

# for a thorough look at using OpenCV in a deep learning solution:
# https://github.com/Azure/ObjectDetectionUsingCntk

