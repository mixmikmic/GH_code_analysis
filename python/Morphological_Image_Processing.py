import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import PIL
import cv2
import skimage as sk
get_ipython().magic('pylab inline')

img = cv2.imread('images/binary_circles.jpg',0)
kernel1 = np.ones((3,3), np.uint8)
erosion1 = cv2.erode(img, kernel1, iterations = 1)
kernel2 = np.ones((5,5), np.uint8)
erosion2 = cv2.erode(img, kernel2, iterations = 1)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img, cmap=plt.cm.gray)
plt.title('Original image')
plt.subplot(132)
plt.imshow(erosion1, cmap=plt.cm.gray)
plt.title(r'Erosion with a $3\times3$ kernel')
plt.subplot(133)
plt.imshow(erosion2, cmap=plt.cm.gray)
plt.title(r'Erosion with a $5\times5$ kernel')

img = cv2.imread('images/binary_retina.png',0)
kernel1 = np.ones((3,3), np.uint8)
dilation1 = cv2.dilate(img, kernel1, iterations = 1)
kernel2 = np.ones((5,5), np.uint8)
dilation2 = cv2.dilate(img, kernel2, iterations = 1)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img, cmap=plt.cm.gray)
plt.title('Original image')
plt.subplot(132)
plt.imshow(dilation1, cmap=plt.cm.gray)
plt.title(r'Dilation with a $3\times3$ kernel')
plt.subplot(133)
plt.imshow(dilation2, cmap=plt.cm.gray)
plt.title(r'Dilation with a $5\times5$ kernel')

img = cv2.imread('images/binary_sand.png',0)
kernel1 = np.ones((5,5), np.uint8)
opening1 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel1)
kernel2 = np.ones((7,7), np.uint8)
opening2 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img, cmap=plt.cm.gray)
plt.title('Original image')
plt.subplot(132)
plt.imshow(opening1, cmap=plt.cm.gray)
plt.title(r'Opening with a $5\times5$ kernel')
plt.subplot(133)
plt.imshow(opening2, cmap=plt.cm.gray)
plt.title(r'Opening with a $7\times7$ kernel')

img = cv2.imread('images/binary_circles.jpg',0)
kernel1 = np.ones((5,5), np.uint8)
closing1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1)
kernel2 = np.ones((7,7), np.uint8)
closing2 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img, cmap=plt.cm.gray)
plt.title('Original image')
plt.subplot(132)
plt.imshow(closing1, cmap=plt.cm.gray)
plt.title(r'Closing with a $5\times5$ kernel')
plt.subplot(133)
plt.imshow(closing2, cmap=plt.cm.gray)
plt.title(r'Closing with a $7\times7$ kernel')

img = cv2.imread('images/binary_circles.jpg',0)
kernel1 = np.ones((3,3), np.uint8)
grad1 = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel1)
kernel2 = np.ones((5,5), np.uint8)
grad2 = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel2)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img, cmap=plt.cm.gray)
plt.title('Original image')
plt.subplot(132)
plt.imshow(grad1, cmap=plt.cm.gray)
plt.title(r'Morphological gradient with a $3\times3$ kernel')
plt.subplot(133)
plt.imshow(grad2, cmap=plt.cm.gray)
plt.title(r'Morphological gradient with a $5\times5$ kernel')

img = cv2.imread('images/binary_angiogram.png',0)
kernel1 = np.ones((5,5), np.uint8)
top1 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel1)
kernel2 = np.ones((9,9), np.uint8)
top2 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel2)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img, cmap=plt.cm.gray)
plt.title('Original image')
plt.subplot(132)
plt.imshow(top1, cmap=plt.cm.gray)
plt.title(r'Morphological tophat with a $5\times5$ kernel')
plt.subplot(133)
plt.imshow(top2, cmap=plt.cm.gray)
plt.title(r'Morphological tophat with a $9\times9$ kernel')

img = cv2.imread('images/binary_circles.jpg',0)
kernel1 = np.ones((5,5), np.uint8)
black1 = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel1)
kernel2 = np.ones((11,11), np.uint8)
black2 = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel2)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img, cmap=plt.cm.gray)
plt.title('Original image')
plt.subplot(132)
plt.imshow(black1, cmap=plt.cm.gray)
plt.title(r'Morphological blackhat with a $5\times5$ kernel')
plt.subplot(133)
plt.imshow(black2, cmap=plt.cm.gray)
plt.title(r'Morphological blackhat with a $11\times11$ kernel')

# Rectangular Kernel
rect = cv2.getStructuringElement(cv2.MORPH_RECT,(25,25))
# Elliptical Kernel
ellip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
# Cross-shaped Kernel
cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(25,25))

plt.matshow(ellip, cmap=cm.gray)
plt.title(r'A $19\times 19$ elliptical / circular kernel')

img = cv2.imread('images/binary_circles.jpg',0)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
closing1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
closing2 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img, cmap=plt.cm.gray)
plt.title('Original image')
plt.subplot(132)
plt.imshow(closing1, cmap=plt.cm.gray)
plt.title(r'Closing with a $5\times5$ circular kernel')
plt.subplot(133)
plt.imshow(closing2, cmap=plt.cm.gray)
plt.title(r'Closing with a $15\times15$ circular kernel')

img = cv2.imread('images/binary_circles.jpg',0)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
black1 = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel1)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
black2 = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel2)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img, cmap=plt.cm.gray)
plt.title('Original image')
plt.subplot(132)
plt.imshow(black1, cmap=plt.cm.gray)
plt.title(r'Blackhat with a $9\times9$ circular kernel')
plt.subplot(133)
plt.imshow(black2, cmap=plt.cm.gray)
plt.title(r'Blackhat with a $15\times15$ circular kernel')



