import cv2
import numpy as np
from matplotlib import pyplot as plt

filename = 'manga.jpg'

img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

original = cv2.imread(filename)
result = cv2.imread(filename)

plt.subplot(111)
plt.imshow(img, cmap='Greys_r')
plt.show()

ret, thresh = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY_INV)

plt.subplot(121),plt.imshow(original),plt.title('ORIGINAL')
plt.subplot(122),plt.imshow(thresh, cmap='Greys_r'),plt.title('THRESHOLD')
plt.show()

contours, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    cv2.drawContours(result, [cnt], -1, 255, 3)

plt.subplot(121),plt.imshow(original),plt.title('ORIGINAL')
plt.subplot(122),plt.imshow(result),plt.title('THRESHOLD')
plt.show()

for cnt in contours:
    hull = cv2.convexHull(cnt)
    cv2.drawContours(result, [hull], -1, 255, -1)

plt.subplot(121),plt.imshow(original),plt.title('ORIGINAL')
plt.subplot(122),plt.imshow(result),plt.title('THRESHOLD')
plt.show()

