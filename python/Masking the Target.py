get_ipython().magic('matplotlib inline')
import cv2
from matplotlib import pyplot as plt
import numpy as np
plt.rcParams['image.cmap'] = 'gray'

image = cv2.imread("RealFullField/15.jpg")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
plt.imshow(blurred)
plt.show()

T = 99
(T, thresh) = cv2.threshold(blurred, T, 255, cv2.THRESH_BINARY)
plt.imshow(thresh)
plt.show()



