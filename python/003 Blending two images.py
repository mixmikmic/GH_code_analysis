get_ipython().magic('pylab inline')
import numpy as np
import cv2

img1 = cv2.imread('../data/images/baseball.jpg', cv2.CV_LOAD_IMAGE_COLOR)
img2 = cv2.imread('../data/images/girl.jpg', cv2.CV_LOAD_IMAGE_COLOR)

img3 = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
pylab.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))

