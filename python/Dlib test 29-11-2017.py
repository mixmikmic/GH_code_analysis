import sys
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import dlib
from skimage import io

file_name = 'demo_3.jpg'
img = plt.imread(file_name)
plt.imshow(img)
plt.show()

detector = dlib.get_frontal_face_detector()

dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))

file_name = 'many-faces.jpg'
img = plt.imread(file_name)
plt.imshow(img)
plt.show()

dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))



