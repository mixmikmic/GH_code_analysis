get_ipython().magic('matplotlib inline')
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os.path
print("OpenCV Version : %s " % cv2.__version__)

def display(path):
    image = cv2.imread(path)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

# there are only 390 sample images, so keep trying random images
# until we get one
path = ""
while not os.path.isfile(path):
    path = "RealFullField/%s.jpg" % np.random.randint(0, 543)

# ok, got one
print(path)
display(path)



