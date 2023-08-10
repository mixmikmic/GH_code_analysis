# Load image
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as BGR
image_bgr = cv2.imread('images/plane_256x256.jpg', cv2.IMREAD_COLOR)

# Calculate the mean of each channel
channels = cv2.mean(image_bgr)

# Sway blue and red values (making it , not BGR)
observation = np.array([(channels[2], channels[1], channels[0])])

# Show mean channel values
observation

# Show image
plt.imshow(observation), plt.axis('off')
plt.show()

