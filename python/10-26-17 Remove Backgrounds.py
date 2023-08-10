# Load image
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
image_bgr = cv2.imread('images/plane_256x256.jpg')

# Convert to RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Rectange values: start x, start y, width, height
rectangle = (0, 56, 250, 150)

# Create initial mask 
mask = np.zeros(image_rgb.shape[:2], np.uint8)

# Create temporary arrays used by Grabcut
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# run grabCut
cv2.grabCut(image_rgb, mask, rectangle, bgdModel,
           fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Create mask where sure and likely backgrounds set to 0, otherwise 1
mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

# Multiply image with new mask to subtract background
image_rgb_nobg = image_rgb * mask_2[:,:,np.newaxis]

# Show image
plt.imshow(image_rgb_nobg), plt.axis('off')
plt.show()

