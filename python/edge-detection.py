get_ipython().magic('run ../../common/import_all.py')

from common.setup_notebook import set_css_style, setup_matplotlib, config_ipython

import cv2

config_ipython()
setup_matplotlib()
set_css_style()

# Read image with OpenCV
image = cv2.imread('../../imgs/pens.jpg')

# OpenCV reads it in BGR, Matplotlib interprets it in RGB, so created a converted colourspace one
RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# also create a grayscale converted image as Canny wants single-channel input
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Showing both the original and the grayscale image

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.grid()
ax1.imshow(RGB_image)

ax2.grid()
plt.gray()
ax2.imshow(gray)

plt.show();

# Perform a Canny edge detection: can vary the thresholds used to see differences
# the Canny method returns image with edges, we are not changing the default aperture size of the Sobel operator (3)
edged = cv2.Canny(gray, 50, 300)

plt.imshow(edged)
plt.grid()
plt.show();

# find the contours on the edged image
# retrieving only the external ones and without compressing them

img, cnt, hyer = contours_edged = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# drawing contours on original image
# we draw them all (-1), in green ((0, 255, 0)) and with line tickness 3

img = cv2.drawContours(RGB_image, cnt, -1, (0, 255, 0), 3)

plt.imshow(img)
plt.grid()

plt.show();



