get_ipython().magic('run ../../common/import_all.py')

import cv2

from common.setup_notebook import set_css_style, setup_matplotlib, config_ipython
config_ipython()
setup_matplotlib()
set_css_style()

# Read the image (the pens one)
image = cv2.imread('../../imgs/pens.jpg')

# make it grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# invert it
inverted = 255 - gray

# Plot both gray image and inverted

plt.gray()
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.grid()
ax1.imshow(gray)

ax2.grid()
ax2.imshow(inverted)

plt.show();



