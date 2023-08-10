get_ipython().magic('run ../../common/import_all.py')

import cv2

from common.setup_notebook import set_css_style, setup_matplotlib, config_ipython
config_ipython()
setup_matplotlib()
set_css_style()

# First read an image
image = cv2.imread('../../imgs/pens.jpg')

# Transform into grayscale and into RGB (for Matplotlib)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# show them both
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.grid()
ax1.imshow(RGB_image)

plt.gray()
ax2.grid()
ax2.imshow(gray)

plt.show();

# See image size and number of pixels

'Num of pixels',  image.size
'Size', image.shape

# Hist of the gray image: channel 0, no mask, 256 pixels, range (0, 256)
gray_hist = cv2.calcHist([gray], [0], None, [256], (0, 256))  # the method can do multiple images at a time

# Hists of the color image, each channel, same args
R_hist = cv2.calcHist([RGB_image], [0], None, [256], (0, 256))
G_hist = cv2.calcHist([RGB_image], [1], None, [256], (0, 256))
B_hist = cv2.calcHist([RGB_image], [2], None, [256], (0, 256))

# Plot of the grayscale image hist
plt.plot(hist, c='k')
plt.title('Grayscale image intensity hist')
plt.xlabel('Pixel value')
plt.ylabel('Count')
plt.show();

# Plot of the RGB image hists
plt.plot(R_hist, c='r', label='RED channel')
plt.plot(G_hist, c='g', label='GREEN channel')
plt.plot(B_hist, c='b', label='BLUE channel')

plt.title('Coloured image intensity histograms')
plt.xlabel('Pixel value')
plt.ylabel('Count')
plt.legend()
plt.show();



