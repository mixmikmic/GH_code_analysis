from __future__ import division, print_function
get_ipython().magic('matplotlib inline')

from matplotlib import pyplot as plt, cm
from skimage import io
image = io.imread('images/zebrafish-spinal-cord.png')

from scipy import ndimage as nd
top, bottom = image[[0, -1], :]

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

top_smooth = nd.gaussian_filter1d(top, sigma=20)
ax0.plot(top, color='blue', lw=2)
ax0.plot(top_smooth, color='orange', lw=2)
ax0.set_title('top')

bottom_smooth = nd.gaussian_filter1d(bottom, sigma=20)
ax1.plot(bottom, color='blue', lw=2)
ax1.plot(bottom_smooth, color='orange', lw=2)
ax1.set_title('bottom')

top_mode = top_smooth.argmax()
top_max = top_smooth[top_mode]
top_width = (top_smooth > float(top_max) / 2).sum()

bottom_mode = bottom_smooth.argmax()
bottom_max = bottom_smooth[bottom_mode]
bottom_width = (bottom_smooth > float(bottom_max) / 2).sum()

width = max(bottom_width, top_width)

print(top_mode, top_width, bottom_mode, bottom_width)

from skimage import measure
trace = measure.profile_line(image, (0, top_mode),
                             (image.shape[0] - 1, bottom_mode),
                             linewidth=width,
                             mode='reflect')

plt.plot(trace, color='black', lw=2)
plt.xlabel('position along embryo')
plt.ylabel('mean fluorescence intensity')

