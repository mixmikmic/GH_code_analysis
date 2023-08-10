import os
import numpy as np
import scipy.fftpack as fp
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# Functions to go from image to frequency-image and back
im2freq = lambda data: fp.rfft(fp.rfft(data, axis=0), axis=1)
freq2im = lambda f: fp.irfft(fp.irfft(f, axis=1), axis=0)

IMG = 'n01515078_1185.JPEG'
PATH = './bird_img'

#plt.close('all')
#fig, axes = plt.subplots(nrows=1, ncols=1)
#axes.get_xaxis().set_visible(False)
#axes.get_yaxis().set_visible(False)
#axes[1].get_xaxis().set_visible(False)
#axes[1].get_yaxis().set_visible(False)
data = plt.imread(os.path.join(PATH, IMG))

plt.imshow(data)

data.shape
h = data.shape[0]
w = data.shape[1]

mask = np.ones_like(data)
mask.shape
plt.imshow(mask[:,:,0], cmap='gray')

'''
You must use integers when indexing matrices or arrays, 
hopefully this is obvious. Double slash '//' in Python 3
requests that integer division be performed.
'''
print(h / 5)
print(int(h / 5)) # e.g in Python 2
print(h // 5)

'''
Aside: 
It's good practice to define variables for numbers you want to 
reuse and possibly change when experimenting. Otherwise, we 
call them 'magic numbers', because they magically worked 
(at one point in time), but you can't remember why. 
'''
D_QUAD = 4 

print("Mask off pixels %d to %d along vertical, and %d to %d along horizontal." % 
      ((h // D_QUAD), (h * (D_QUAD - 1) // D_QUAD), (w // D_QUAD), (w * (D_QUAD - 1) // D_QUAD)))

# First, lets mask-off a square region in the center of the image
mask[h // D_QUAD : h * (D_QUAD - 1) // D_QUAD, w // D_QUAD : w * (D_QUAD - 1) // D_QUAD, :] = 0

# check that the previous command did something reasonable
total_zeros = np.sum(mask == 0) # mask == 0 returns an element-wise boolean-mask, then np.sum() counts them up
total_ones = np.sum(mask == 1)
print("This mask will zero %.2f%% of the pixels in data" % (100 * total_zeros / total_ones))

'''
Visualize our new mask. Only plot one channel in mask
because [0, 0, 0] or [1, 1, 1] both appear black 
in matplotlib.
'''
plt.imshow(mask[:,:,0], cmap='gray')
plt.show()

THIRD = 3
mask[h // THIRD : h * (THIRD - 1) // THIRD, w // THIRD: w * (THIRD - 1) // THIRD, :] = 1

# check that the previous command did something reasonable
total_zeros = np.sum(mask == 0)
total_ones = np.sum(mask == 1)
print("This mask will zero %.2f%% of the pixels in data" % (100 * total_zeros / total_ones))

plt.imshow(mask[:,:,0], cmap='gray')
plt.show()

freq = im2freq(data)

'''
The frequency response of natural images follows a power law decay. 
'''
plt.imshow(freq**2)

plt.imshow(freq * mask)

img = freq2im(freq)

plt.imshow(img[:,:,2] + img[:,:,1] + img[:,:,0])

img_f = freq2im(freq * mask)

plt.imshow(img_f[:,:,2] + img_f[:,:,1] + img_f[:,:,0])

plt.imshow(img_f)

zeros = np.sum(mask == 0)
ones = np.sum(mask == 1)
zeros / ones

plt.imshow(mask)

plt.imshow(freq*mask)



