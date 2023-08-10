get_ipython().magic('matplotlib inline')
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = 8, 8

pl_img = Image.open('1.jpg')
plt.imshow(np.array(pl_img));

def plot(img, **kwargs):
    plt.imshow(np.asarray(img), **kwargs)
    
plot(pl_img)

plot(pl_img.rotate(45))

img = np.array(pl_img)
print(img.shape, img.dtype)

a = img.flatten()
a[2::3]

plt.hist(img.flatten()[0::3], 128, color='r');
plt.hist(img.flatten()[1::3], 128, color='g');
plt.hist(img.flatten()[2::3], 128, color='b');

gray = np.array(pl_img.convert('L'))
plt.gray()
plot(gray)

plot(pl_img)

# invert image
plot(255 - gray)

import matplotlib.colors as colors
# clamp between 100,200
plot(gray* (100./255) + 100, norm=colors.Normalize(0, 255));

clamp = gray.copy()
clamp[clamp < 50] = 255
clamp[clamp > 150] = 255
plot(clamp, norm=colors.Normalize(0, 255))

def histogram_eq(img, bins=256):
    flat = img.flatten()
    hist, bins = np.histogram(flat, bins, normed=True)
    cdf = hist.cumsum()
    cdf = 255*cdf / cdf[-1]
    im = np.interp(flat, bins[:-1], cdf)
    return im.reshape(img.shape)

plot(histogram_eq(gray, 12))
plt.figure()
plot(histogram_eq(gray, 256))   

from scipy.ndimage import filters

im = filters.gaussian_filter(img, sigma=5)
plot(im)

plot(filters.gaussian_filter(img, sigma=15))

plot(filters.gaussian_filter(img, sigma=2))

gray32 = gray.astype('int32')  # need higher precision
imx = filters.sobel(gray32, 1)
imy = filters.sobel(gray32, 0)
mag = np.hypot(imx, imy)
mag *= 255.0 / np.max(mag)
plot(mag)

plot(imy)

plot(imx)

std = 3
imx = filters.gaussian_filter(gray32, sigma=(std, std), order=(0, 1))
imy = filters.gaussian_filter(gray32, sigma=(std, std), order=(1, 0))
mag = np.hypot(imx, imy)
#mag *= 255.0 / np.max(mag)
plot(mag)

from scipy.ndimage import measurements, morphology


im2=(gray<128)*1
labels, N = measurements.label(im2)
N

#labels[labels>0]=255
plot(labels)

