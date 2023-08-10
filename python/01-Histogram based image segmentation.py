get_ipython().magic('matplotlib inline')
import sys
sys.path.insert(0,'..')
from IPython.display import HTML,Image,SVG,YouTubeVideo
from helpers import header

HTML(header())

Image('http://crashingpatient.com/wp-content/images/part1/tissuedensities.jpg')

from skimage import data
import matplotlib.pyplot as plt
import numpy as np

def norm_hist(ima):
    hist,bins = np.histogram(ima.flatten(),range(256))  # histogram is computed on a 1D distribution --> flatten()
    return 1.*hist/np.sum(hist) # normalized histogram

def display_hist(ima):
    plt.figure(figsize=[10,5])
    if ima.ndim == 2:
        nh = norm_hist(ima)
    else:
        nh_r = norm_hist(ima[:,:,0])
        nh_g = norm_hist(ima[:,:,1])
        nh_b = norm_hist(ima[:,:,2])
    # display the results
    plt.subplot(1,2,1)
    plt.imshow(ima,cmap=plt.cm.gray)
    plt.subplot(1,2,2)
    if ima.ndim == 2:
        plt.plot(nh,label='hist.')
    else:
        plt.plot(nh_r,color='r',label='r')
        plt.plot(nh_g,color='g',label='g')
        plt.plot(nh_b,color='b',label='b')
    plt.legend()
    plt.xlabel('gray level');

display_hist(data.horse())

display_hist(data.checkerboard())

display_hist(data.page())

display_hist(data.coins())

display_hist(data.camera())

display_hist(data.hubble_deep_field()[:,:,0])

display_hist(data.moon())

display_hist(data.coffee())

ima = data.hubble_deep_field()[:,:,0]
display_hist(ima)

perc = .95 #galaxies are the 5% brighter pixels in the image

h,bins = np.histogram(ima[:],range(256))

c = 1.*np.cumsum(h)/np.sum(h)
plt.plot(c)

plt.gca().hlines(perc,0,plt.gca().get_xlim()[1]);

th_perc = np.where(c>=perc)[0][0]

plt.gca().vlines(th_perc,0,plt.gca().get_ylim()[1]);
plt.figure()
plt.imshow(ima>=th_perc);

from skimage.filters import threshold_otsu

ima = 255*data.imread('https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Monocyte_no_vacuoles.JPG/712px-Monocyte_no_vacuoles.JPG',as_grey=True)
th = threshold_otsu(ima)
display_hist(ima)
plt.gca().vlines(th,0,plt.gca().get_ylim()[1]);
plt.title("Otsu's thresold = %d"%th );

ima = data.astronaut()[:,:,0]

h,_ = np.histogram(ima[:],range(256),normed=True)

def entropy(symb,freq):
    idx = np.asarray(freq) != 0
    s = np.sum(freq)
    p = 1. * freq/s
    e = - np.sum(p[idx] * np.log2(p[idx]))
    return e

e0 = []
e1 = []

for g in range(1,255):
    e0.append(entropy(range(0,g),h[:g]))
    e1.append(entropy(range(g,255),h[g:255]))

e0 = np.asarray(e0)
e1 = np.asarray(e1)

e_max = np.argmax(e0+e1)
plt.plot(e0)
plt.plot(e1)
plt.plot(e0+e1)

plt.figure()
plt.plot(h)
plt.gca().vlines(e_max,0,plt.gca().get_ylim()[1]);
plt.figure()
plt.imshow(ima>e_max);

from skimage.filters import threshold_otsu
from skimage.color import rgb2hsv

rgb = data.immunohistochemistry()

r = rgb[:,:,0]
g = rgb[:,:,1]
b = rgb[:,:,2]

th_r = threshold_otsu(r)
th_g = threshold_otsu(g)
th_b = threshold_otsu(b)

mix = (r>th_r).astype(np.uint8) + 2 * (g>th_g).astype(np.uint8) + 4 * (b>th_b).astype(np.uint8)

plt.figure(figsize=[8,6])
plt.imshow(rgb)

plt.figure(figsize=[8,4])
plt.subplot(1,3,1)
plt.imshow(r>th_r)
plt.subplot(1,3,2)
plt.imshow(g>th_g)
plt.subplot(1,3,3)
plt.imshow(b>th_b)

plt.figure(figsize=[8,6])
plt.imshow(mix)
plt.colorbar();


import skimage.filter.rank as skr
from skimage.morphology import disk

ima = (128*np.ones((255,255))).astype(np.uint8)
ima[50:100,50:150] = 150
ima[130:200,130:200] = 100

ima = skr.mean(ima,disk(10))

mask = 10.*np.ones_like(ima)
mask[:,100:180] = 15
# add variable noise

n = np.random.random(ima.shape)-.5
ima = (ima + mask*n).astype(np.uint8)


entropy = 255.*skr.entropy(ima,disk(4))/8

plt.figure(figsize=[8,8])
plt.imshow(ima,cmap=plt.cm.gray);

display_hist(ima)


display_hist(entropy)

r = np.zeros_like(ima)
r[ima>110] = 1
r[ima>140] = r[ima>140]+1

plt.imshow(r)
plt.colorbar();

s = np.zeros_like(entropy)
s[entropy>110] = 1
s[entropy>140] = 0
plt.imshow(s)
plt.colorbar();



