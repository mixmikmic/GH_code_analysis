import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
# notebook

import detection
import imageProc
import utils

da = utils.Data()
da.read()

utils.mtv(da, b=10, alpha=0.8)
xlim, ylim = (80, 400), (100, 400)

plt.xlim(xlim); plt.ylim(ylim)
plt.show()

utils.mtv(da, b=10, alpha=0.0, fig=2)
plt.xlim(xlim); plt.ylim(ylim)
plt.show()

raw = utils.Data()
raw.read(readRaw=True)

utils.mtv(raw, b=10, alpha=0.3)
plt.xlim(740, 810); plt.ylim(230, 290)
plt.show()

def gaussian2D(beta):
    size = int(3*abs(beta) + 1)
    x, y = np.mgrid[-size:size+1, -size:size+1]
    phi = np.exp(-(x**2 + y**2)/(2*beta**2))
    phi /= phi.sum()

    return phi

def convolveWithGaussian(image, beta):
    phi = gaussian2D(beta)

    return scipy.signal.convolve(image, phi, mode='same')

# %%timeit -n 1 -r 1

sda = da.copy()
beta = 2.5
sda.image = convolveWithGaussian(sda.image, beta)

utils.mtv(sda.image)

phi = gaussian2D(beta)
n_eff = 1/np.sum(phi**2)
print "n_eff = %.3f (analytically: %.3f)" % (n_eff, 4*pi*beta**2)

def convolveWithGaussian(image, beta):
    def gaussian1D(beta):
        size = int(3*abs(beta) + 1)
        x = np.arange(-size, size+1)
        phi = np.exp(-x**2/(2*beta**2))
        phi /= phi.sum()

        return phi

    beta = 2.5
    phi = gaussian1D(beta)

    for y in range(0, image.shape[0]):
        image[y] = scipy.signal.convolve(image[y], phi, mode='same')
    for x in range(0, image.shape[1]):
        image[:, x] = scipy.signal.convolve(image[:, x], phi, mode='same')
        
    return image

nsigma = 3.5
threshold = nsigma*sqrt(np.median(sda.variance)/n_eff)
footprints = detection.findObjects(sda.image, threshold, grow=3)

print "I found %d objects" % (len(footprints))

nShow = 10
for foot in footprints.values()[0:nShow]:
    print "(%5d, %5d) %3d" % (foot.centroid[0], foot.centroid[1], foot.npix)
if len(footprints) > nShow:
    print "..."

sda.clearMaskPlane("DETECTED")
detection.setMaskFromFootprints(sda, footprints, "DETECTED")

utils.mtv(sda)

plt.xlim(xlim); plt.ylim(ylim)
plt.show()

da.clearMaskPlane("DETECTED")
detection.setMaskFromFootprints(da, footprints, "DETECTED")

utils.mtv(da, alpha=0.3)
plt.xlim(xlim); plt.ylim(ylim)
plt.show()

t = utils.Data(image=da.truth, mask=sda.mask)
utils.mtv(t, I0=1, b=0.01, alpha=0.6)
plt.xlim(xlim); plt.ylim(ylim)
plt.show()

import scipy.special

pixelSize = 0.200

nPerPsf = 0.5*scipy.special.erfc(nsigma/sqrt(2))
nPerDeg = nPerPsf*3600**2/0.5

print "False positives per degree: %d  In data: %d" % (
    nPerDeg, nPerDeg/(3600/(da.image.shape[0]*pixelSize))**2)

# %%timeit -n 1 -r 1

detection = reload(detection)

ndeg = 1.0/2.0                       # Size of image we'll simulate (in degrees)
size = int(3600*ndeg/pixelSize)      # Size of image we'll simulate (in pixels)
im = np.zeros((size, size))

nsigma, Poisson= 5, False
np.random.seed(667)
sigma = 10
if Poisson:
    mu = sigma**2
    im += np.random.poisson(lam=mu, size=size*size).reshape(size, size) - mu
else:
    im += np.random.normal(scale=sigma, size=size*size).reshape(size, size)

sim = convolveWithGaussian(im, beta)
n_eff = 4*pi*beta**2   # Effective area of PSF

threshold = nsigma*sigma/sqrt(n_eff)
footprints = detection.findObjects(sim, threshold, grow=0)
print "%s %g %d %.1f" % (("Poisson" if Poisson else "Gaussian"), nsigma,                       len(footprints)/ndeg**2,                       3600**2*1/(2**2.5*pi**1.5*(beta*pixelSize)**2)*nsigma*exp(-nsigma**2/2))

if not False:    
    tmp = utils.Data(sim)
    tmp.clearMaskPlane("DETECTED")
    detection.setMaskFromFootprints(tmp, footprints, "DETECTED")

    utils.mtv(tmp, alpha=1)

