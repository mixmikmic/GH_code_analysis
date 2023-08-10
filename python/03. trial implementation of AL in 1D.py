import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pandas as pd

from numpy.polynomial.hermite import hermval
x = np.arange(-6,6,0.1)
h0 = hermval(x, [1, 0, 0])
h1 = hermval(x, [0, 1, 0])
h2 = hermval(x, [0, 0, 1])/10
plt.plot(x, h0)
plt.plot(x, h1)
plt.plot(x, h2)

def gaussian(x, m=0., s=1.0):
    out = 1/(s*np.sqrt(2*np.pi))*np.exp(-(x-m)**2./(2.*s**2.))
    return out / out.sum() / (x[1] - x[0])

gh0 = gaussian(x) * h0
gh1 = gaussian(x) * h1
gh2 = gaussian(x) * h2
plt.plot(x, gh0)
plt.plot(x, gh1)
plt.plot(x, gh2)

from numpy.polynomial.chebyshev import chebval
h0 = chebval(x, [1, 0, 0])
h1 = chebval(x, [0, 1, 0])
h2 = chebval(x, [0, 0, 1])/10
plt.plot(x, h0)
plt.plot(x, h1)
plt.plot(x, h2)

gh0 = gaussian(x) * h0
gh1 = gaussian(x) * h1
gh2 = gaussian(x) * h2
plt.plot(x, gh0)
plt.plot(x, gh1)
plt.plot(x, gh2)

# Parameters from stack
sigGauss = [0.75, 1.5, 3.0]
degGauss = [4, 2, 2]
betaGauss = 2   # in the Becker et al. paper sigGauss is 1 but PSF is more like 2 pixels?
# Parameters from and Becker et al. (2012)
#sigGauss = [0.75, 1.5, 3.0]
#degGauss = [6, 4, 2]

def chebGauss(x, m=0., s=1., ord=0, beta=1.):
    ga = gaussian(x, m, s/beta)
    coef = np.zeros(ord+1)
    coef[-1] = 1
    print s, ord, coef
    ch = chebval(x, coef)
    return ga * ch

basis = [chebGauss(x, m=0, s=sig, ord=deg, beta=betaGauss) for i,sig in enumerate(sigGauss) for deg in range(degGauss[i])]
basis = np.vstack(basis).T  # put the bases into columns
print basis.shape
# basis = pd.DataFrame(basis); basis.plot()
for b in basis.T:
    plt.plot(x, b)

im1 = gaussian(x, m=0.0, s=0.8)  # template
im2 = gaussian(x, m=0.4, s=1.1)  # science image; include a slight registration error
plt.plot(x, im1); plt.plot(x, im2); plt.plot(x, im2-im1)

# Test convolve template with the first basis
tmp = np.convolve(im1, basis[:,0], mode='same')
print im2.shape, basis[:,0].shape, tmp.shape, x.shape
plt.plot(x, im1)
plt.plot(x, basis[:,0])
plt.plot(x, tmp)

# Single call to do it with all bases
basis2 = [np.convolve(im1, b, mode='same') - im1 for b in basis.T]
basis2 = np.vstack(basis2).T 
print basis2.shape

pars = np.linalg.lstsq(basis2, im2)[0]
print pars
fit = (pars * basis2).sum(1)
print basis2.shape, fit.shape
#plt.plot(x, im2 - fit)  # science - convolved template (red)
#plt.plot(x, im2 - im1)  # science - original template (blue)
plt.plot(im1)  # original template (red)
plt.plot(fit)  # convolved template (blue)  -- note looks purple because it's right on top of im2
plt.plot(im2, ls='-.', lw=3)  # science image (dotted, purple)
plt.plot(im2 - fit)  # diffim (grey)

fit = (pars * basis).sum(1)
fit /= fit.sum()
plt.plot(x, fit)  # this plots the matching kernel

conv_im1 = np.convolve(im1, fit, mode='same')
plt.plot(im1)  # original template (red)
plt.plot(conv_im1)  # convolved template (blue)
plt.plot(im2, ls='-.', lw=3)  # science image (dotted, purple)
plt.plot(im2 - conv_im1)  # diffim (grey)



