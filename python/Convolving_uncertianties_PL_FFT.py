get_ipython().magic('matplotlib inline')
from collections import namedtuple
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
import scipy.stats as stats
from scipy.interpolate import interp1d
import scipy
import statsmodels.api as sm
from scipy import signal
from numpy.fft import fft, ifft, ifftshift
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit


bigfontsize=20
labelfontsize=16
tickfontsize=16
sns.set_context('talk')
plt.rcParams.update({'font.size': bigfontsize,
                     'axes.labelsize':labelfontsize,
                     'xtick.labelsize':tickfontsize,
                     'ytick.labelsize':tickfontsize,
                     'legend.fontsize':tickfontsize,
                     })

full_width_of_PL = np.sqrt(12.)*0.035
center_of_PL = -full_width_of_PL / 2
intrinsic = stats.uniform(loc=center_of_PL,scale=full_width_of_PL)
errscale = 0.049
photometric = stats.norm(loc=0,scale=errscale)

## I'm not including a metallicity term at this point, becuase we haven't worked it out yet.

# NB Kernel support array **MUST** be symmetric about centre of the kernel (error PDF) for this to work right. 
# Support also needs to extend about any significant areas of the component PDFs.
# Here, we just define one massive support for both input PDF, and error PDF (kernel)
# But we can do much better (see later)

#NB step-size determines precision of approximation
delta = 1e-3
big_grid = np.arange(-.5,.5,delta)

# Cannot analytically convolve continuous PDFs, in general.
# So we now make a probability mass function on a fine grid 
# - a discrete approximation to the PDF, amenable to FFT...
pmf1 = intrinsic.pdf(big_grid)*delta
pmf2 = photometric.pdf(big_grid)*delta
conv_pmf = scipy.signal.fftconvolve(pmf1,pmf2,'same') # Convolved probability mass function
print "Grid length, sum(gauss_pmf), sum(uni_pmf),sum(conv_pmf):"
print len(big_grid), sum(photometric.pdf(big_grid)*delta), sum(intrinsic.pdf(big_grid)*delta), sum(conv_pmf)
conv_pmf = conv_pmf/sum(conv_pmf)

plt.plot(big_grid,pmf1, label='Intrinsic PL relation')
plt.plot(big_grid,pmf2, label='Gaussian photometric errors')
plt.plot(big_grid,conv_pmf, label='Convolution')
plt.xlim(-0.6,max(big_grid))
plt.legend(loc='best'), plt.suptitle('PL relation convolved PMFs - no metallicity term')

full_width_of_PL = np.sqrt(12.)*0.035
center_of_PL = -full_width_of_PL / 2.
intrinsic = stats.uniform(loc=center_of_PL,scale=full_width_of_PL)
errscale = 0.049
photometric = stats.norm(loc=0.0,scale=errscale)

observed_PL_1sigma = 0.0843634290098
observed_PL_relation = stats.norm(loc=0,scale=observed_PL_1sigma)

#NB step-size determines precision of approximation
delta = 1e-3
big_grid = np.arange(-.3,.3,delta)


## Setting up the PDFs and PMFs for the components
pmf_o = observed_PL_relation.pdf(big_grid)*delta
pmf_i = intrinsic.pdf(big_grid)*delta
pmf_p = photometric.pdf(big_grid)*delta

#print pmf_o, pmf_i, pmf_p
#print len(pmf_o), len(pmf_i), len(pmf_p)

conv_pmf = scipy.signal.fftconvolve(pmf_i,pmf_p,'same')
conv_pmf = conv_pmf/sum(conv_pmf)

plt.plot(big_grid,pmf_o, label='Observed PL relation')
plt.plot(big_grid,pmf_i, label='Intrinsic PL relation')
plt.plot(big_grid,pmf_p, label='Photometric Uncertainties')
plt.plot(big_grid, conv_pmf, label='Intrinsic + Phot')
plt.xlim(min(big_grid),max(big_grid))
plt.ylim(0,0.01)
plt.legend(loc='best'), plt.suptitle('Observed PL relation and known components')


0 - full_width_of_PL/2., center_of_PL





known_pmf = scipy.signal.fftconvolve(pmf_i,pmf_p,'same')
obs_pmf = observed_PL_relation.pdf(big_grid)*delta

plt.plot(big_grid, known_pmf, label="Known components")
plt.plot(big_grid, obs_pmf, label="Observed PL")
plt.legend(loc='best'), plt.suptitle('Observed PL relation and known components')


## Deconvolution is hard.

## Can I brute force this??


## Take the fourier transform of the known and observed components
#np.fft.fftshift(np.abs(np.fft.fft(y)))/ np.sqrt(2 * N)
known_fft = np.fft.fftshift(np.abs(np.fft.fft(known_pmf)))/np.sqrt(2*len(big_grid))
obs_fft = np.fft.fftshift(np.abs(np.fft.fft(obs_pmf)))/np.sqrt(2*len(big_grid))

ratio = obs_fft / known_fft
norm_ratio = ratio / sum(ratio)

plt.plot(big_grid, known_fft, label='known_fft')
plt.plot(big_grid, obs_fft, label="obs_fft")

#plt.xlim(-0.1, 0.1)
plt.plot(big_grid, (norm_ratio), label='ratio')
title_text = 'Fourier transforms'
plt.legend(loc='best'), plt.suptitle(title_text)

max(ratio), min(ratio), big_grid[np.where(ratio==max(ratio))], big_grid[np.where(ratio==min(ratio))]

#plt.xlim(-0.1, 0.1)
plt.plot(big_grid, (ratio), label='ratio')

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

p0 = [3000., 0., 0.05]

coeff, var_matrix = curve_fit(gauss, big_grid, ratio, p0=p0)

# Get the fitted curve
hist_fit = gauss(big_grid, *coeff)

plt.plot(big_grid, ratio, label='Ratio of FFTs')
plt.plot(big_grid, hist_fit, label='Fitted Gaussian')
#plt.xlim()
print coeff




## KDE is taking forever. Trying polynomial fitting instead
## Gaussian mixture models
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html

#normalised_ratio = ratio / sum(ratio)

density = gaussian_kde(ratio)

xs = np.arange(-0.3, 0.3, 0.01)
ys = density.evaluate(xs)

#ys = ys / sum(ys)

plt.plot(big_grid, ratio, label='ratio')
title_text = 'Fourier transforms'
plt.plot(xs, ys, antialiased=True, linewidth=2, color="#A81450",label='KDE')
plt.fill_between(xs, ys, alpha=.5, zorder=5, antialiased=True,                  color="#E01B6A")
#plt.ylim(0,0.02)
plt.legend(loc='best'), plt.suptitle(title_text)
        
 





len(ratio)

## Now take the inverse fourier transform of the ratio of the observed and known components

# Have to shift the ifft also
metal_pmf = np.fft.fftshift(np.abs(np.fft.ifft(ratio)))/np.sqrt(2*len(big_grid))

norm_metal = metal_pmf / sum(metal_pmf)

plt.plot(big_grid,norm_metal, label="Metallicity term")

title_text = 'ifft of ratio - metallicity component'
plt.legend(loc='best'), plt.suptitle(title_text)
    

density_z = gaussian_kde(metal_pmf)
density_z.covariance_factor = lambda : 0.05
density_z._compute_covariance()


xs_z = np.arange(-0.5, 0.5, 0.01)
ys_z = density(xs_z)

ys_z = ys_z / sum(ys_z)

plt.plot(big_grid,pmf_o, label='Observed PL relation')
plt.plot(big_grid,pmf_i, label='Intrinsic PL relation')
plt.plot(big_grid,pmf_p, label='Photometric Uncertainties')
#plt.plot(xs_z, ys_z, label='Metallicity KDE')


title_text = 'ifft of ratio - metallicity component'
plt.legend(loc='best'), plt.suptitle(title_text)
plt.xlim(min(big_grid),max(big_grid))
#plt.ylim(0,0.01)

#convolve the ifft with a gaussian to smooth it?
normalised_gaussian = stats.norm(loc=0,scale=1.0)
ng_pmf = normalised_gaussian.pdf(big_grid)*delta
metal_gauss = scipy.signal.fftconvolve(metal_pmf,ng_pmf, 'same')
metal_gauss = metal_gauss / sum(metal_gauss)
plt.plot(big_grid,metal_gauss, label="Metallicity term")



##metal_pmf = metal_pmf / np.sum(metal_pmf)

## Convolve the components back together to see if it makes any sense

model_pmf = scipy.signal.fftconvolve(known_pmf, metal_gauss, 'same')

plt.plot(big_grid,obs_pmf, label='Observed PL relation')
plt.plot(big_grid, known_pmf, label='Known components')
#plt.plot(big_grid,pmf_i, label='Intrinsic PL relation')
#plt.plot(big_grid,pmf_p, label='Gaussian photometric errors')
#plt.plot(big_grid,norm_metal, label="Metallicity term")
plt.plot(big_grid, model_pmf, label="Best Model")


title_text = 'Fourier deconvolution'
plt.legend(loc='best'), plt.suptitle(title_text)






obs_pmf



