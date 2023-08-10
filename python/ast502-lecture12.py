#curse of dimensionality
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import scipy.special as sp
from matplotlib import pyplot as plt

def fractionVol(dimension):
    return (np.pi**(dimension/2.))/(dimension*2.**(dimension-1.)*sp.gamma(dimension/2.))

dim = np.linspace(1,100)

#------------------------------------------------------------
# Plot the results
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dim,fractionVol(dim))

ax.set_yscale('log')
ax.set_xlabel('$Dimension$')
ax.set_ylabel('$Fractional Volume$')

plt.show()

#Example, using ellipse from before
import numpy as np
from sklearn.decomposition import PCA

##make ellipse
np.random.seed(42)
r = 0.9

sigma1 = 0.25
sigma2 = 0.08
rotation = np.pi / 6
s = np.sin(rotation)
c = np.cos(rotation)

X = np.random.normal(0, [sigma1, sigma2], size=(100, 2)).T
R = np.array([[c, -s],[s, c]])
X = np.dot(R, X)  ## 2D dimensional ellipse 

## prepare data for sk.learn
x = X[0]
y = X[1]
X_sk = np.c_[x, y] ##transpose of X; format for sk learn

#if you want to project onto higher dimension..
#R2 = np.random.random((2,100)) # projection matrix
#Y = np.dot(X_sk,R2) # X is now 10-dim, with 2 intrinsic dims

#Do the PCA computation
pca = PCA(n_components=2) # n_components can be optionally set
pca.fit(X_sk) 

##some of the outputs
mean = pca.mean_ # mean along each direction
print 'mean coordinates'
print mean
print 'eigenvalues (variance)'
print pca.explained_variance_
print 'relative variance of each eigenvector'
print pca.explained_variance_ratio_
V = pca.components_ # Eigenvectors
print 'principal components'
print V

axs1x = [-1*V[0][0], 1*V[0][0]] # for plotting eigenvectors
axs1y = [-1*V[0][1], 1*V[0][1]]
axs2x = [-0.5*V[1][0], 0.5*V[1][0]]
axs2y = [-0.5*V[1][1], 0.5*V[1][1]]

plt.subplot(1, 2, 1)
plt.plot(x, y, 'o')
plt.plot(axs1x, axs1y, 'k')
plt.plot(axs2x, axs2y, 'k')

###  when there is a mean not factored out...

X_mean = X_sk + 1 
pca2 = PCA(n_components = 2)
pca2.fit(X_mean)

#When the mean is not factored out...add the mean vector to the principal componenets
#mean becomes the first principle component; others move down a level

#print pca2.explained_variance_ratio_
V2 = pca2.components_ # 4x10 matrix of components
print 'principal components without mean_sub'
print V2
axs1x2 = [-1*V2[0][0], 1*V2[0][0]] # for plotting eigenvectors
axs1y2 = [-1*V2[0][1], 1*V2[0][1]]
axs2x2 = [-0.5*V2[1][0], 0.5*V2[1][0]]
axs2y2 = [-0.5*V2[1][1], 0.5*V2[1][1]]

plt.subplot(1,2,2)
plt.plot(x+1, y+1, 'o')
plt.plot(axs1x2, axs1y2, 'k')
plt.plot(axs2x2, axs2y2, 'k')


plt.tight_layout()
plt.show()

get_ipython().run_line_magic('matplotlib', 'inline')
#EXAMPLE FOR A set of spectra

# Example from Andy Connolly
# See Ivezic, Figure 7.4
import numpy as np
from matplotlib import pyplot as plt

import os

from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA

from astroML.datasets import sdss_corrected_spectra
from astroML.decorators import pickle_results

#------------------------------------------------------------
# Download data
#data = sdss_corrected_spectra.fetch_sdss_corrected_spectra()

# if the download fails
# type in your internet browser: http://staff.washington.edu/jakevdp/spec4000.npz
# move data into the directory you are working in

npz_name = 'spec4000.npz'

def fetch_spectra_manual(npz_name):
    '''
    use if data isn't downloading
    '''   
    data = np.load(npz_name)

    data_dict = dict([(key, data[key]) for key in data.files])
    np.savez(npz_name, **data_dict)
    
    return data
data = fetch_spectra_manual(npz_name)
#data 
spectra = sdss_corrected_spectra.reconstruct_spectra(data)
wavelengths = sdss_corrected_spectra.compute_wavelengths(data)
#print len(spectra), len(wavelengths) ##4000 spectra 

#----------------------------------------------------------------------

##prepare for PCA
np.random.seed(500)
nrows = 2000 # We'll just look at 2000 random spectra
n_components = 5 # Do the fit with 5 components, which is the mean plus 4
##the line above means to stop the PCA computation after 5 eigenvectors are recorded.

ind = np.random.randint(spectra.shape[0], size=nrows) # narrow to 2000 spec
spec_mean = spectra[ind].mean(0) # Compute the mean spectrum, which is the first component
# spec_mean = spectra[:50].mean(0)
#print spec_mean.shape

# use Randomized setting for speed
pca = PCA(n_components=n_components-1, svd_solver='randomized')
pca.fit(spectra[ind])  ## perform PCA analysis on 
pca_comp = np.vstack([spec_mean,pca.components_]) #Add the mean to the components
evals = pca.explained_variance_ratio_ 
print evals # Print the eigenvalues

get_ipython().run_line_magic('matplotlib', 'inline')
#Make plots
fig = plt.figure(figsize=(10, 8))
fig.subplots_adjust(left=0.05, right=0.95, wspace=0.05,
                    bottom=0.1, top=0.95, hspace=0.05)

titles = 'PCA components'

for j in range(n_components):
    
    # plot the components
    ax = fig.add_subplot(n_components, 2, 2*j+2)
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.xaxis.set_major_locator(plt.MultipleLocator(1000))
    if j < n_components - 1:
        ax.xaxis.set_major_formatter(plt.NullFormatter())
    else:
        ax.set_xlabel('wavelength (Angstroms)')
    ax.plot(wavelengths, pca_comp[j], '-k', lw=1)

    # plot zero line
    xlim = [3000, 7999]
    ax.plot(xlim, [0, 0], '-', c='gray', lw=1)
    ax.set_xlim(xlim)

    # adjust y limits
    ylim = plt.ylim()
    dy = 0.05 * (ylim[1] - ylim[0])    
    ax.set_ylim(ylim[0] - dy, ylim[1] + 4 * dy)
    
    # plot the first j spectra
    ax2 = fig.add_subplot(n_components, 2, 2*j+1)
    ax2.yaxis.set_major_formatter(plt.NullFormatter())
    ax2.xaxis.set_major_locator(plt.MultipleLocator(1000))
    if j < n_components - 1:
        ax2.xaxis.set_major_formatter(plt.NullFormatter())
    else:
        ax2.set_xlabel('wavelength (Angstroms)')
    ax2.plot(wavelengths, spectra[j], '-k', lw=1)
    
    # plot zero line
    ax2.plot(xlim, [0, 0], '-', c='gray', lw=1)
    ax2.set_xlim(xlim)

    if j == 0:
        ax.set_title(titles, fontsize='medium')

    if j == 0:
        label = 'mean'
    else:
        label = 'component %i' % j

    # adjust y limits
    ylim = plt.ylim()
    dy = 0.05 * (ylim[1] - ylim[0])    
    ax2.set_ylim(ylim[0] - dy, ylim[1] + 4 * dy)


    ax.text(0.02, 0.95, label, transform=ax.transAxes,
            ha='left', va='top', bbox=dict(ec='w', fc='w'),
            fontsize='small')


plt.show()

# Scree plot
import numpy as np
from matplotlib import pyplot as plt

print 'evals'
print evals
#----------------------------------------------------------------------
# Plot the results
fig = plt.figure(figsize=(10, 5))

n_components = 5

ax = fig.add_subplot(121)
ax.set_yscale('log')
#ax.plot(np.arange(n_components-1), evals, 'o-', markersize=10)
ax.plot([1, 2, 3, 4], evals, 'o-', markersize=10)
ax.set_xlabel("eigenvalue number")
ax.set_ylabel("eigenvalue ")

ax = fig.add_subplot(122)
#ax.plot(np.arange(n_components-1), evals.cumsum(), 'o-', markersize=10)
ax.plot([1, 2, 3, 4],evals.cumsum(), 'o-', markersize=10)
ax.set_xlabel("eigenvalue number")
ax.set_ylabel("cumulative eigenvalue")

print("The first component explains {:.3f} of the variance in the data.".format(evals[0]))
print("The second component explains {:.3f} of the variance in the data.".format(evals[1]))
print("All components explain {:.3f} of the variance in the data.".format(np.sum(evals)))

plt.show()

#image reconstruction example
import numpy as np
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA

from astroML.datasets import sdss_corrected_spectra
from astroML.decorators import pickle_results

#------------------------------------------------------------
# Download data
#data = sdss_corrected_spectra.fetch_sdss_corrected_spectra()
spectra = sdss_corrected_spectra.reconstruct_spectra(data)
wavelengths = sdss_corrected_spectra.compute_wavelengths(data)

#------------------------------------------------------------
# Compute PCA components

# Eigenvalues can be computed using PCA as in the commented code below:

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(spectra)
evals = pca.explained_variance_ratio_
evals_cs = evals.cumsum()
evecs = pca.components_


#  because the spectra have been reconstructed from masked values, this
#  is not exactly correct in this case: we'll use the values computed
#  in the file compute_sdss_pca.py
#evals = data['evals'] ** 2
#evals_cs = evals.cumsum()
#evals_cs /= evals_cs[-1]
#evecs = data['evecs']
spec_mean = spectra.mean(0)

#------------------------------------------------------------
# Find the coefficients of a particular spectrum
spec = spectra[1]
coeff = np.dot(evecs, spec - spec_mean)

#------------------------------------------------------------
# Plot the sequence of reconstructions
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(hspace=0)

for i, n in enumerate([0, 4, 8, 20]):
    ax = fig.add_subplot(411 + i)
    ax.plot(wavelengths, spec, '-', c='gray')
    ax.plot(wavelengths, spec_mean + np.dot(coeff[:n], evecs[:n]), '-k')

    if i < 3:
        ax.xaxis.set_major_formatter(plt.NullFormatter())

    ax.set_ylim(-2, 21)
    ax.set_ylabel('flux')

    if n == 0:
        text = "mean"
    elif n == 1:
        text = "mean + 1 component\n"
        text += r"$(\sigma^2_{tot} = %.2f)$" % evals_cs[n - 1]
    else:
        text = "mean + %i components\n" % n
        text += r"$(\sigma^2_{tot} = %.2f)$" % evals_cs[n - 1]

    ax.text(0.01, 0.95, text, ha='left', va='top', transform=ax.transAxes)

fig.axes[-1].set_xlabel(r'${\rm wavelength\ (\AA)}$')
plt.show()

# missing data example
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker

from astroML.datasets import fetch_sdss_corrected_spectra
from astroML.datasets import sdss_corrected_spectra

#------------------------------------------------------------
# Get spectra and eigenvectors used to reconstruct them
#data = fetch_sdss_corrected_spectra()
spec = sdss_corrected_spectra.reconstruct_spectra(data)
lam = sdss_corrected_spectra.compute_wavelengths(data)
evecs = data['evecs']
mu = data['mu']
norms = data['norms']
mask = data['mask']

#------------------------------------------------------------
# plot the results
i_plot = ((lam > 5750) & (lam < 6350))
lam = lam[i_plot]

specnums = [20, 8, 9]
subplots = [311, 312, 313]

fig = plt.figure(figsize=(8, 10))
fig.subplots_adjust(hspace=0)

for subplot, i in zip(subplots, specnums):
    ax = fig.add_subplot(subplot)

    # compute eigen-coefficients
    spec_i_centered = spec[i] / norms[i] - mu
    coeffs = np.dot(spec_i_centered, evecs.T)

    # blank out masked regions
    spec_i = spec[i]
    mask_i = mask[i]
    spec_i[mask_i] = np.nan

    # plot the raw masked spectrum
    ax.plot(lam, spec_i[i_plot], '-', color='k', lw=2,
            label='True spectrum')

    # plot two levels of reconstruction
    for nev in [10]:
        if nev == 0:
            label = 'mean'
        else:
            label = 'N EV=%i' % nev
        spec_i_recons = norms[i] * (mu + np.dot(coeffs[:nev], evecs[:nev]))
        ax.plot(lam, spec_i_recons[i_plot], label=label)

    # plot shaded background in masked region
    ylim = ax.get_ylim()
    mask_shade = ylim[0] + mask[i][i_plot].astype(float) * ylim[1]
    plt.fill(np.concatenate([lam[:1], lam, lam[-1:]]),
             np.concatenate([[ylim[0]], mask_shade, [ylim[0]]]),
             lw=0, fc='k', alpha=0.2)

    ax.set_xlim(lam[0], lam[-1])
    ax.set_ylim(ylim)
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    if subplot == 311:
        ax.legend(loc=1, prop=dict(size=14))

    ax.set_xlabel('$\lambda\ (\AA)$')
    ax.set_ylabel('normalized flux')

plt.show()

