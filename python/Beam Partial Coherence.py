get_ipython().magic('pylab')
get_ipython().magic('matplotlib inline')
import SimMLA.fftpack as simfft
import SimMLA.grids   as grids

# Create a grid to sample the input field and build the coherence function on
numLenslets  = 21    # Must be odd; corresponds to the number of lenslets in one dimension
lensletSize  = 500   # microns
focalLength  = 13700 # microns

wavelength   = 0.642 # microns

subgridSize  = 10001                     # Number of grid (or lattice) sites for a single lenslet
physicalSize = numLenslets * lensletSize # The full extent of the MLA

# dim = 1 makes the grid 1D
grid = grids.GridArray(numLenslets, subgridSize, physicalSize, wavelength, focalLength, dim = 1)

# Define the amplitude variation and correlation length of the phase screen
sigma_r = 10000 * np.sqrt(4 * np.pi) # amplitude variation, microns
sigma_f = 1000                       # correlation length, microns

GS_Criterion = sigma_r**2 / 4 / np.pi / sigma_f**2
print('{0:.4f}'.format(GS_Criterion))
print('GSM coherence length: {0:.4f}'.format(np.sqrt(8 * np.pi * sigma_f**4 / sigma_r**2)))

randNum = sigma_r * np.sqrt(12) * np.random.rand(grid.gridSize)

print('Theoretical variance: {0:.4f}'.format(sigma_r**2))
print('Simulated variance: {0:.4f}'.format(np.var(randNum)))

dx            = grid.px[1] - grid.px[0]

# Show a subsection of the phase screen
plt.plot(grid.px, randNum, '.')
plt.xlim((-200, 200))
plt.xlabel(r'x-position, $\mu m$')
plt.ylabel('Amplitude of phase screen, dimensionless')
plt.show()

f = 1 / np.sqrt(2 * np.pi) / sigma_f * np.exp(-(grid.px**2) / 2 / sigma_f**2)

plt.plot(grid.px, f)
plt.xlabel(r'x-position, $\mu m$')
plt.grid(True)
plt.show()

F    = dx * np.fft.fft(np.fft.ifftshift(f))
Rand = dx * np.fft.fft(np.fft.ifftshift(randNum))

Conv = F * Rand
conv = (1/dx) * np.fft.fftshift(np.fft.ifft(Conv))

# Is the noise correlated?
plt.plot(grid.px, np.abs(conv), '.')
plt.xlim((0, 5000))
plt.grid(True)
plt.show()

# Generate a Gaussian input beam
beamStd = 1000 # microns
uIn   = lambda x: (1 + 0j) / np.sqrt(2 * np.pi) / beamStd * exp(-x**2 / 2 / beamStd**2)
uIn   = np.vectorize(uIn)

# Field passed through the random phase screen
uRand = uIn(grid.px) * np.exp(1j * conv)

plt.plot(grid.px, np.abs(uRand))
plt.show()

F = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(test(grid.px)))) * dx / grid.wavelength / grid.focalLength

plt.plot(grid.pX, np.abs(F))
plt.xlim((-500,500))
plt.show()

import SimMLA.fields as fields

beam = fields.GaussianBeamWaistProfile(100, 1000)

test = fields.GSMBeamRealization(100, 1000, 500)

plt.plot(grid.px, np.abs(test(grid.px)))
#plt.xlim((-2000, 2000))
plt.show()

import importlib

importlib.reload(fields)



