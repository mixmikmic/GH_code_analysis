get_ipython().magic('pylab')
get_ipython().magic('matplotlib inline')
import SimMLA.fftpack as simfft
import SimMLA.grids   as grids

numLenslets = 11    # Must be odd; corresponds to the number of lenslets in one dimension
lensletSize = 200   # microns
focalLength = lensletSize / 2 / 0.11 # microns

wavelength  = 0.520 # microns

print('Spatial sampling period: {0:.4f} microns'.format(200 / 100001.0))

subgridSize  = 100001                     # Number of grid (or lattice) sites for a single lenslet
physicalSize = numLenslets * lensletSize # The full extent of the MLA

# dim = 1 makes the grid 1D
# zeroPad = N makes the computational grid N times larger than the MLA
grid = grids.GridArray(numLenslets, subgridSize, physicalSize, wavelength, focalLength, dim = 1, zeroPad = 5)

powerIn = 100 # mW
Z0      = 376.73 # Impedance of free space, Ohms

# Determine the field amplitude
fieldAmp = np.sqrt((powerIn / 1000) * Z0 / physicalSize) # Volts / sqrt(microns)

# Plane wave
uIn = lambda x: fieldAmp
uIn = np.vectorize(uIn)

# Plot the input field over the array
plt.plot(grid.px, uIn(grid.px), linewidth = 2)
plt.xlabel(r'x-position, $\mu m$')
plt.ylabel(r'Field amplitude, $V / \sqrt{\mu m}$')
plt.grid(True)
plt.show()

# Compute the interpolated fields
# Linear interpolation is used for speed
get_ipython().magic('time interpMag, interpPhase = simfft.fftSubgrid(uIn, grid)')

# Check that we've successfully computed the Fourier transform of a lenslet.
m = 6 # lenslet index, integers between [0, numLenslets - 1]
plt.plot(grid.px, np.abs(interpMag[m](grid.px) * np.exp(1j * interpPhase[m](grid.px))))
plt.xlabel(r'x-position, $\mu m$')
plt.ylabel(r'Field amplitude, $V / \sqrt{\mu m}$')
plt.xlim((0,400))
plt.grid(True)
plt.show()

# Check that we've successfully computed the Fourier transform of a lenslet.
m = 6 # lenslet index, integers between [0, numLenslets - 1]
plt.plot(grid.px, np.angle(interpMag[m](grid.px) * np.exp(1j * interpPhase[m](grid.px))),'.')
plt.xlabel(r'x-position, $\mu m$')
plt.ylabel(r'Phase, radians')
plt.xlim((0,400))
plt.grid(True)
plt.show()

get_ipython().run_cell_magic('time', '', 'fObj        = 100000 # microns\nnewGridSize = subgridSize * numLenslets # microns\n\n# Upsample grid by an odd factor to reduce rippling\nnewGrid = grids.Grid(5*newGridSize, 5*physicalSize, wavelength, fObj, dim = 1)\nfield   = np.zeros(newGrid.gridSize)\n\n# For each interpolated magnitude and phase corresponding to a lenslet\n# 1) Compute the full complex field\n# 2) Sum it with the other complex fields\nfor currMag, currPhase in zip(interpMag, interpPhase):\n    fieldMag   = currMag(newGrid.px)\n    fieldPhase = currPhase(newGrid.px)\n    \n    currField = fieldMag * np.exp(1j * fieldPhase)\n    field     = field + currField')

fig, (ax0, ax1) = plt.subplots(nrows = 1,
                               ncols = 2,
                               sharex = True,
                               sharey = False,
                               figsize = (10,6))

ax0.plot(newGrid.px, np.abs(field))
ax0.set_xlim((-7500, 7500))
ax0.set_xlabel(r'x-position, $\mu m$')
ax0.set_ylabel(r'Amplitude, $V / \sqrt{\mu m}$')
ax0.grid()

ax1.plot(newGrid.px, np.angle(field), '.')
ax1.set_xlabel(r'x-position, $\mu m$')
ax1.set_ylabel('Phase, radians')
ax1.grid()

plt.tight_layout()
plt.show()

from scipy.integrate import simps
# Factor of 1000 converts from Watts to milliWatts
power1Prime = simps(np.abs(field)**2, newGrid.px) / Z0 * 1000 

print('Input power is: {0:.4f} mW'.format(powerIn))
print('Diffracted power is: {0:.4f} mW'.format(power1Prime))

# Sample plane irradiance including limiting aperture in the BFP
scalingFactor = newGrid.physicalSize / (newGrid.gridSize - 1) / np.sqrt(newGrid.wavelength * newGrid.focalLength)
F             = scalingFactor * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field)))

plt.plot(newGrid.pX, np.abs(F)**2 / Z0 * 1000)
plt.xlim((-15000, 15000))
plt.xlabel(r'x-position, $\mu m$')
plt.ylabel(r'Irradiance, $mW / \mu m$')
plt.grid(True)
plt.show()

powerSample = simps(np.abs(F)**2, newGrid.pX) / Z0 * 1000
print('Power delivered to sample: {0:.4f} mW'.format(powerSample))



