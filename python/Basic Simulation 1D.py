get_ipython().magic('pylab')
import SimMLA.fftpack as simfft
import SimMLA.grids   as grids
plt.style.use('dark_background')
plt.rcParams['image.cmap'] = 'plasma'

numLenslets = 11    # Must be odd; corresponds to the number of lenslets in one dimension
lensletSize = 500   # microns
focalLength = 13700 # microns

wavelength  = 0.642 # microns

subgridSize  = 20001                      # Number of grid (or lattice) sites for a single lenslet
physicalSize = numLenslets * lensletSize # The full extent of the MLA

# dim = 1 makes the grid 1D
grid = grids.GridArray(numLenslets, subgridSize, physicalSize, wavelength, focalLength, dim = 1)

grid.px.size / numLenslets

power   = 100  # mW
beamStd = 1000 # microns
# Gaussian field
# uIn     = lambda x: np.sqrt(power) * (2 * np. pi)**(-0.5) /  beamStd * np.exp(-x**2 / 2 / beamStd**2)

# Plane wave
uIn = lambda x: np.sqrt(power) / physicalSize
uIn = np.vectorize(uIn)

plt.plot(grid.px, uIn(grid.px), linewidth = 2)
plt.xlabel('x-position, microns')
plt.grid(True)
plt.show()

grid.px.size

# Compute the interpolated fields.
get_ipython().magic('time interpMag, interpPhase = simfft.fftSubgrid(uIn, grid)')

plt.plot(grid.px, np.abs(interpMag[5](grid.px) * np.exp(1j * interpPhase[5](grid.px))))
plt.show()

get_ipython().run_cell_magic('time', '', 'fObj        = 3300 # microns\nnewGridSize = subgridSize * numLenslets # microns\n\nnewGrid = grids.Grid(newGridSize, physicalSize, wavelength, fObj, dim = 1)\nfield   = np.zeros(newGrid.gridSize)\n\n\n# For each interpolated magnitude and phase corresponding to a lenslet\n# 1) Compute the full complex field\n# 2) Sum it with the other complex fields\nfor currMag, currPhase in zip(interpMag, interpPhase):\n    fieldMag   = currMag(newGrid.px)\n    fieldPhase = currPhase(newGrid.px)\n    \n    currField = fieldMag * np.exp(1j * fieldPhase)\n    field = field + currField')

fig, (ax0, ax1) = plt.subplots(nrows = 1, ncols = 2, sharey = False)
ax0.plot(newGrid.px, np.abs(field))
ax0.set_xlabel('x-position, microns')

ax1.plot(newGrid.px, np.angle(field))
ax1.set_xlabel('x-position, microns')
plt.show()

np.save('field', field)

plt.plot(newGrid.pX, np.abs(np.fft.fftshift(np.fft.fft(np.fft.fftshift(field))))**2)
plt.show()



