get_ipython().magic('pylab')
import SimMLA.fftpack as simfft
import SimMLA.grids   as grids
plt.style.use('dark_background')
plt.rcParams['image.cmap'] = 'plasma'

numLenslets = 11    # Must be odd; corresponds to the number of lenslets in one dimension
lensletSize = 500   # microns
focalLength = 13700 # microns

wavelength  = 0.642 # microns

subgridSize  = 501                       # Number of grid (or lattice) sites for a single lenslet
physicalSize = numLenslets * lensletSize # The full extent of the MLA

grid = grids.GridArray(numLenslets, subgridSize, physicalSize, wavelength, focalLength)

power   = 100  # mW
beamStd = 1000 # microns
uIn     = lambda x, y: np.sqrt(power) / (2 * np. pi * beamStd**2) * np.exp(-(x**2 + y**2) / 2 / beamStd**2)

plt.imshow(uIn(grid.px, grid.py),
           extent = (grid.px.min(), grid.px.max(), grid.py.min(), grid.py.max()))
plt.xlabel('x-position, microns')
plt.ylabel('y-position, microns')
plt.show()

# Compute the interpolated fields.
get_ipython().magic('time interpMag, interpPhase = simfft.fftSubgrid(uIn, grid)')

get_ipython().run_cell_magic('time', '', 'fObj        = 3300 # microns\nnewGridSize = subgridSize * numLenslets # microns\n\nnewGrid = grids.Grid(newGridSize, physicalSize, wavelength, fObj)\nfield   = np.zeros((newGrid.gridSize, newGrid.gridSize))\n\n\n# For each interpolated magnitude and phase corresponding to a lenslet\n# 1) Compute the full complex field\n# 2) Sum it with the other complex fields\nfor currMag, currPhase in zip(interpMag, interpPhase):\n    fieldMag   = currMag(np.unique(newGrid.py), np.unique(newGrid.px))\n    fieldPhase = currPhase(np.unique(newGrid.py), np.unique(newGrid.px))\n    \n    currField = fieldMag * np.exp(1j * fieldPhase)\n    field = field + currField')

fig, (ax0, ax1) = plt.subplots(nrows = 1, ncols = 2, sharey = True)
ax0.imshow(np.abs(field),
           interpolation = 'nearest',
           extent = (newGrid.px.min(), newGrid.px.max(), newGrid.py.min(), newGrid.py.max()))
ax0.set_xlabel('x-position, microns')
ax0.set_ylabel('y-position, microns')

ax1.imshow(np.angle(field),
           interpolation = 'nearest',
           extent = (newGrid.px.min(), newGrid.px.max(), newGrid.py.min(), newGrid.py.max()))
ax1.set_xlabel('x-position, microns')
plt.show()

np.save('field', field)

plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(field))),
           interpolation = 'nearest',
           extent = (newGrid.pX.min(), newGrid.pX.max(), newGrid.pY.min(), newGrid.pY.max()))



