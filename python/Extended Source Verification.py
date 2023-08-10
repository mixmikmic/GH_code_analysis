get_ipython().magic('pylab')
get_ipython().magic('matplotlib inline')
import SimMLA.fftpack as simfft
import SimMLA.grids   as grids
import SimMLA.fields  as fields

numLenslets = 11    # Must be odd; corresponds to the number of lenslets in one dimension
lensletSize = 200   # microns
focalLength = lensletSize / 2 / 0.11 # microns

wavelength  = 0.520 # microns

subgridSize  = 1001                     # Number of grid (or lattice) sites for a single lenslet
physicalSize = numLenslets * lensletSize # The full extent of the MLA

# dim = 1 makes the grid 1D
grid = grids.GridArray(numLenslets, subgridSize, physicalSize, wavelength, focalLength, dim = 1, zeroPad = 3)

Z0       = 376.73 # Impedance of free space, Ohms
power    = 100  # mW
beamStd  = 1000 # microns

# Collimating lens
fc = 50000 # microns

# Diffuser properties
# We won't actually create the deterministic Gaussian beam; 
# We generate only random plane waves by setting powerScat = 1
# The ratio of beamSize to grainSize determines the number of independent sources.
grainSize = 100 # microns
beamSize  = 0.68 * lensletSize * fc / focalLength # microns (derived from rho in Büttner and Zeitner, 2002)
powerScat = 1  # fraction of power scattered by diffuser (remove the Gaussian part of the beam)

fieldAmp = np.sqrt(power / 1000 * Z0 / beamStd / np.sqrt(np.pi)) # Factor of 1000 converts from mW to W
beam     = fields.GaussianWithDiffuser(fieldAmp,
                                       beamStd,
                                       physicalSize, # MLA aperture size, not grid size
                                       powerScat  = powerScat,
                                       wavelength = wavelength,
                                       fc         = fc,
                                       grainSize  = grainSize,
                                       beamSize   = beamSize)

fObj         = 100000   # microns

# Grid for interpolating the field after the second MLA
newGridSize = subgridSize * numLenslets # meters
newGrid     = grids.Grid(5*newGridSize, 5*physicalSize, wavelength, fObj, dim = 1)

beamSize

beamSize * focalLength / fc / lensletSize + 1 # Should be rho = 1.68

get_ipython().run_cell_magic('time', '', 'nIter = 100\n\navgIrrad = np.zeros(newGrid.px.size, dtype=np.float128)\nfor realization in range(nIter):\n\n    # Field propagation\n    # Compute the interpolated fields\n    interpMag, interpPhase = simfft.fftSubgrid(beam, grid)\n\n    field   = np.zeros(newGrid.gridSize)\n\n    # For each interpolated magnitude and phase corresponding to a lenslet\n    # 1) Compute the full complex field\n    # 2) Sum it with the other complex fields\n    for currMag, currPhase in zip(interpMag, interpPhase):\n        fieldMag   = currMag(newGrid.px)\n        fieldPhase = currPhase(newGrid.px)\n\n        currField = fieldMag * np.exp(1j * fieldPhase)\n        field     = field + currField\n        \n    # No propagation or clipping\n    \n    # Propagate the field in the BFP to the sample\n    scalingFactor = newGrid.physicalSize / (newGrid.gridSize - 1) / np.sqrt(newGrid.wavelength * newGrid.focalLength)\n    F             = scalingFactor * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field)))    \n        \n    # Compute the irradiance on the sample\n    Irrad = np.abs(F)**2 / Z0 * 1000\n    \n    # Save the results for this realization\n    avgIrrad = avgIrrad + Irrad\n    \n# Average irradiance\navgIrrad = avgIrrad / nIter')

plt.plot(newGrid.pX, avgIrrad)
plt.xlim((-15000,15000))
plt.xlabel(r'Sample plane x-position, $\mu m$')
plt.ylabel(r'Irradiance, $W / \mu m$')
plt.grid(True)
plt.show()



