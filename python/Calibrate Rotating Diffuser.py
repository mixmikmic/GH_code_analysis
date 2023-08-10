get_ipython().magic('pylab')
get_ipython().magic('matplotlib inline')
import SimMLA.fftpack as simfft
import SimMLA.grids   as grids
import SimMLA.fields  as fields
from numpy.fft import fft, ifft, fftshift, ifftshift
from scipy.integrate import simps
from scipy.interpolate import interp1d

numLenslets = 21     # Must be odd; corresponds to the number of lenslets in one dimension
lensletSize = 500    # microns
focalLength = 13700  # microns, lenslet focal lengths
fc          = 50000  # microns, collimating lens focal length
dR          = -10000  # microns, distance of diffuser from telescope focus
L1          = 500000 # microns, distance between collimating lens and first MLA
L2          = 200000 # microns, distance between second MLA and objective BFP
wavelength  = 0.642 # microns

subgridSize  = 10001 # Number of grid (or lattice) sites for a single lenslet
physicalSize = numLenslets * lensletSize # The full extent of the MLA

# dim = 1 makes the grid 1D
collGrid = grids.Grid(20001, 5000, wavelength, fc, dim = 1)
grid     = grids.GridArray(numLenslets, subgridSize, physicalSize, wavelength, focalLength, dim = 1, zeroPad = 3)

Z0              = 376.73 # Impedance of free space, Ohms
power           = 100 # mW
beamStd         = 6   # microns
sigma_f         = 10  # microns, diffuser correlation length
sigma_r         = 1   # variance of the random phase
fieldAmp = np.sqrt(power / 1000 * Z0 / beamStd / np.sqrt(np.pi)) # Factor of 1000 converts from mW to W

# The diffuser sits 'dR' microns from the focus
beam     = lambda x: fields.GaussianBeamDefocused(fieldAmp, beamStd, wavelength, dR)(x)                    * fields.diffuserMask(sigma_f, sigma_r, collGrid)(x)

# Sample the beam at the diffuser
beamSample = beam(collGrid.px)

# Propagate the sample back to the focal plane of the telescope
beamSample = simfft.fftPropagate(beamSample, collGrid, -dR)

plt.plot(collGrid.px, np.abs(beamSample), linewidth = 2)
plt.xlim((-1000,1000))
plt.xlabel(r'x-position')
plt.ylabel(r'Field amplitude')
plt.grid(True)
plt.show()

plt.plot(collGrid.px, np.angle(beamSample), linewidth = 2, label ='Phase')
plt.plot(collGrid.px, np.abs(beamSample) / np.max(np.abs(beamSample)) * np.angle(beamSample), label = 'Phase with Gaussian envelope')
plt.xlim((-1000,1000))
plt.ylim((-4, 4))
plt.xlabel(r'x-position')
plt.ylabel(r'Field phase, rad')
plt.grid(True)
plt.legend()
plt.show()

scalingFactor = collGrid.physicalSize / (collGrid.gridSize - 1) / np.sqrt(collGrid.wavelength * collGrid.focalLength)
inputField    = scalingFactor * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(beamSample)))

plt.plot(collGrid.pX, np.abs(inputField))
plt.xlim((-20000, 20000))
plt.grid(True)
plt.show()

# Interpolate this field onto the MLA grid
mag = np.abs(inputField)
ang = np.angle(inputField)

inputMag = interp1d(collGrid.pX,
                    mag,
                    kind         = 'nearest',
                    bounds_error = False,
                    fill_value   = 0.0)
inputAng = interp1d(collGrid.pX,
                    ang,
                    kind         = 'nearest',
                    bounds_error = False,
                    fill_value   = 0.0)

plt.plot(grid.px, np.abs(inputMag(grid.px) * np.exp(1j * inputAng(grid.px))))
plt.xlim((-5000, 5000))
plt.grid(True)
plt.show()

field2 = lambda x: inputMag(x) * np.exp(1j * inputAng(x))

interpMag, interpAng = simfft.fftSubgrid(field2, grid)

# Plot the field behind the second MLA center lenslet
plt.plot(grid.pX, np.abs(interpMag[10](grid.pX) * np.exp(1j * interpAng[10](grid.pX))))
plt.xlim((-500, 500))
plt.xlabel('x-position')
plt.ylabel('Field amplitude')
plt.grid(True)
plt.show()

fObj    = 3300           # microns
bfpDiam = 2 * 1.4 * fObj # microns, BFP diameter, 2 * NA * f_OBJ

# Grid for interpolating the field after the second MLA
newGridSize = subgridSize * numLenslets # microns
newGrid     = grids.Grid(5*newGridSize, 5*physicalSize, wavelength, fObj, dim = 1)

get_ipython().run_cell_magic('time', '', "nIter   = 100\n#sigma_r = np.array([0.1, 0.3, 1, 3])\nsigma_r = np.array([1])\n\n# Create multiple sample irradiance patterns for various values of sigma_r\nfor sigR in sigma_r:\n       \n    # New phase mask; the diffuser sits 'dR' microns from the focus\n    beam = lambda x: fields.GaussianBeamDefocused(fieldAmp, beamStd, wavelength, dR)(x) \\\n                   * fields.diffuserMask(sigma_f, sigR, collGrid)(x)\n        \n    avgIrrad = np.zeros(newGrid.px.size, dtype=np.float128)\n    for realization in range(nIter):\n        print('sigma_r: {0:.2f}'.format(sigR))\n        print('Realization number: {0:d}'.format(realization))\n\n        # Propagate the field from the diffuser to the telescope focus\n        beamSample = beam(collGrid.px)\n        beamSample = simfft.fftPropagate(beamSample, collGrid, -dR)\n        \n        # Compute the field in the focal plane of the collimating lens\n        scalingFactor = collGrid.physicalSize / (collGrid.gridSize - 1) / np.sqrt(collGrid.wavelength * collGrid.focalLength)\n        afterColl     = scalingFactor * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(beamSample)))\n        \n        # Interpolate the input onto the new grid;\n        # Propagate it to the first MLA at distance L1 away from the focal plane of the collimating lens\n        inputMag = interp1d(collGrid.pX,\n                            np.abs(afterColl),\n                            kind         = 'nearest',\n                            bounds_error = False,\n                            fill_value   = 0.0)\n        inputAng = interp1d(collGrid.pX,\n                            np.angle(afterColl),\n                            kind         = 'nearest',\n                            bounds_error = False,\n                            fill_value   = 0.0)\n        inputField = lambda x: simfft.fftPropagate(inputMag(x) * np.exp(1j * inputAng(x)), grid, L1)\n\n        # Compute the field magnitude and phase for each individual lenslet just beyond the second MLA\n        interpMag, interpPhase = simfft.fftSubgrid(inputField, grid)\n\n        # For each interpolated magnitude and phase corresponding to a lenslet\n        # 1) Compute the full complex field\n        # 2) Sum it with the other complex fields\n        field   = np.zeros(newGrid.gridSize)\n        for currMag, currPhase in zip(interpMag, interpPhase):\n            fieldMag   = currMag(newGrid.px)\n            fieldPhase = currPhase(newGrid.px)\n\n            currField = fieldMag * np.exp(1j * fieldPhase)\n            field     = field + currField\n\n        # Propagate the field to the objective's BFP and truncate the region outside the aperture\n        field = simfft.fftPropagate(field, newGrid, L2)\n        field[np.logical_or(newGrid.px < -bfpDiam / 2, newGrid.px > bfpDiam / 2)] = 0.0\n            \n        # Propagate the truncated field in the BFP to the sample\n        scalingFactor = newGrid.physicalSize / (newGrid.gridSize - 1) / np.sqrt(newGrid.wavelength * newGrid.focalLength)\n        F             = scalingFactor * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field)))    \n\n        # Compute the irradiance on the sample\n        Irrad = np.abs(F)**2 / Z0 * 1000\n\n        # Save the results for this realization\n        avgIrrad = avgIrrad + Irrad\n\n    # Average irradiance\n    avgIrrad = avgIrrad / nIter\n\n    # Save the results\n    # The folder 'Rotating Diffuser Calibration' should already exist.\n    #np.save('Rotating Diffuser Calibration/x-coords_sigR_{0:.3f}.npy'.format(sigR), newGrid.pX)\n    #np.save('Rotating Diffuser Calibration/avgIrrad_sigR_{0:.3f}.npy'.format(sigR), avgIrrad)")

plt.plot(newGrid.pX, avgIrrad)
plt.xlim((-100,100))
plt.xlabel(r'Sample plane x-position, $\mu m$')
plt.ylabel(r'Irradiance, $mW / \mu m$')
plt.grid(True)
plt.show()

# Check the output power
powerOut = simps(avgIrrad, newGrid.pX)
print('The output power is {0:.2f} mW'.format(powerOut))

