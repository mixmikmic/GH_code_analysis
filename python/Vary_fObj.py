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
dR          = -5000  # microns, distance of diffuser from telescope focus
L1          = 700000 # microns, distance between collimating lens and first MLA
L2          = 200000 # microns, distance between second MLA and objective BFP
wavelength  = 0.642 # microns

subgridSize  = 20001 # Number of grid (or lattice) sites for a single lenslet
physicalSize = numLenslets * lensletSize # The full extent of the MLA

# dim = 1 makes the grid 1D
collGrid = grids.Grid(20001, 5000, wavelength, fc, dim = 1)
grid     = grids.GridArray(numLenslets, subgridSize, physicalSize, wavelength, focalLength, dim = 1, zeroPad = 3)

Z0              = 376.73 # Impedance of free space, Ohms
power           = 100  # mW
beamStd         = 6    # microns
sigma_f         = 10   # microns, diffuser correlation length
sigma_r         = 1.75 # variance of the random phase
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

fObj = np.array([2000, 1800, 1650])
NA   = np.array([1.45, 1.40, 1.46])

bfpDiam = 2 * NA * fObj # microns, BFP diameter, 2 * NA * f_OBJ

# Grid for interpolating the field after the second MLA
newGridSize = subgridSize * numLenslets # microns

bfpDiam

nIter   = 1000

# Create multiple sample irradiance patterns for various values of sigma_r
for currFObj, currBFPDiam in zip(fObj, bfpDiam):
       
    # New phase mask; the diffuser sits 'dR' microns from the focus
    beam = lambda x: fields.GaussianBeamDefocused(fieldAmp, beamStd, wavelength, dR)(x)                    * fields.diffuserMask(sigma_f, sigma_r, collGrid)(x)
    
    newGrid     = grids.Grid(5*newGridSize, 5*physicalSize, wavelength, currFObj, dim = 1)
    
    avgIrrad0 = np.zeros(collGrid.px.size, dtype=np.float64)
    avgIrrad1 = np.zeros(grid.px.size, dtype=np.float64)
    avgIrrad2 = np.zeros(newGrid.px.size, dtype=np.float64)
    avgIrrad3 = np.zeros(newGrid.px.size, dtype=np.float64)
    avgIrrad4 = np.zeros(newGrid.pX.size, dtype=np.float64)
    
    for realization in range(nIter):
        print('BFP Diameter: {0:.2f}'.format(currBFPDiam))
        print('Realization number: {0:d}'.format(realization))

        # Propagate the field from the diffuser to the telescope focus
        beamSample = beam(collGrid.px)
        beamSample = simfft.fftPropagate(beamSample, collGrid, -dR)
        
        avgIrrad0 = avgIrrad0 + np.abs(beamSample)**2 / Z0 * 1000
        
        # Compute the field in the focal plane of the collimating lens
        scalingFactor = collGrid.physicalSize / (collGrid.gridSize - 1) / np.sqrt(collGrid.wavelength * collGrid.focalLength)
        afterColl     = scalingFactor * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(beamSample)))
        
        # Interpolate the input onto the new grid;
        # Propagate it to the first MLA at distance L1 away from the focal plane of the collimating lens
        inputMag = interp1d(collGrid.pX,
                            np.abs(afterColl),
                            kind         = 'nearest',
                            bounds_error = False,
                            fill_value   = 0.0)
        inputAng = interp1d(collGrid.pX,
                            np.angle(afterColl),
                            kind         = 'nearest',
                            bounds_error = False,
                            fill_value   = 0.0)
        inputField = lambda x: simfft.fftPropagate(inputMag(x) * np.exp(1j * inputAng(x)), grid, L1)
        
        avgIrrad1 = avgIrrad1 + np.abs(inputField(grid.px))**2 / Z0 * 1000

        # Compute the field magnitude and phase for each individual lenslet just beyond the second MLA
        interpMag, interpPhase = simfft.fftSubgrid(inputField, grid)

        # For each interpolated magnitude and phase corresponding to a lenslet
        # 1) Compute the full complex field
        # 2) Sum it with the other complex fields
        field   = np.zeros(newGrid.gridSize)
        for currMag, currPhase in zip(interpMag, interpPhase):
            fieldMag   = currMag(newGrid.px)
            fieldPhase = currPhase(newGrid.px)

            currField = fieldMag * np.exp(1j * fieldPhase)
            field     = field + currField

        avgIrrad2 = avgIrrad2 + np.abs(field)**2 / Z0 * 1000    
        
        # Propagate the field to the objective's BFP and truncate the region outside the aperture
        field = simfft.fftPropagate(field, newGrid, L2)
        field[np.logical_or(newGrid.px < -currBFPDiam / 2, newGrid.px > currBFPDiam / 2)] = 0.0
        
        avgIrrad3 = avgIrrad3 + np.abs(field)**2 / Z0 * 1000
            
        # Propagate the truncated field in the BFP to the sample
        scalingFactor = newGrid.physicalSize / (newGrid.gridSize - 1) / np.sqrt(newGrid.wavelength * newGrid.focalLength)
        F             = scalingFactor * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field)))    

        # Compute the irradiance on the sample
        Irrad = np.abs(F)**2 / Z0 * 1000

        # Save the results for this realization
        avgIrrad4 = avgIrrad4 + Irrad

    # Average irradiance
    avgIrrad0 = avgIrrad0 / nIter
    avgIrrad1 = avgIrrad1 / nIter
    avgIrrad2 = avgIrrad2 / nIter
    avgIrrad3 = avgIrrad3 / nIter
    avgIrrad4 = avgIrrad4 / nIter

    # Save the results
    np.save('x-coords0_BFPDiam_{0:d}.npy'.format(int(currBFPDiam)), collGrid.px)
    np.save('avgIrrad0_BFPDiam_{0:d}.npy'.format(int(currBFPDiam)), avgIrrad0)
    
    np.save('x-coords1_BFPDiam_{0:d}.npy'.format(int(currBFPDiam)), grid.px)
    np.save('avgIrrad1_BFPDiam_{0:d}.npy'.format(int(currBFPDiam)), avgIrrad1)
    
    np.save('x-coords2_BFPDiam_{0:d}.npy'.format(int(currBFPDiam)), newGrid.px)
    np.save('avgIrrad2_BFPDiam_{0:d}.npy'.format(int(currBFPDiam)), avgIrrad2)
    
    np.save('x-coords3_BFPDiam_{0:d}.npy'.format(int(currBFPDiam)), newGrid.px)
    np.save('avgIrrad3_BFPDiam_{0:d}.npy'.format(int(currBFPDiam)), avgIrrad3)
    
    np.save('x-coords4_BFPDiam_{0:d}.npy'.format(int(currBFPDiam)), newGrid.pX)
    np.save('avgIrrad4_BFPDiam_{0:d}.npy'.format(int(currBFPDiam)), avgIrrad4)

# Check the output power
powerOut = simps(avgIrrad4, newGrid.pX)
print('The output power is {0:.2f} mW'.format(powerOut))



