get_ipython().magic('pylab')
get_ipython().magic('matplotlib inline')
import SimMLA.fftpack as simfft
import SimMLA.grids   as grids
import SimMLA.fields  as fields
from numpy.fft import fft, ifft, fftshift, ifftshift
from scipy.integrate import simps

focalLength = 250e-3

wavelength  = 650e-9

gridSize     = 251   # Number of grid (or lattice) sites
physicalSize = 50e-3 # The full extent of the grid

# dim = 1 makes the grid 1D
grid = grids.Grid(gridSize, physicalSize, wavelength, focalLength, dim = 1)

Z0              = 376.73 # Impedance of free space, Ohms
power           = 100  # mW
beamStd         = 1e-3
coherenceLength = 8e-3
fieldAmp = np.sqrt(power / 1000 * Z0 / beamStd / np.sqrt(np.pi)) # Factor of 1000 converts from mW to W

beam     = fields.GSMBeamRealization(fieldAmp, beamStd, coherenceLength, grid)

beamSample = beam(grid.px)

plt.plot(grid.px, np.angle(beamSample), linewidth = 2)
plt.xlabel(r'x-position')
plt.ylabel(r'Field phase, rad')
plt.grid(True)
plt.show()

slitWidth  = 2e-3 
slitOffset = 2.5e-3
beamPower  = 100  # mW

Z0    = 376.73 # Impedance of free space, Ohms
amp   = np.sqrt((beamPower / 1000) * Z0 / slitWidth / 2)
field = np.zeros(grid.px.size)
field[np.logical_and(grid.px >  (-slitOffset - slitWidth / 2), grid.px <= (-slitOffset + slitWidth / 2))] = amp
field[np.logical_and(grid.px >= ( slitOffset - slitWidth / 2), grid.px <  ( slitOffset + slitWidth / 2))] = amp

plt.plot(grid.px, field, linewidth = 2)
plt.xlim((-5e-3, 5e-3))
plt.xlabel(r'x-coordinate, $\mu m$')
plt.ylabel(r'Field amplitude, $V / \sqrt{\mu m}$')
plt.grid(True)
plt.show()

dx = grid.px[1] - grid.px[0]
F  = fftshift(fft(ifftshift(field))) * dx
Irrad = np.abs(F)**2 / wavelength / focalLength

plt.plot(grid.pX, Irrad)
plt.xlim((-2e-4,2e-4))
plt.xlabel('x-position')
plt.ylabel('Irradiance, $W / m$')
plt.grid(True)
plt.show()

# Check power conservation
intPower = simps(Irrad, grid.pX)
print(intPower / Z0 * 1000) 

# Generate a new phase screen
beamSample = beam(grid.px)

t = np.exp(1j * np.angle(beamSample))
newField = t * field

# Plot the real part of the field
plt.plot(grid.px, np.real(newField))
plt.show()

# Propagate and plot the new field
dx = grid.px[1] - grid.px[0]
F  = fftshift(fft(ifftshift(newField))) * dx
newIrrad = np.abs(F)**2 / wavelength / focalLength

plt.plot(grid.pX, newIrrad)
plt.plot(grid.pX, Irrad)
#plt.xlim((-2e-4,2e-4))
plt.xlabel('x-position')
plt.ylabel('Irradiance, $W / m$')
plt.grid(True)
plt.show()

nIter = 1000
dx = grid.px[1] - grid.px[0]

finalIrrad = np.zeros(grid.px.size)
for ctr in range(nIter):
    # Create a new realization of the field
    t = np.exp(1j * np.angle(beam(grid.px)))
    newField = t * field
    
    F  = fft(ifftshift(newField)) * dx
    Irrad = np.abs(F)**2 / wavelength / focalLength
    
    finalIrrad = finalIrrad + Irrad
    
# Find the averaged irradiance pattern    
finalIrrad = fftshift(finalIrrad) / nIter

plt.plot(grid.pX, finalIrrad)
#plt.xlim((-2e-4,2e-4))
plt.xlabel('x-position')
plt.ylabel('Irradiance, $W / m$')
plt.grid(True)
plt.show()

# Check the integrated power
intPower = simps(newIrrad, grid.pX)
print(intPower / Z0 * 1000)

from scipy.optimize import curve_fit as cf

def sinc(x):
    if (x != 0):
        # Prevent divide-by-zero
        return np.sin(np.pi * x) / (np. pi * x)
    else:
        return 1
sinc = np.vectorize(sinc)

def theory(x, *p):
    mu = p
    
    return 2 * slitWidth**2 * amp**2 / wavelength / focalLength * sinc(x * slitWidth / wavelength / focalLength)**2         * (1 + mu * np.cos(2 * np.pi * (slitOffset * 2) * x / wavelength / focalLength))

initGuess = 0.6
popt, _ = cf(theory, grid.pX, finalIrrad, p0 = initGuess)

mu = popt[0]

plt.plot(grid.pX, finalIrrad, linewidth = 2, label = 'FFT')
plt.plot(grid.pX, theory(grid.pX, mu), '--', linewidth = 3, label = 'Fit, d.o.c. = {0:.2f}'.format(mu))
#plt.xlim((-2e-4,2e-4))
plt.xlabel('x-position')
plt.ylabel('Irradiance, $W / m$')
plt.grid(True)
plt.legend()
plt.show()


numericCohLength = np.sqrt(- (2 * slitOffset)**2 / np.log(mu))
print('The fitted coherence length is: {0:.4f}'.format(numericCohLength))
print('The input coherence length is: {0:.4f}'.format(coherenceLength))



