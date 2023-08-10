get_ipython().magic('pylab')
get_ipython().magic('matplotlib inline')
import SimMLA.fftpack as simfft
import SimMLA.grids   as grids
import SimMLA.fields  as fields

# Define a Gaussian beam
Z0         = 376.73 # Impedance of free space, Ohms
power      = 100    # mW
beamStd    = 1000   # microns
wavelength = 0.642  # microns

fieldAmp = np.sqrt(power / 1000 * Z0 / beamStd / np.sqrt(np.pi)) # Factor of 1000 converts from mW to W
beam     = fields.GaussianBeamWaistProfile(fieldAmp, beamStd)

w = np.sqrt(2 * 1e3**2) * np.sqrt(1 + ((1e7 * 0.642)/(np.pi * 2 * (1e3)**2))**2)
print(w)
print(w / np.sqrt(2) )

gridSize     = 10001 # samples
physicalSize = 100000 # microns
propDistance = 1e7 # microns
grid = grids.Grid(gridSize, physicalSize, wavelength, 1, dim = 1) # focalLength doesn't matter

u2 = simfft.fftPropagate(beam(grid.px), grid, propDistance)

plt.plot(grid.px, beam(grid.px), linewidth = 3, label = 'Beam at z = 0')
plt.plot(grid.px, np.abs(u2), linewidth = 2, label = 'Beam at z = {0:0.0e}'.format(propDistance))
plt.xlim((-5000, 5000))
plt.grid(True)
plt.legend()
plt.show()

from scipy.optimize import curve_fit

def Gaussian(x, *p):
    amp, std = p
    
    return amp * exp(-x**2 / 2 / std**2)
pinit   = [0.015, 3000]
popt, _ = curve_fit(Gaussian, grid.px, np.abs(u2), p0 = pinit)

print('The theoretical beam standard deviation is: {:.2f}'.format(w / np.sqrt(2)))
print('The numerical beam standard deviation is: {0:.2f}'.format(popt[1]))

