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

gridSize     = 100001 # samples
physicalSize = 200000 # microns
propDistance = 1e7 # microns
grid = grids.Grid(gridSize, physicalSize, wavelength, 1, dim = 1) # focalLength doesn't matter

# Use numerical FFT propagation for verification
theory = simfft.fftPropagate(beam(grid.px), grid, propDistance)
u2     = fields.GaussianBeamDefocused(fieldAmp, beamStd, wavelength, propDistance)

plt.plot(grid.px, np.abs(theory),            linewidth = 2, label = 'FFT beam at z = {0:0.0e}'.format(propDistance))
plt.plot(grid.px, np.abs(u2(grid.px)), '--', linewidth = 4, label = 'Defined beam at z = {0:0.0e}'.format(propDistance))
plt.xlim((-5000, 5000))
plt.ylim((0, 0.2))
plt.xlabel('Distance from axis')
plt.ylabel('Squared amplitude')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(grid.px, np.angle(theory),            linewidth = 2, label = 'FFT beam at z = {0:0.0e}'.format(propDistance))
plt.plot(grid.px, np.angle(u2(grid.px)), '--', linewidth = 4, label = 'Defined beam at z = {0:0.0e}'.format(propDistance))
plt.xlim((-5000, 5000))
plt.ylim((-5, 5))
plt.xlabel('Distance from axis')
plt.ylabel('Phase')
plt.grid(True)
plt.legend(loc = 'lower center')
plt.show()

from scipy.optimize import curve_fit

def Gaussian(x, *p):
    amp, std = p
    
    return amp * exp(-x**2 / 2 / std**2)
pinit   = [0.015, 3000]
popt, _ = curve_fit(Gaussian, grid.px, np.abs(u2(grid.px)), p0 = pinit)

print('The theoretical beam standard deviation is: {:.2f}'.format(w / np.sqrt(2)))
print('The SimMLA defined beam standard deviation is: {0:.2f}'.format(popt[1]))

from scipy.integrate import simps

Irrad    = np.abs(u2(grid.px))**2 / Z0 * 1000
powerOut = simps(Irrad, grid.px)
print('The output power is {0:.2f} mW'.format(powerOut))



