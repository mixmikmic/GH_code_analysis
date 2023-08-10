import mstm
import matplotlib
get_ipython().magic('matplotlib notebook')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate
from mpl_toolkits.mplot3d import Axes3D

# make target object
xpos = np.array([0])
ypos = np.array([0])
zpos = np.array([0])
radii = np.array([0.125])
n_matrix = 1.0
n_spheres = 1.54
target = mstm.Target(xpos, ypos, zpos, radii, n_matrix, n_spheres)

# set up parameters for calculation
theta = np.linspace(0, 181, 181)
phi = np.array([0])
stokes_par = np.array([1, 1, 0, 0])
stokes_perp = np.array([1, -1, 0, 0])
wavelength = 0.4

# run MSTM and calculate the intensities
calculation = mstm.MSTMCalculation(target, wavelength, theta, phi)
result = calculation.run()
intensity_par = result.calc_intensity(stokes_par)
intensity_perp = result.calc_intensity(stokes_perp)
result.efficiencies[0]

# plot intensity versus theta
ax = intensity_par[0].plot(x='theta',y='intensity', label='phi=0, par')
intensity_perp[0].plot(x='theta', y='intensity', ax=ax, label='phi=0, perp')

# calculate azimuthal average intensity and plot
calculation = mstm.MSTMCalculation(target, wavelength, theta, phi=None)
result = calculation.run()
intensity_par = result.calc_intensity(stokes_par)
intensity_perp = result.calc_intensity(stokes_perp)

ax = intensity_par[0].plot(x='theta',y='intensity', label='azimuthal average, par')
intensity_perp[0].plot(x='theta',y='intensity', ax=ax, label='azimuthal average, perp')

plt.xlabel('theta (degrees)')
plt.ylabel('intensity')
result.efficiencies[0]

theta = np.linspace(0, 180, 181)
phi = np.linspace(0, 360, 10)
stokes_vec = np.array([1, 1, 0, 0])

# calculate the intensities
calculation = mstm.MSTMCalculation(target, wavelength, theta, phi)
result = calculation.run()
intensity = result.calc_intensity(stokes_vec)

# reshape the result to be a 2d array
table = intensity[0].pivot_table('intensity', 'theta', 'phi')

plt.figure()
ax = plt.axes(projection = '3d')
p,t=np.meshgrid(phi,theta)
ax.plot_surface(p,t,table.as_matrix())
ax.set_zlabel('intensity')
plt.xlabel('phi (deg)')
plt.ylabel('theta (deg)')
plt.title('intensities for parallel polarized light')

theta = np.linspace(0, 180, 181)
phi = np.linspace(0, 360, 10)
stokes_vec = np.array([1, -1, 0, 0])

# calculate the intensities
calculation = mstm.MSTMCalculation(target, wavelength, theta, phi)
result = calculation.run()
intensity = result.calc_intensity(stokes_vec)

# reshape the result to be a 2d array
table = intensity[0].pivot_table('intensity', 'theta', 'phi')

plt.figure()
ax = plt.axes(projection = '3d')
p,t=np.meshgrid(phi,theta)
ax.plot_surface(p,t,table.as_matrix())
ax.set_zlabel('intensity')
plt.xlabel('phi (deg)')
plt.ylabel('theta (deg)')
plt.title('intensities for perpendicularly polarized light')

wavelength = 0.35, 0.7, 20
theta = np.linspace(0, 180, 1800)

# calculate the cross section for horizontal polarization
calculation = mstm.MSTMCalculation(target, wavelength, theta, phi=None)
result = calculation.run()
total_csca_perp = result.calc_cross_section(np.array([1, 1, 0, 0]), 0., 180.)

# calculate the cross section for horizontal polarization
total_csca_par = result.calc_cross_section(np.array([1, -1, 0, 0]), 0., 180.)

plt.figure()
plt.plot(result.wavelength, total_csca_par, label='parallel')
plt.plot(result.wavelength, total_csca_perp, label='perpendicular')
plt.plot(result.wavelength, (total_csca_par+total_csca_perp)/2, '*', label='averaged')
plt.xlabel('Wavelength (um)')
plt.xlim([0.35,0.7])
plt.ylabel('Cross Section (um^2)')
plt.title('Total cross section from integration')

# calculate the total cross section using the scattering efficiency
qsca = np.array([result.efficiencies[i].loc['par','qsca'] for i in range(len(result.efficiencies))])
csca = qsca*np.pi*radii[0]**2
#plt.figure()
plt.plot(result.wavelength, csca, label='from scattering efficiency')
plt.legend()

