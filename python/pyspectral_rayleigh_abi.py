from pyspectral import rayleigh

import numpy as np

sunz = np.array([[50., 60.], [50., 60.]])

satz = np.array([[40., 50.], [40., 50.]])

azidiff = np.array([[160, 160], [20, 20]])

corr = rayleigh.Rayleigh('GOES-16', 'abi')

print corr.get_reflectance(sunz, satz, azidiff, 'ch1')

print corr.get_reflectance(sunz, satz, azidiff, 'ch2')

print corr.get_reflectance(sunz, satz, azidiff, 'ch3')

corr = rayleigh.Rayleigh('GOES-16', 'abi', atmosphere='midlatitude summer', aerosol_type='rural_aerosol')

print corr.get_reflectance(sunz, satz, azidiff, 'ch1')

