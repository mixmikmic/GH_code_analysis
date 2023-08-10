# Import all libraries needed for this notebook, then apply some settings
import SciServer.CasJobs as CasJobs # query with CasJobs
import numpy as np                  # standard Python lib for math ops
import pandas                       # data manipulation package
import matplotlib.pyplot as plt     # another graphing package
import astroML
from astroML.datasets import fetch_sdss_spectrum
from astropy.io import ascii
print('All libraries imported')
# ensure columns get written completely in notebook
pandas.set_option('display.max_colwidth', -1)
# do *not* show python warnings 
import warnings
warnings.filterwarnings('ignore')
print('Settings applied')

# http://www.astroml.org/user_guide/datasets.html#sdss-data

plate = 274
fiber = 102
mjd = 51913
spec = fetch_sdss_spectrum(plate=plate,fiber=fiber,mjd=mjd)

#------------------------------------------------------------
# Plot the resulting spectrum
#
# http://www.astroml.org/examples/datasets/plot_sdss_spectrum.html
#locals()['mjd']
ax = plt.axes()
ax.plot(spec.wavelength(), spec.spectrum, '-k', label='spectrum')
ax.plot(spec.wavelength(), spec.error, '-', color='r', label='error')

ax.legend(loc=4)

ax.set_title('Plate = {0:.0f}, MJD = {1:.0f}, Fiber = {2:.0f}'.format(locals()['plate'],locals()['mjd'],locals()['fiber']))

ax.set_xlabel(r'$\lambda (\AA)$')
ax.set_ylabel('Flux')
ax.axis([6500,7500, -5, 20])

plt.show()

# Space here to replot your spectrum, this time with major emission lines labeled



sky_lines_y = [60., 60, 60, 60]
sky_lines_x = [5578.5,5894.6,6301.7,7246.0]

"""
### Galaxy Lines
abs_lines_y = []
abs_lines_x = np.array([3934.777,3969.588,4305.61,5176.7,5895.6,8500.36,8544.4,8664.52])
for m in range(0,len(abs_lines)):
    abs_lines_y.append(30.)

em_lines_y = []
em_lines_x = np.array([6549.86,6564.61,6585.27,6718.29,6732.67])
for m in range(0,len(em_lines_x)):
    em_lines_y.append(35.)
"""

### Quasar Lines
abs_lines_y = []
abs_lines_x = np.array([1215.7, 1241., 1398., 1548.2, 1550.8, 2796.4, 2803.5, 2383., 2344., 2374.])
for m in range(0,len(abs_lines_x)):
    abs_lines_y.append(6.)

em_lines_y = []
em_lines_x = np.array([1215.7, 1241., 1398., 1548.2, 2796.])
for m in range(0,len(em_lines_x)):
    em_lines_y.append(35.)

z_qso = 2.95
z_abs = 6069./2796.4 -1.
abs_lines_x_shifted = abs_lines_x*(1.+z_abs)
em_lines_x_shifted = em_lines_x*(1.+z_qso)

ax = plt.axes()
ax.plot(spec.wavelength(), spec.spectrum, '-k', label='spectrum')
ax.plot(spec.wavelength(), spec.error, '-', color='r', label='error')
ax.plot(abs_lines_x_shifted, abs_lines_y, '*', label = 'absorption')
ax.plot(em_lines_x_shifted, em_lines_y, 'o', label = 'emission')
ax.plot(sky_lines_x, sky_lines_y, 'x', label = 'sky')

ax.legend(loc='upper right',numpoints=1)

ax.set_title('Plate = {0:.0f}, MJD = {1:.0f}, Fiber = {2:.0f}'.format(locals()['plate'],locals()['mjd'],locals()['fiber']))

ax.set_xlabel(r'$\lambda (\AA)$')
ax.set_ylabel('Flux')
ax.axis([3800,5500, 0, 40])

plt.show()

# Space here to replot your spectrum, this time with one or more absorption line systems labeled

# Space here to replot your spectrum, this time with one or more absorption lines labeled

# Read in a filter file

u_data = ascii.read("sdss_filters/u.dat")  
print(u_data) 


u_wave = u_data['col1']
u_tp = u_data['col2']


## If you want, you can use a query to fetch quasars only
query="""
SELECT TOP 10 specObjID, z, survey, plate, fiberID, mjd
FROM SpecObj
WHERE class = 'QSO' AND zWarning = 0 AND z>2 
"""
# send query to CasJobs
qsos = CasJobs.executeQuery(query, "dr14")
qsos



