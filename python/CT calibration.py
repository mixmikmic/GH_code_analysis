import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
get_ipython().magic('matplotlib inline')

CT_data = np.genfromtxt('CT to electron density.txt', skip_header=1)

CT_vals = CT_data[:,0]   # ct values
ed_vals = CT_data[:,1]  # electron density vals

plt.plot(CT_vals, ed_vals)  # calibration data

f = interpolate.interp1d(ed_vals, CT_vals)   # returns an interpolate function

ed_new = np.arange(0, 4.5, 0.25)
CT_new = f(ed_new)   # use interpolation function returned by `interp1d`

plt.plot(CT_vals, ed_vals, '-', CT_new, ed_new, 'o');

CT_data_new = np.vstack((CT_new, ed_new)).T
CT_data_new

np.savetxt('CT_converted.csv', CT_data_new, delimiter=',')



