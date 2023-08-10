get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pylab as plt

# create an array of x-axis values, from 0 to 1000, with step size of 1
delta_n = 1.0
n_array = np.arange(0,1000.,delta_n)

# create two cos functions with specified periods and amplitudes
P1 = 100.
A1 = 1.0
fn1 = A1*np.cos(2.*np.pi*n_array/P1) 

P2 = 50.
A2 = 0.5
fn2 = A2*np.cos(2.*np.pi*n_array/P2)

# add the functions to form our array x to Fourier transform
x_array = fn1 + fn2

# print the periods and frequencies of the two components of our x array
print 'Periods:     ', P1, '  ', P2
print 'Frequencies: ', 1./P1, '   ', 1./P2

# plot our x array
plt.plot(n_array,x_array)
plt.xlabel('time')
plt.ylabel('f(t)')
plt.show() 

N = len(n_array)
k_array = np.arange(-(N/2.0-1.0),N/2.0,1.0)

X = np.zeros(len(k_array), dtype=np.complex)

# iterate over the fourier-space variable k
for k_indx,k in enumerate(k_array):
    # iterate over the original-space variable n
    for n_indx,n in enumerate(n_array):
        arg = x_array[n_indx]*np.exp(-1.j*2.0*np.pi*k*n/N)
        X[k_indx] = X[k_indx] + arg

# create plot
f, ax = plt.subplots(1,2,figsize=[13,3])

ax[0].plot(k_array/N,X.real,'o-')
ax[0].set_xlabel('Frequency')
ax[0].set_ylabel('Real')

ax[1].plot(k_array/N,X.imag,'o-')
ax[1].set_xlabel('Frequency')
ax[1].set_ylabel('Imag')

plt.stem(k_array/N,X.real)
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.xlim(0,0.05)
plt.ylim(-10,510)

import os, sys
import numpy
import matplotlib
import IPython

print 'OS:          ', os.name, sys.platform
print 'Python:      ', sys.version.split()[0]
print 'IPython:     ', IPython.__version__
print 'Numpy:       ', numpy.__version__
print 'matplotlib:  ', matplotlib.__version__



