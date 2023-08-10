get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pylab as plt

def ft(y):
    """Returns the fourier transform of y"""
    return np.fft.fftshift(np.fft.fft(np.fft.fftshift(y)))

def ift(y):
    """Returns the inverse fourier transform of y"""
    return np.fft.fftshift(np.fft.ifft(np.fft.fftshift(y)))

def ftfreqs(N,dt):
    """Returns the Fourier frequencies"""
    return np.fft.fftshift(np.fft.fftfreq(N,dt))

# constant function of amplitude A
def constant(t,A):
    return A*np.ones(len(t))

# spike of amplitude A at given value of t
def spike(t,t0,A):
    output = np.zeros(len(t))
    output[t==t0] = A
    print sum(t==t0)
    return output

# top-hat function
def tophat(t,width,A):
    output = np.zeros(len(t))
    output[abs(t)<width/2.0] = A
    return output

# sinc function
def sincfunc(t,scale):
    return np.sinc(t/scale)

# gaussian function
def gaussian(t,sigma):
    return (1./(np.sqrt(2.*np.pi)*sigma))*np.exp(-t**2.0/(2.0*sigma**2.0))

# comb function
def comb(t,t_space,A):
    output = np.zeros(len(t))
    mod_array = np.array([round(i,2)%t_space for i in t])
    output[mod_array==0] = A
    return output

delta_t = 0.02
t = np.arange(-1200,1200,delta_t)              # set x-axis value

freqs = ftfreqs(len(t),delta_t)                # get our Fourier transform frequency values

h = 5.0*np.cos(2.*np.pi*t*0.5)                    # create function f(t)
H = ft(h)                                      # Fourier transform

fig, ax = plt.subplots(1,2,figsize=[13,3])     # create plot

ax[0].plot(t,h.real)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('h(t) (real)')

ax[1].plot(freqs,H.real)
ax[1].set_xlabel('Frequency')
ax[1].set_ylabel('H(f) (real)')

ax[0].set_xlim(-50.,50.)
ax[1].set_xlim(-0.7,0.7)

g = sincfunc(t,20.)                            # create function f(t)
G = ft(g)                                      # Fourier transform

fig, ax = plt.subplots(1,2,figsize=[13,3])     # create plot

ax[0].plot(t,g)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('g(t) (real)')

ax[1].plot(freqs,G.real)
ax[1].set_xlabel('Frequency')
ax[1].set_ylabel('G(f) (real)')

ax[0].set_xlim(-150.,150.)
ax[1].set_xlim(-0.7,0.7)

# lets convolve our Fourier transformed functions, using the numpy convolve function:
Y = np.convolve(H,G,mode='same')
# then reverse Fourier transform:
y = ift(Y)

fig, ax = plt.subplots(1,2,figsize=[13,3])     # create plot

ax[0].plot(t,y.real)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('y(t) (real)')

ax[1].plot(freqs,Y.real)
ax[1].set_xlabel('Frequency')
ax[1].set_ylabel('Y(f) (real)')

ax[0].set_xlim(-80.,80.)
ax[1].set_xlim(-0.7,0.7)

fig, ax = plt.subplots(3,2,figsize=[13,9]) 

# plot h and H
ax[0,0].plot(t,h)
ax[0,0].set_ylabel('h(t) (real)')

ax[0,1].plot(freqs,H.real)
ax[0,1].set_ylabel('H(f) (real)')

ax[0,0].set_xlim(-50.,50.)
ax[0,1].set_xlim(-0.7,0.7)

# plot g and G
ax[1,0].plot(t,g)
ax[1,0].set_ylabel('g(t) (real)')

ax[1,1].plot(freqs,G.real)
ax[1,1].set_ylabel('G(f) (real)')

ax[1,0].set_xlim(-100.,100.)
ax[1,1].set_xlim(-0.7,0.7)

# plot y and Y
ax[2,0].plot(t,y.real)
ax[2,0].set_xlabel('Time')
ax[2,0].set_ylabel('y(t) (real)')

ax[2,1].plot(freqs,Y.real)
ax[2,1].set_xlabel('Frequency')
ax[2,1].set_ylabel('Y(f) (real)')

ax[2,0].set_xlim(-100.,100.)
ax[2,1].set_xlim(-0.7,0.7)

import os, sys
import numpy
import matplotlib
import IPython

print 'OS:          ', os.name, sys.platform
print 'Python:      ', sys.version.split()[0]
print 'IPython:     ', IPython.__version__
print 'Numpy:       ', numpy.__version__
print 'matplotlib:  ', matplotlib.__version__



