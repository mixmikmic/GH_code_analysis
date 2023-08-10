#python dependencies
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, RadioButtons

#change the default font set (matplotlib.rc)
mpl.rc('mathtext', fontset='stixsans', default='regular')
#increase text size somewhat
mpl.rcParams.update({'axes.labelsize':12, 'font.size': 12})

#set up notebook for inline plots
get_ipython().magic('matplotlib inline')

#generate an angular frequency range

def ang_freq_range(fstart = 9685000, fstop = 0.0968, pts_per_decade = 12):
    """returns an angular frequency range as a numpy array, 
       between fstart [Hz] and fstop [Hz], with a set number
       of points per decade (defaults to 12)"""
    decades = np.log10(fstart)-np.log10(fstop)
    pts_total = np.around(decades*pts_per_decade)
    frange = np.logspace(np.log10(fstop),np.log10(fstart), pts_total, endpoint=True)
    return 2*np.pi*frange

w = ang_freq_range()

#define function that returns impedance of -R-(RC)- circuit

def Z_R_RC(R0, R1, C1, w):
    """Returns the impedance of a -R-(RC)- circuit.
    Input
    =====
    R0 = series resistance (Ohmic resistance) of circuit
    R1 = resistance of parallel connected circuit element
    C1 = capacitance of parallel connected circuit element
    w = angular frequency, accepts an array as well as a single number
    
    Output
    ======
    The frequency dependent impedance as a complex number."""
    Z_R0 = R0
    Z_R1 = R1
    Z_C1 = -1j/(w*C1) #capacitive reactance
    Z_RC = 1/(1/Z_R1 + 1/Z_C1) #parallel connection
    return Z_R0 + Z_RC #Z_R0 and Z_RC connected in series

#define 2D plot function(Nyquist / Complex Plane)

def plot_nyquist(R0,R1,C1):
    Z = Z_R_RC(R0,R1,C1,np.logspace(7,0,7*12))
    #set up a figure canvas with two plot areas (sets of axes)
    fig,ax = plt.subplots(nrows=2, ncols=1)
    #add a Nyquist plot (first plot)
    ax[0].plot(Z.real, -1*Z.imag, marker='o',ms=5, mec='b', mew=0.7, mfc='none')
    ax[0].set_xlim(0,60)
    ax[0].set_ylim(0,25)
    ax[0].set_aspect('equal')
    ax[0].set_xlabel('Z$_{real}$ [$\Omega$]')
    ax[0].set_ylabel('-Z$_{imag}$ [$\Omega$]')
    #add a Bode plot with 
    ax[1].plot(np.logspace(7,0,7*12)/2*np.pi,-1*Z.imag, marker='o',ms=5, mec='b', mew=0.7, mfc='none')
    ax[1].set_xscale("log")
    ax[1].set_xlim(min(w),max(w))
    ax[1].set_ylim()
    ax[1].set_ylabel('-Z$_{imag}$ [$\Omega$]')
    ax[1].set_xlabel('frequency [Hz]')
    plt.tight_layout()

interact(plot_nyquist, R0=(1,20), R1=(1,40), C1=(1e-6, 1e-4, 1e-6))

get_ipython().magic('load_ext version_information')

get_ipython().magic('version_information numpy, matplotlib, ipywidgets')

