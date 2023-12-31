# Import functions and libraries
from __future__ import division
import numpy as np
from scipy import signal
import scipy
from numpy import *

from rtsonar import rtsonar

# Copy and paste your 5 functions here:
def genPulseTrain(pulse, Nrep, Nseg):
    # Funtion generates a pulse train from a pulse. 
    #Inputs:
    #    pulse = the pulse generated by genChirpPulse
    #    Nrep  =  number of pulse repetitions
    #    Nseg  =  Length of pulse segment >= length(pulse)
    pulse = np.append(pulse, np.zeros(Nseg - len(pulse)))
    ptrain = np.tile(pulse, Nrep)
    return ptrain


def genChirpPulse(Npulse, f0, f1, fs):
    #     Function generates an analytic function of a chirp pulse
    #     Inputs:
    #             Npulse - pulse length in samples
    #             f0     - starting frequency of chirp
    #             f1     - end frequency of chirp
    #             fs     - sampling frequency
    #     Output:
    #              pulse - chirp pulse
    
    t = r_[0.0:Npulse]/fs

    # generate f_of_t
    fff = ((f1 - f0)/(2*(Npulse/fs)))*t + f0

    # generate chirp signal
    pulse = np.exp(1j*2*np.pi*fff*t)
    return pulse

def crossCorr( rcv, pulse_a ):
    # Funtion generates cross-correlation between rcv and pulse_a
    # Inputs:
    #    rcv - received signal
    #    pulse_a - analytic pulse
    # Output:
    #    Xrcv - cross-correlation between rcv and pulse_a
    Xrcv = signal.fftconvolve(rcv, pulse_a[::-1])
    return Xrcv

def findDelay(Xrcv, Nseg=0):
    # finds the first pulse
    # Inputs:  
    #         Xrcv - the received matched filtered signal
    #         Nseg - length of a segment
    # Output:
    #          idx - index of the beginning of the first pulse
    of_max = 4
    pret_th = scipy.stats.threshold(Xrcv, threshmin=max(Xrcv)/of_max, newval = 0)
    th = scipy.stats.threshold(pret_th, threshmax=max(Xrcv)/of_max, newval = 1)
    
    for idx in range(len(th)):
        if th[idx]: return idx
        
def dist2time( dist, temperature):
    # Converts distance in cm to time in secs
    # Inputs:
    # dist        - distance to object in cm
    # temperature - in celcius
    # Output:
    # t           - time in seconds between transmitted pulse and echo
    Mdist = dist/100
    v = 331.5*np.sqrt(1 + temperature/273.17)
    return 2*Mdist/v

# Run this for Real-time Sonar
# Change the parameters!
fs = 44100 # Sampling frequency
f0 = 14000 # Chirp initial frequency
f1 = 24000 # Chirp ending frequency

Npulse = 750 # Length of Chirp Pulse
Nrep = 48 # Number of repetition in a pulse train (determines vertical length of plot )
Nseg = 2048*8 # Number of samples between pulses (determines maximum time-of-arrival)
Nplot = 200 # Number of pixels in plot along horizontal direction (higher resolution is slower)
maxdist = 300 # Maximum distance in cm
temperature = 22 # Temperature     

functions = (genChirpPulse, genPulseTrain, crossCorr, findDelay, dist2time) #join the functions together

stop_flag = rtsonar( f0, f1, fs, Npulse, Nseg, Nrep, Nplot, maxdist, temperature, functions )

# Run this to stop the sonar
stop_flag.set()





