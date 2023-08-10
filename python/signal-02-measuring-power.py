# imports and define some constants
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import math
import scipy.signal
from ipy_table import *
h = constants.h
c = constants.c
k = constants.k
sb = constants.Stefan_Boltzmann

# Pick a frequency
nu = 1e9 # Frequency is 1 GHz
period = 1/nu

# Sample times for one period
t = np.linspace(0, period, 1000, endpoint=False)

# Choose the peak voltage
v_p = 3 # Peak voltage is 3 Volts

# Calculate sine wave
v = v_p*np.sin(2*np.pi*t*nu)

# Here is voltage squared
v2 = np.power(v,2)

# Calculate the root mean square of voltage.
# 1. square each voltage  
# 2. take the mean of those values  
# 3. take the sqrt of the mean
vrms = math.sqrt(v2.mean())


fig, ax1 = plt.subplots()
ax1.plot(t,v, color='b')
ax1.set_ylabel('V', color='b')
ax1.set_ylim((-1.1*v_p, 1.1*v_p))
for tl in ax1.get_yticklabels():
    tl.set_color('b')

ax2 = ax1.twinx()
ax2.plot(t, v2, 'r')
ax2.set_ylabel('V^2', color='r')
ax2.set_ylim((-1.1*v_p**2, 1.1*v_p**2))
for tl in ax2.get_yticklabels():
    tl.set_color('r')
ax1.set_xlabel("time (seconds)")

t = ax1.set_title("$V_{RMS}  = V_P/\sqrt{2} = %.2f$"%(vrms))

# Calculus using sympy
from sympy import *
x = Symbol('x', positive=True)
vp = Symbol('vp', positive=True)
num = integrate( (vp*sin(x))**2, (x, 0, 2*pi))
den = integrate(      1        , (x, 0, 2*pi))
print " num =",num
print " den =",den
vrms = sqrt(num/den)
print "vrms =",vrms

nus = 10e3 # sample at 10 MHz.
ns = 256 # Number of sampled used for each power spectrum calculation

# The first sine wave
vRms1 = 0.1
vPeak1 = vRms1 *np.sqrt(2)
nu1   = 100.0

# The second sine wave
vRms2 = 0.05
vPeak2 = vRms2 * np.sqrt(2)
nu2   = 1000.0

# How long will it take for 3 complete cycles of the first sine wave
tExposure = 3/nu1
nData = tExposure*nus
times = np.linspace(0, tExposure, nData, endpoint=False)
signal = vPeak1*np.sin(2*np.pi*times*nu1) + vPeak2*np.sin(2*np.pi*times*nu2)

# Plot the signal as a function of time
plt.subplot(211)
plt.plot(times/1e-3, signal)
plt.xlabel("time (mSec)")
plt.ylabel("Signal Voltage")
# Mark the expected maximum and minimum voltages
plt.axhline(vPeak1+vPeak2, color='r', linestyle=":")
plt.axhline(-vPeak1-vPeak2, color='r', linestyle=":")

# Calculate the power spectrum
f,pxx = scipy.signal.welch(signal, fs=nus, nperseg=ns, scaling='spectrum') # note the scaling

# Plot the power spectrum
plt.subplot(212)
plt.plot(f[1:-1], np.sqrt(pxx[1:-1])) # Skip the first and last frequency bins
plt.ylim(ymax=1.1*vRms1)
# Mark the height of the two sine waves
plt.axhline(vRms1, color='r', linestyle=":")
plt.axhline(vRms2, color='r', linestyle=":")
# Mark the frequencies of the two sine waves
plt.axvline(nu1, color='r', linestyle="-.")
plt.axvline(nu2, color='r', linestyle="-.")
plt.title("$\Delta \\nu = %.1f$ Hz"%(nus/ns))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude Spectrum (V)")
plt.tight_layout()

nus = 10e3 # sample at 10 MHz.
ns = 256 # Number of sampled used for each power spectrum calculation

# The first sine wave
vRms1 = 0.1
vPeak1 = vRms1 *np.sqrt(2)
nu1   = 100.0

# The second sine wave
vRms2 = 0.05
vPeak2 = vRms2 * np.sqrt(2)
nu2   = 1000.0

# How long will it take for 3 complete cycles of the first sine wave
tExposure = 3/nu1
nData = tExposure*nus
times = np.linspace(0, tExposure, nData, endpoint=False)
## Add 5000 to both frequencies
signal = vPeak1*np.sin(2*np.pi*times*(nu1+5000)) + vPeak2*np.sin(2*np.pi*times*(nu2+5000))

# Plot the signal as a function of time
plt.subplot(211)
plt.plot(times/1e-3, signal)
plt.xlabel("time (mSec)")
plt.ylabel("Signal Voltage")
# Mark the expected maximum and minimum voltages
plt.axhline(vPeak1+vPeak2, color='r', linestyle=":")
plt.axhline(-vPeak1-vPeak2, color='r', linestyle=":")

# Calculate the power spectrum
f,pxx = scipy.signal.welch(signal, fs=nus, nperseg=ns, scaling='spectrum') # note the scaling

# Plot the power spectrum
plt.subplot(212)
plt.plot(f[1:-1], np.sqrt(pxx[1:-1])) # Skip the first and last frequency bins
plt.ylim(ymax=1.1*vRms1)
# Mark the height of the two sine waves
plt.axhline(vRms1, color='r', linestyle=":")
plt.axhline(vRms2, color='r', linestyle=":")
# Mark the frequencies of the two sine waves
plt.axvline(nu1, color='r', linestyle="-.")
plt.axvline(nu2, color='r', linestyle="-.")
plt.title("$\Delta \\nu = %.1f$ Hz"%(nus/ns))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude Spectrum (V)")
plt.tight_layout()

n = 10
fActual = np.arange(n)
fObserved = np.mod(fActual, 2)
plt.plot(fActual, fObserved)
plt.ylim((-0.1, 1.1))
plt.xlabel("Actual Frequency / Nyquist Frequency")
l = plt.ylabel("Measured Frequency/ Nyquist Frequency$")

def dbmToWatts(dbm):
    return 1e-3 * math.pow(10,dbm/10.0)
def dbmToVrms(dbm, r=50):
    return math.sqrt(r * 1e-3 * math.pow(10,dbm/10.0))
def vrmsToDbm(vrms, r=50):
    return 10*math.log10(vrms**2/(r*1e-3))
def vrmsToVp(vrms):
    return math.sqrt(2)*vrms
def vpToVrms(vp):
    return vp/math.sqrt(2)

# Duplicate this table:  http://ifmaxp1.ifm.uni-hamburg.de/DBM.shtml
lines = [['dBm', "Watts","Volts rms","Volts peak", "Volts pp"]]
for dbm in range(-180, 40, 10):
    # Test that converting back and forth returns the same number
    dbm0 = vrmsToDbm(dbmToVrms(dbm))
    assert abs(dbm-dbm0) < 1e-6, "hello:  dbm=%f dbm0=%f"%(dbm,dbm0)   
    lines.append([dbm, dbmToWatts(dbm), dbmToVrms(dbm), vrmsToVp(dbmToVrms(dbm)), 2*vrmsToVp(dbmToVrms(dbm))])
make_table(lines)
apply_theme('basic')
set_global_style(float_format='%.2e')

tKelvin = 300
rOhm = 1000
sampleRate = 1e3
nperseg = 256
fNyquist = sampleRate/2.0
fCritical = 450

dNu = sampleRate/nperseg
vSigma = np.sqrt(4*k*tKelvin*rOhm*dNu)
psdExpected = vSigma**2/fNyquist
np.random.seed(123454321)
n = 100000
vRaw = np.random.normal(size=n, scale=vSigma)
# Define a filter that allows frequencies < fCritical
order = 3
b, a = scipy.signal.butter(order, fCritical/fNyquist, btype='low')
vFiltered = scipy.signal.filtfilt(b, a, vRaw)

# Calculate the power spectrum
f,pxxFiltered = scipy.signal.welch(vFiltered, fs=sampleRate, scaling='density', nperseg=nperseg) # note the scaling
f,pxxRaw = scipy.signal.welch(vRaw, fs=sampleRate, scaling='density', nperseg=nperseg) # note the scaling
plt.semilogy(f[1:-1], pxxFiltered[1:-1], label="Filtered")
plt.semilogy(f[1:-1], pxxRaw[1:-1], label="Raw")
plt.axvline(fCritical, color='r', linestyle=":")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density  V^2/Hz")
plt.axhline(psdExpected, color='r', linestyle=":")
plt.legend(loc="lower left")
plt.ylim(psdExpected/1e3, psdExpected*2)
asdNv = np.sqrt(psdExpected)/1e-9
t = plt.title("$R = %d \Omega$   $T=%d$K  PSD = %.2e V$^2$/Hz"%(rOhm, tKelvin, psdExpected))

import numpy as np
fs = 10000;
f1 = 1234;
amp1 = 2.82842712474619;
f2 = 2500.2157;
amp2 = 1;
ulsb = 1e-3;
n = 1000000

t = np.linspace(0, n/float(fs), num=n, endpoint=False)
vRaw = amp1 * np.sin(2*np.pi*f1*t) + amp2 * np.sin(2*np.pi*f2*t)
vDigitized = np.floor(vRaw/ulsb + 0.5)*ulsb

nperseg = 4096
f,psd = scipy.signal.welch(vDigitized, fs=fs, scaling='density', nperseg=nperseg)
f,ps = scipy.signal.welch(vDigitized, fs=fs, scaling='spectrum', nperseg=nperseg)
plt.subplot(121)
plt.semilogy(f[1:-1]/1e3,np.sqrt(psd[1:-1]))
vd = ulsb/np.sqrt(6*fs)
plt.axhline(vd, color='r')
plt.title("LSD")
plt.xlabel("Frequency (kHz)")
plt.ylabel("LSD ($V_{RMS}/\sqrt{Hz}$)")
plt.annotate("%.1f $\mu$V/rtHz"%(vd/1e-6), xy=(3,vd), xytext=(2.8, vd*5),
            arrowprops=dict(facecolor='red', width=1, frac=.3, edgecolor='red'),
            )
plt.subplot(122)
plt.semilogy(f[1:-1]/1e3, np.sqrt(ps[1:-1]))
plt.title("LS")
plt.xlabel("Frequency (kHz)")
plt.ylabel("LS ($V_{RMS}$)")
plt.axhline(amp1/np.sqrt(2), color='r', linestyle=":")
plt.axhline(amp2/np.sqrt(2), color='r', linestyle=":")
plt.text(f1/1e3, amp1/np.sqrt(2), "%.2f"%(amp1/np.sqrt(2)), horizontalalignment='center')
plt.text(f2/1e3, amp2/np.sqrt(2), "%.2f"%(amp2/np.sqrt(2)), horizontalalignment='center')
plt.tight_layout()

nus = 10e6
ns = 1024
lines = [["T (K)","Description","dBm","V Peak (nV)"]]
tdList = [[77, "Liquid Nitrogen"], [273, "Ice"], [293, "Room Temperature"], [373, "Boiling Water"]]
for t,description in tdList:
    dbm = 10*np.log10((4*k*t*nus)/(1e-3*ns))
    vPeak = vrmsToVp(dbmToVrms(dbm))
    lines.append([t, description, dbm, vPeak/1e-9])
make_table(lines)
apply_theme('basic')
set_global_style(float_format='%.1f')

