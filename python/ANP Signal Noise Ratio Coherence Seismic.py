import numpy as np
import pylab 
from scipy import signal

get_ipython().magic('matplotlib inline')

n = 2000
t = np.linspace(0, 2*3.14159, n)
y = np.cos(t*3*np.pi)+np.cos(t*11*np.pi)+signal.gausspulse(t-0.5*np.pi, fc=9)-signal.gausspulse(t-np.pi, fc=5) # noise free
y = y*signal.hanning(n) # in case you want to test just uncomment it
y -= y.mean()
y0 = y + 0.05*y*(2*np.random.rand(n)-1) # + noise A 5%
y1  = y + 0.05*y*(2*np.random.rand(n)-1) # + noise B 5%
y2 = y + 0.15*y*(2*np.random.rand(n)-1) # + noise C 15%
y3  = y + 0.15*y*(2*np.random.rand(n)-1) # + noise D 15%
pylab.figure(figsize=(15,2))
pylab.plot(t, y, 'b', label="noise free signal")
pylab.legend(loc=3)
pylab.figure(figsize=(15,2))
pylab.plot(t, y1, ',g-', label="+ 5% random noise")
pylab.legend(loc=3)
pylab.figure(figsize=(15,2))
pylab.plot(t, y2, ',r-', label="+ 15% random noise")
pylab.legend(loc=3)
pylab.show()

fs = np.fft.fftfreq(n, 3.14159/n)
pylab.plot(fs[:n/2], np.abs(np.fft.fft(y[:n/2])), 'b-', label="noise free signal")  # just half spectrum needed (real functions)
pylab.plot(fs[:n/2], np.abs(np.fft.fft(y2[:n/2])), '.r', label="+ 15% random noise")
pylab.xlim(0, 20)
pylab.ylim(1, 1000)
pylab.legend(loc=1)
pylab.yscale('log')
pylab.xlabel('frequency (Hz)')
pylab.ylabel('Log Spectrum')
pylab.show()

def coherence(t0, t1):
    """
    Coefficient of coherence between two identical traces
    As defined by G. Henry the difference between those two traces
    will be due the random noise present
    
    Parameters:
    
    * t0 : array
        first trace must be stationary (zero mean)    
    * t1 : array
        second trace must be stationary (zero mean)
        
    Returns:
    
    * coherence: array
    
    """
    aut0 = np.correlate(t0, t1, 'same')
    aut1 = np.correlate(t1, t0, 'same')
    aut = np.fft.fft(np.correlate(aut0, aut1, 'same'))
    cros = np.fft.fft(np.correlate(np.correlate(t0, t0, 'same'), np.correlate(t1, t1,'same'), 'same'))    
    return np.sqrt(np.abs(aut/cros)) # just the modulus part

fs = np.fft.fftfreq(n, 3.14159/n)
c0 = coherence(y0, y1)
c1 = coherence(y2, y3)
pylab.figure(figsize=(15,3))
ax = pylab.subplot(111)
ax.plot(fs[:n/2], c0[:n/2], '+b', label="traces 5% random noise")
ax.plot(fs[:n/2], c1[:n/2], '.r', label="traces 15% random noise")
pylab.legend()
pylab.xlim(0, 20) # just the range of frequencies we did input
pylab.ylim(0.95, 1.05)
pylab.ylabel('coherence')
pylab.xlabel('Frequency (Hz)')
pylab.figure(figsize=(15,3))
ax = pylab.subplot(111)
ax.plot(fs[:n/2], np.abs(1-c0[:n/2]), '+b', label="traces 5% random noise")
ax.plot(fs[:n/2], np.abs(1-c1[:n/2]), '.r', label="traces 15% random noise")
pylab.xlim(0, 20) # just the range of frequencies we did input
pylab.ylabel('|1-coherence| (log scale)')
pylab.yscale('log')
pylab.legend()
pylab.show()

fs = np.fft.fftfreq(n, 3.14159/n)
sn0 = c0/np.abs(1-c0)
sn1 = c1/np.abs(1-c1)
# db scale (amplitude is 20*)
sn0 = 20*np.log(sn0)
sn1 = 20*np.log(sn1)
pylab.figure(figsize=(15,4))
ax = pylab.subplot(111)
ax.plot(fs[:n/2], sn0[:n/2], '.b-', label="traces 5% random noise")
ax.plot(fs[:n/2], sn1[:n/2], '+r-', label="traces 15% random noise")
pylab.xlim(0, 20)
pylab.ylabel('S/N ratio (dB)') # 
pylab.xlabel('Frequency (Hz)')
pylab.legend(loc=3)
pylab.show()













sg = np.correlate(y, y, 'same')
ns = np.correlate(y2, y2, 'same')
pylab.figure(figsize=(15,3))
pylab.plot(sg, 'b-')
pylab.plot(ns, '.r')
pylab.figure(figsize=(15,3))
pylab.plot(np.abs(ns-sg), '.r')
pylab.ylabel('noisy-signal')
#print sg.sum()
#print ns.sum()

