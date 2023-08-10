get_ipython().magic('matplotlib inline')
from pylab import *
from IPython.display import Audio

Fs = 8000.0
Len = 5
t = linspace(0,Len,Fs*Len)
f1 = 442.0
f2 = 440.0
signal = sin(2*pi*f1*t) + sin(2*pi*f2*t)

Audio(data=signal, rate=Fs)

freqs = t*Fs/Len   # a list of frequencies, in Hertz
fsig = abs(fft(signal))  # Fourier transform of the signal
plot(freqs,fsig)  # amplitude versus frequency

plot(freqs[2100:2300],fsig[2100:2300])



