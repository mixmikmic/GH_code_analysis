import numpy as np

timeseries = wormData[0]['deltaFOverF'][70]
x = wormData[0]['tv'][0]

N=x.shape[0]

plt.figure(figsize=(20,5))
plt.title('DeltaFOverF for a random neuron')
plt.ylabel('Ca++-dependent Fluorescence')
plt.xlabel('Time (s)')
plt.plot(x,timeseries)
plt.show()

# get the time interval between kato data points
dt = (x[1]-x[0])

g = np.fft.fft(timeseries)
w = np.fft.fftfreq(timeseries.size,d=dt)

g2 = np.fft.ifft(g)

plt.xlabel("frequency (Hz)")
plt.ylabel("amplitude")
plt.xlim(0.0, 0.1)
#plt.plot(np.linspace(0, 1/(2*dt), N/2), np.abs(g[:N/2]))
plt.plot(w, np.abs(g))
plt.show()

#g = np.fft.fft(timeseries)
#w = np.fft.fftfreq(timeseries.size)/dt
#g *= dt*np.exp(-complex(0,1)*w*t0)/(np.sqrt(2*np.pi))
#plt.figure(figsize=(20,5))
#plt.title('Adding fourier waves')
#plt.ylim((-1.5,1.5))

plt.figure(figsize=(20,5))
plt.title('Reconstructed (inverse FFT) DeltaFOverF for a random neuron')
plt.xlabel("time(s)")
plt.ylabel("Ca++-dependent Fluorescence")
#plt.xlim(-0.05, 20)
#plt.ylim(-5, 100)
plt.xlim(0, 1200)
plt.plot(x,g2)
plt.show()

# find significant frequencies
for i in range(0,N/2):
    if (np.abs(g[i]) > 200): 
        print (1/w[i])

