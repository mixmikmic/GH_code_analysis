# importing esseitial libraries
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# sound sampling frequency
fs = 44100

# sine function defined as A sin(2 pi f t + phi)
# A = amplitude
# phi = phase
def sine(amplitude, frequency, phase, duration, rate):
    length = int(duration * rate)
    return amplitude * np.sin(np.arange(length) * float(frequency) * (np.pi * 2) / rate + phase)

#preparing a stream object
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,channels=1, rate=44100, output=1)

# create a tone of sine function as a numpy array
tone = sine(1, 131, 0, 1, fs) 

# convert type to string 
stream.write(tone.astype(np.float32).tostring())

# kill string object
stream.close()
p.terminate()

#plot the tone to see the shape
plt.plot(np.arange(len(tone))/float(fs),tone)
plt.xlabel('Time(S)') # add label
plt.xlim([0,0.05]) # set range in x

