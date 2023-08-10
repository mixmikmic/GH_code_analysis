# importing esseitial libraries
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import json
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

steps = np.arange(13);
#steps = np.arange(7);
#steps = np.arange(7,13);

tonic = 232.

for step in steps:
    # create a tone of sine function as a numpy array
    freqtoPlay = tonic*1.5**step
    
    
    while freqtoPlay/tonic > 2:
            freqtoPlay = freqtoPlay/2
    
    print freqtoPlay
    tone = sine(0.5, freqtoPlay, 0, 1, fs) 
    # convert type to string 
    stream.write(tone.astype(np.float32).tostring())

# kill string object
stream.close()
p.terminate()

# sine function defined as A sin(2 pi f t + phi)
# A = amplitude
# phi = phase
def sine(amplitude, frequency, phase, duration, rate):
    length = int(duration * rate)
    return amplitude * np.sin(np.arange(length) * float(frequency) * (np.pi * 2) / rate + phase)

#preparing a stream object
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,channels=1, rate=44100, output=1)

steps = np.arange(13);

tonic = 232.

for step in steps:
    # create a tone of sine function as a numpy array
    freqtoPlay = tonic*(4./3)**step
    
    
    while freqtoPlay/tonic > 2:
            freqtoPlay = freqtoPlay/2
    
    print freqtoPlay
    
    tone = sine(0.5, freqtoPlay, 0, 1, fs) 
    # convert type to string 
    stream.write(tone.astype(np.float32).tostring())

# kill string object
stream.close()
p.terminate()



