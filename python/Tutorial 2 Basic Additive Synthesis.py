# importing esseitial libraries
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import json
get_ipython().magic('matplotlib inline')

# sound sampling frequency
fs = 44100

def sine(amplitude, frequency, phase, duration, rate):
    length = int(duration * rate)
    return amplitude * np.sin(np.arange(length) * float(frequency) * (np.pi * 2) / rate + phase)

# like above to combining all sine waves (partials) into a synthesized note
# read from each json table
def sumPartials(duration,notetoPlay):
    filename = 'sharc-angular-master/json/oboe/'+notetoPlay+'.json'
    partials = json.loads(open(filename).read())
    sumTone = 0
    for harmonic in partials['harmonics']:
        sumTone = sumTone + sine(harmonic['amp'],harmonic['freq'], harmonic['phase'], duration, fs)
    sumTone = sumTone / max(sumTone)
    return sumTone

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,channels=1, rate=44100, output=1)

tone = sumPartials(1,'c4')

stream.write(tone.astype(np.float32).tostring())


stream.close()
p.terminate()



