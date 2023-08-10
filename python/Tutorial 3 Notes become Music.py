# importing esseitial libraries
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import json
get_ipython().magic('matplotlib inline')

# sound sampling frequency
fs = 44100

# c major scale
score = [(1,'c3'),(1,'d3'),(1,'e3'),(1,'f3'),(1,'g3'),(1,'a3'),(1,'b3'),(1,'c4')]

#voi che sapete from Figaro
score = [(1,'as4'),(0.5,'f4'),(0.5,'f4'),(1,'c5'),(1,'f4'),(1,'d5'),(0.375,'as4'),(0.125,'c5'),(0.375,'d5'),(0.125,'ds5'),(1,'c5'),(1,'r'),          (1,'d5'),(0.5,'ds5'),(0.5,'e5'),(0.75,'f5'),(0.25,'d5'),(0.5,'as4'),(0.5,'r'),(1,'c5'),(0.5,'cs5'),(0.5,'d5'),(1,'ds5'),(1,'r'),
         (0.5,'f5'),(0.5,'d5'),(0.5,'f5'),(0.5,'d5'),(0.5,'c5'),(0.5,'ds5'),(0.5,'c5'),(0.5,'ds5'),(1,'as4'),(0.375,'a4'),(0.0625,'as4'),(0.375,'c5'),(0.0625,'d5'),(1,'as4'),(1,'r')]

def sine(amplitude, frequency, phase, duration):
    length = int(duration * fs)
    return amplitude * np.sin(np.arange(length) * float(frequency) * (np.pi * 2) / fs + phase)
 
def sumPartials(duration,notetoPlay):
    if notetoPlay == 'r':
         sumTone = np.zeros(int(duration * fs))
    else:
        filename = 'sharc-angular-master/json/oboe/'+ notetoPlay +'.json'
        partials = json.loads(open(filename).read())
        sumTone = 0
        for harmonic in partials['harmonics']:
            sumTone = sumTone + sine(harmonic['amp'],harmonic['freq'], harmonic['phase'], duration)
        sumTone = sumTone / max(sumTone)
    return sumTone
    
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,channels=1, rate=44100, output=1)

#like above but add a for loop
for note in score:
    tone = sumPartials(note[0],note[1])
    stream.write(tone.astype(np.float32).tostring())


stream.close()
p.terminate()



