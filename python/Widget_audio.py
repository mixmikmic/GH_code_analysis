# You need numpy, matplotlib and widgets for this to work. And some audio tools
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interactive
from IPython.display import Audio, display

def beat_freq(f1=220.0, f2=224.0):
    max_time = 5
    rate = 8000
    times = np.linspace(0,max_time,rate*max_time)
    signal = np.sin(2*np.pi*f1*times) + np.sin(2*np.pi*f2*times)
    display(Audio(data=signal, rate=rate))
    return signal

v = interactive(beat_freq, f1=(200.0,300.0), f2=(200.0,300.0))
display(v)

v.kwargs

f1, f2 = v.children[:2]
f1.value = 255
f2.value = 260
plt.plot(v.result[0:6000]);



