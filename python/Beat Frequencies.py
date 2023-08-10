get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

from IPython.html.widgets import interactive
from IPython.display import Audio, display
import numpy as np

def beat_freq(f1=220.0, f2=224.0):
    max_time = 3
    rate = 8000
    times = np.linspace(0,max_time,rate*max_time)
    signal = np.sin(2*np.pi*f1*times) + np.sin(2*np.pi*f2*times)
    print(f1, f2, abs(f1-f2))
    display(Audio(data=signal, rate=rate))
    return signal

v = interactive(beat_freq, f1=(200.0,300.0), f2=(200.0,300.0))
display(v)

v.kwargs

f1, f2 = v.children
f1.value = 255
f2.value = 260
plt.plot(v.result[0:6000])

