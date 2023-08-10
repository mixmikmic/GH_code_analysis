# Imports
get_ipython().magic('matplotlib inline')
from __future__ import division
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = (9.0, 5.0)
import numpy as np
from IPython.html.widgets import interactive
from IPython.display import display, Audio
from scipy import signal
import matplotlib.pyplot as plt,mpld3
mpld3.enable_notebook()

# Define Frequencies
rows = [697,770,852,941]
cols = [1209,1336,1477]
key_pad = np.array([[1,2,3],[4,5,6],[7,8,9],[-1,0,-1]])


def play_key(key_num,duration=1):
    samp_rate = 8000
    f1 = rows[int(np.where(key_pad==key_num)[0])]
    f2 = cols[int(np.where(key_pad==key_num)[1])]
    print key_num, f1, f2
    t = np.linspace(0,duration,duration*samp_rate)
    signal = np.cos(2*np.pi*f1*t)+np.cos(2*np.pi*f2*t)
    display(Audio(data=signal, rate=samp_rate,autoplay=True))
    return signal

x = play_key(4)

# spectrum
x_spectrum = np.fft.fft(x)

# DFT frequencies
freq = np.fft.fftfreq(x_spectrum.size)

plt.figure(2)
plt.plot(freq,abs(x_spectrum))

samp_rate = 8000
# create sinusoid to correlate
f_c = 1209
duration = 1
t = np.linspace(0,duration,duration*samp_rate)
reference_signal = np.cos(2*np.pi*f_c*t)

# multiply and accumulate reference and DTMF signals
print "correlation is", np.sum(reference_signal*x)

# make a function for correlation
def corr_w_signal(x,f_c=1209,duration=1):
    t = np.linspace(0,duration,duration*samp_rate)
    reference_signal = np.cos(2*np.pi*f_c*t)
    return np.sum(reference_signal*x)

# test function
print "correlation w/ function is", corr_w_signal(x,f_c,duration=1)

# Calulate correlation for each frequency possibility
col_corr_vals = [corr_w_signal(x,f_c,duration=1) for f_c in cols]
row_corr_vals = [corr_w_signal(x,f_c,duration=1) for f_c in rows]
print "col_corr_vals =",col_corr_vals
print "row_corr_vals =",row_corr_vals

plt.figure(1)
plt.bar(cols,col_corr_vals);
plt.bar(rows,row_corr_vals);

def key_estimate(x):
    col_corr_vals = [corr_w_signal(x,f_c,duration=1) for f_c in cols]
    row_corr_vals = [corr_w_signal(x,f_c,duration=1) for f_c in rows]
    return key_pad[np.argmax(row_corr_vals),np.argmax(col_corr_vals)]

key_estimate(x)

key_num = 0

x = play_key(key_num)

print "key_estimate =", key_estimate(x)



