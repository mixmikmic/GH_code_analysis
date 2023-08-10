import numpy as np
from scipy import signal

fs = 10e3
N = 1e3
amp = 20
freq = 1234.0
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
#print time
b, a = signal.butter(2, 0.25, 'low')
xsig = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
ysig = signal.lfilter(b, a, xsig)
xsig += amp*np.sin(2*np.pi*freq*time)
ysig += np.random.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)

puresin = amp*np.sin(2*np.pi*freq*time)
purecos = amp*np.cos(2*np.pi*freq*time)

# Took sample signals from scipy reference
# pure sin and pure cos are exact opposite signals

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
from plotly.graph_objs import *
from plotly import tools
init_notebook_mode()

def get_lin(indata, numvals):
    return np.linspace(min(indata), max(indata), numvals)

# Setup plotly data

xdata = Scatter(
    x = range(0, len(xsig)),
    y = xsig,
    name = 'X Sample'
)

ydata = Scatter(
    x = range(0, len(ysig)),
    y = ysig,
    name = 'Y Sample'
)

data = [xdata, ydata]

# Setup layout

layout = dict(title = 'Simulated Data',
              xaxis = dict(title = 'Time'),
              yaxis = dict(title = 'Unit'),
              )

# Make figure object

fig = dict(data=data, layout=layout)

iplot(fig)

# Setup plotly data

sindata = Scatter(
    x = range(0, len(puresin)),
    y = puresin,
    name = 'Sin Sample'
)

cosdata = Scatter(
    x = range(0, len(purecos)),
    y = purecos,
    name = 'Cos Sample'
)

data = [sindata, cosdata]

# Setup layout

layout = dict(title = 'Simulated Data',
              xaxis = dict(title = 'Time'),
              yaxis = dict(title = 'Unit'),
              )

# Make figure object

fig = dict(data=data, layout=layout)

iplot(fig)

# implementation of algorithm done via function in scipy; no real need for wrapper
f, Cxy = signal.coherence(xsig, ysig, fs, nperseg=256)
fpure, Csc = signal.coherence(puresin, purecos, fs, nperseg=256)

# Setup plotly data

puredata = Scatter(
    x = f,
    y = Csc,
    name = 'Pure Waves Coherence'
)

data = [puredata]

# Setup layout

layout = dict(title = 'Pure Waves Coherence',
              xaxis = dict(title = 'Frequency(Hz)'),
              yaxis = dict(title = 'Coherence'),
              )

# Make figure object

fig = dict(data=data, layout=layout)

iplot(fig)

# to fully understand the graph above, plotting fft of signals

sinspec = np.fft.fft(puresin)
sinfreq = np.fft.fftfreq(puresin.shape[-1])

cosspec = np.fft.fft(purecos)
cosfreq = np.fft.fftfreq(purecos.shape[-1])

# Setup plotly data

xdata = Scatter(
    x = sinfreq,
    y = np.abs(sinspec),
    name = 'Sine Frequency Spectrum'
)

ydata = Scatter(
    x = cosfreq,
    y = np.abs(cosspec),
    name = 'Cosine Frequency Spectrum'
)

data = [xdata, ydata]

# Setup layout

layout = dict(title = 'Frequency Spectrums',
              xaxis = dict(title = 'Frequency(Hz)'),
              yaxis = dict(title = 'Power'),
              )

# Make figure object

fig = dict(data=data, layout=layout)

iplot(fig)

data = [ydata]
fig = dict(data=data, layout=layout)

iplot(fig)

# Setup plotly data

xydata = Scatter(
    x = f,
    y = Cxy,
    name = 'XY Coherence'
)

data = [xydata]

# Setup layout

layout = dict(title = 'XY Coherence',
              xaxis = dict(title = 'Frequency(Hz)'),
              yaxis = dict(title = 'Coherence'),
              )

# Make figure object

fig = dict(data=data, layout=layout)

iplot(fig)

# to fully understand the graph above, plotting fft of signals

xspec = np.fft.fft(xsig)
xfreq = np.fft.fftfreq(xsig.shape[-1])

yspec = np.fft.fft(ysig)
yfreq = np.fft.fftfreq(ysig.shape[-1])

# Setup plotly data

xdata = Scatter(
    x = xfreq,
    y = np.abs(xspec),
    name = 'X Frequency Spectrum'
)

ydata = Scatter(
    x = yfreq,
    y = np.abs(yspec),
    name = 'Y Frequency Spectrum'
)

data = [xdata]

# Setup layout

layout = dict(title = 'X Frequency Spectrum',
              xaxis = dict(title = 'Frequency(Hz)'),
              yaxis = dict(title = 'Power'),
              )

# Make figure object

fig = dict(data=data, layout=layout)

iplot(fig)

data = [ydata]

layout = dict(title = 'Y Frequency Spectrum',
              xaxis = dict(title = 'Frequency(Hz)'),
              yaxis = dict(title = 'Power'),
              )

fig = dict(data=data, layout=layout)

iplot(fig)

# sum the freq * coherence values
print np.mean(Csc)
print np.mean(Cxy)

# implementation of algorithm done via function in scipy; no real need for wrapper
sigarr = [xsig, ysig, puresin, purecos]
f, Cxy = signal.coherence(xsig, ysig, fs, nperseg=256)
fpure, Csc = signal.coherence(puresin, purecos, fs, nperseg=256)
coherearr = []
for i in range (0, len(sigarr)):
    temparr = []
    for j in range (0, len(sigarr)):
        temparr.append(np.mean(
                signal.coherence(sigarr[i], sigarr[j], fs)[1]
            ))
    print i, temparr
    coherearr.append(temparr)
dummytitles = ['X Signal', 'Y Signal', 'Sine', 'Cosine']
#print coherearr

data = [
    Heatmap(
        x = dummytitles,
        y = dummytitles,
        z=coherearr
    )
]
iplot(data, filename='labelled-heatmap')

import os
import ast
os.chdir('c:/Users/Nitin/Documents/Hopkins/BCI/orange-panda/notes/analysis/temp')
from main import (acquire_data, clean, set_args, make_h5py_object)
set_args()
with open("pipeline.conf") as f:
    args = ast.literal_eval(f.read())

os.chdir('C:/Users/Nitin/Documents/Hopkins/BCI/datashare')

D = make_h5py_object('full_A00051826_01.mat')
#print D
eeg_data, times, coords = clean([D])[0]

d = eeg_data[:,:,-1]
currd = d[:, 2][:5000]

t = times[:,:,-1]
currt = t[0:5000]

data = [
    Scatter(
        x = currt,
        y = currd,
    )
]

# Setup layout

layout = Layout(title = 'Actual Data',
              xaxis = dict(title = 'Time (ms)'),
              yaxis = dict(title = 'Voltage (mV)'),
              )

iplot(Figure(data = data, layout = layout))

currd = d[:, :][:5000]
currd = currd.T
print len(sigarr)
print len(sigarr[0])
coherearr = []
for i in range (0, len(currd)):
    temparr = []
    for j in range (0, len(currd)):
        temparr.append(np.mean(
                signal.coherence(currd[i], currd[j], fs)[1]
            ))
    print i, temparr
    coherearr.append(temparr)

data = [
    Heatmap(
        x = range(111),
        y = range(111),
        z=coherearr
    )
]
iplot(data, filename='labelled-heatmap')

def heatmap_coherence(data, chans, 
                      skip=1, start=0, end=5000):
    currd = data[:, chans][start:end:skip]
    currd = currd.T
    coherearr = []
    for i in range (0, len(currd)):
        temparr = []
        for j in range (0, len(currd)):
            temparr.append(np.mean(
                    signal.coherence(currd[i], currd[j], fs)[1]
                ))
        coherearr.append(temparr)
    data = [
        Heatmap(
            x = chans,
            y = chans,
            z=coherearr
        )
    ]
    iplot(Figure(data = data, layout = layout), filename='labelled-heatmap')



