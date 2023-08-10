# This section will take a few moments to run

import time
import numpy as np

from pynq.overlays.base import BaseOverlay
base = BaseOverlay("base.bit")
pAudio = base.audio

from scipy import signal
from scipy.fftpack import fft
get_ipython().run_line_magic('matplotlib', 'inline')

from bokeh.models.sources import ColumnDataSource
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, push_notebook

output_notebook()

# Define functions
def setVecSize():
    """Returns the initial vector that has the correct vector size we are going to use in the plot"""
    pAudio.record(.0671)
    af_uint8 = np.unpackbits(pAudio.buffer.astype(np.int16)
                         .byteswap(True).view(np.uint8))
    af_dec = signal.decimate(af_uint8,8,zero_phase=True)
    af_dec = signal.decimate(af_dec,6,zero_phase=True)
    af_dec = signal.decimate(af_dec,2,zero_phase=True)
    af_dec = (af_dec[50:-50]-af_dec[50:-50].mean())
    return af_dec

def blackmanHarrisWin(L):
    """Given the length L, returns the Blackman-Harris window"""

    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    nn = np.arange(L)
    NN = L
    DNom = NN - 1
    return a0 - a1*np.cos(2*np.pi*nn / DNom) + a2*np.cos(4*np.pi/DNom) - a3*np.cos(6*np.pi/DNom)

def getSpectrumUpdate(window):
    """Gets the next fft of data for plotting the spectrum"""
    pAudio.record(.0671)
    af_uint8 = np.unpackbits(pAudio.buffer.astype(np.int16)
                             .byteswap(True).view(np.uint8))
    af_dec = signal.decimate(af_uint8,8,zero_phase=True)
    af_dec = signal.decimate(af_dec,6,zero_phase=True)
    af_dec = signal.decimate(af_dec,2,zero_phase=True)
    af_dec = (af_dec[50:-50]-af_dec[50:-50].mean())
    yf = fft(af_dec*window)
    xf = np.arange(len(yf) + 1)
    xf = np.log2(xf[1:])
    return (xf[11:1025], yf[10:1024])

af_dec = setVecSize()
window = blackmanHarrisWin(len(af_dec))

# Bokeh is used here for continuously plotting the spectrum

my_figure = figure(plot_width=800, plot_height=400, y_range=[-.1, 2.0])
test_data = ColumnDataSource(data=dict(x=[0], y=[0]))
line = my_figure.line("x", "y", source=test_data)
handle = show(my_figure, notebook_handle=True)

new_data=dict(x=[0], y=[0])
x = []
y = []

n_show = 1010  # number of points to keep and show
while 1:
    x, y = getSpectrumUpdate(window)
    new_data['x'] = x 
    new_data['y'] = np.abs(y) 
    test_data.stream(new_data, n_show)

    push_notebook(handle=handle)
    
# Use the tools to zoom into the part of the spectrum you are interested in
    



