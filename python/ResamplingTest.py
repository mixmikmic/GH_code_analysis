import sys
sys.path.append('../python')

from HTKFeat import MFCC_HTK

import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as P

mfcc=MFCC_HTK()

signal = mfcc.load_raw_signal('../python-test/file.raw')

def draw_signal(signal, fs):
    sig_len=signal.size/fs #in seconds

    P.figure(figsize=(15,4))
    t=np.linspace(0,sig_len,signal.size)
    P.plot(t,signal)
    P.figure(figsize=(15,4))
    s=P.specgram(signal,Fs=fs)
    P.xlim(0,sig_len)
    P.figure(figsize=(15,4))
    f=np.linspace(0,fs/2,signal.size/2+1)
    P.plot(f,np.abs(np.fft.rfft(signal)))
    
draw_signal(signal,16000)

from ipywidgets import *
from IPython.display import display
import scipy.io.wavfile

def html5audio(signal, fs, filepath):

    scipy.io.wavfile.write(filepath,fs,signal.astype('int16'))

    src = "<audio controls=\"controls\" style=\"width:600px\" >        <source src=\"{}\" type=\"audio/wav\" />        Your browser does not support the audio element.        </audio>".format(filepath)
    
    display(HTML(src))

html5audio(signal,16000,'../python-test/file.wav')

from scipy.signal import resample

signal8k=resample(signal,signal.size/2)

draw_signal(signal8k,8000)

html5audio(signal8k,8000,'../python-test/file.wav')

