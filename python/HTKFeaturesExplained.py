import sys
sys.path.append('../python')

from HTKFeat import MFCC_HTK
from HTK import HCopy, HTKFile

import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as P

mfcc=MFCC_HTK()

signal = mfcc.load_raw_signal('../python-test/file.raw')

sig_len=signal.size/16000 #in seconds

P.figure(figsize=(15,4))
t=np.linspace(0,sig_len,signal.size)
P.plot(t,signal)
P.figure(figsize=(15,4))
s=P.specgram(signal,Fs=16000)
P.xlim(0,sig_len)

print "Before: "+str(np.mean(signal))

signal = signal - np.mean(signal)

print "After: "+str(np.mean(signal))

#these are a part of the configuration:
win_shift=160 
win_len=400

#the values above are in samples
#so 10ms * 16000 samples per second and 25ms * 16000 samples per second respectively

sig_len=len(signal)

#this is how we compute the window number while discarding the ones that don't fit the signal completely
win_num=np.floor((sig_len-win_len)/win_shift).astype('int')+1

wins=[]
for w in range(win_num):
    
    #these are the start and end of each window
    s=w*win_shift
    e=s+win_len
    
    #we need the copy of the data, because the numpy array slicing gives us a view of the data
    #and we don't won't to mess up the original signal when we start modifying the windows
    win=signal[s:e].copy()
    
    wins.append(win)
    
wins=np.asarray(wins)

k=0.97
h = [1,-k]

f=np.linspace(0,8000,257)
P.plot(f,np.abs(np.fft.rfft(h,n=512)))
P.xlabel('Frequency')
P.ylabel('Amplitude correction')

for win in wins:
    win-=np.hstack((win[0],win[:-1]))*k

rect = np.ones(400)

P.figure(figsize=(12,4))
P.subplot(1,2,1)
P.stem(rect)
P.xlim(-100,500)
P.ylim(-0.1,1.1)
P.title('Square window')

f=np.linspace(0,8000,257)
P.subplot(1,2,2)
P.plot(f,np.log(np.abs(np.fft.rfft(rect,n=512))))
P.xlabel('Frequency')
P.ylabel('Amplitude (log)')

hamm = np.hamming(400)

P.figure(figsize=(12,4))
P.subplot(1,2,1)
P.stem(hamm)
P.xlim(-100,500)
P.ylim(-0.1,1.1)
P.title('Hamming function')

f=np.linspace(0,8000,257)
P.subplot(1,2,2)
P.plot(f,np.log(np.abs(np.fft.rfft(hamm,n=512))))
P.xlabel('Frequency')
P.ylabel('Amplitude (log)')

for win in wins:
    win*=hamm

fft_len=np.asscalar(2**(np.floor(np.log2(win_len))+1).astype('int'))

ffts=[]
for win in wins:
    win=np.abs(np.fft.rfft(win,n=fft_len)[:-1])
    ffts.append(win)

ffts=np.asarray(ffts)

P.figure(figsize=(10,5))
P.pcolormesh(ffts.T,cmap='gray')
P.xlim(0,win_num)
P.ylim(0,fft_len/2)

freq2mel = lambda freq: 1127*(np.log(1+((freq)/700.0)))

f = np.linspace(0,8000,1000)
m = freq2mel(f)

P.plot(f,m)
P.xlabel('Frequency')
P.ylabel('Mel')

#this method creates the filters for our class
#normally it is run from the constructor
mfcc.create_filter(26)

P.figure(figsize=(15,3))
for f in mfcc.filter_mat.T:
    P.plot(f)
P.xlim(0,256)

mfcc.load_filter('../python-test/filter.csv')

P.figure(figsize=(15,3))
for f in mfcc.filter_mat.T:
    P.plot(f)
P.xlim(0,256)

melspec=[]
for f in ffts:
    m = np.dot(f,mfcc.filter_mat)
    melspec.append(m)
melspec=np.asarray(melspec)

P.figure(figsize=(15,5))
P.pcolormesh(melspec.T,cmap='gray')
P.xlim(0,win_num)
P.ylim(0,26)

melspec = np.log(melspec)

P.figure(figsize=(15,5))
P.pcolormesh(melspec.T,cmap='gray')
P.xlim(0,win_num)
P.ylim(0,26)

filter_num=26
mfcc_num=12

dct_base=np.zeros((filter_num,mfcc_num));
for m in range(mfcc_num):
    dct_base[:,m]=np.cos((m+1)*np.pi/filter_num*(np.arange(filter_num)+0.5))
    
P.figure(figsize=(6,3))
P.pcolormesh(dct_base.T,cmap='gray')
P.xlim(0,24)
P.ylim(0,12)
P.xlabel('Filters')
P.ylabel('MFCCs')

filter_num=26
mfcc_num=26

dct_base_f=np.zeros((filter_num,mfcc_num));
for m in range(mfcc_num):
    dct_base_f[:,m]=np.cos((m+1)*np.pi/filter_num*(np.arange(filter_num)+0.5))

f,ax=P.subplots(12,1,figsize=(8,8))
f.suptitle('Lower coefficients',fontsize=14)
for i in range(12):
    ax[i].plot(dct_base_f[:,i])
    ax[i].yaxis.set_visible(False)
    ax[i].xaxis.set_visible(False)
    
f,ax=P.subplots(14,1,figsize=(8,8))
f.suptitle('Higher coefficients',fontsize=14)
for i in range(12,26):
    ax[i-12].plot(dct_base_f[:,i])
    ax[i-12].yaxis.set_visible(False)
    ax[i-12].xaxis.set_visible(False)

filter_num=26
mfcc_num=12

mfccs=[]
for m in melspec:
    c=np.dot(m,dct_base)
    mfccs.append(c)
mfccs=np.asarray(mfccs)

P.figure(figsize=(15,5))
P.pcolormesh(mfccs.T,cmap='gray')
P.xlim(0,win_num)
P.ylim(0,mfcc_num)

mfnorm = np.sqrt(2.0 / filter_num)

mfccs*=mfnorm

lifter_num=22

lifter=1+(lifter_num/2)*np.sin(np.pi*(1+np.arange(mfcc_num))/lifter_num)

P.plot(lifter)

mfccs*=lifter

P.figure(figsize=(15,5))
P.pcolormesh(mfccs.T,cmap='gray')
P.xlim(0,win_num)
P.ylim(0,mfcc_num)

raw_energy=[]
for win in wins:
    raw_energy.append(np.log(np.sum(win**2)))
raw_energy=np.asarray(raw_energy)

ceps_energy=[]
for m in melspec:
    ceps_energy.append(np.sum(m)*mfnorm)
ceps_energy=np.asarray(ceps_energy)
    
P.figure(figsize=(15,5))
P.plot(raw_energy,'r',label='Raw energy')
P.plot(ceps_energy,'b',label='Cepstral energy')
P.xlim(0,win_num)
P.legend()

x=np.linspace(0,10,100)
y=x**3
y=y.reshape((y.size,1))

d=mfcc.get_delta(y,deltawin=2) #deltas
a=mfcc.get_delta(d,deltawin=2) #acceleration

#plots are normalized to <0,1> to fit on the same screen
P.figure(figsize=(10,5))
P.plot(x,y/np.max(y),label='Original - x^3')
P.plot(x,d/np.max(d),label='Delta - x^2')
P.plot(x,a/np.max(a),label='Accel. - x')
P.ylim(0,1)
P.legend(loc='upper left')

