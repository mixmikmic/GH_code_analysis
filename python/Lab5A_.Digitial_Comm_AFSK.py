# # # Serialize object
# f = open('jonsOH.pickle', 'wb')
# pickle.dump(y, f, protocol=pickle.HIGHEST_PROTOCOL)
# f.close()

# Import functions and libraries
get_ipython().magic('pylab')
import numpy as np
import matplotlib.pyplot as plt
import pyaudio, Queue, threading,time, sys, threading,time, serial
from __future__ import division
from numpy import pi, sin, zeros, r_
from scipy import signal
from rtlsdr import RtlSdr

import cPickle as pickle
# # # Serialize object
# f = open('pick_name.pickle', 'wb')
# pickle.dump(NN, f, protocol=pickle.HIGHEST_PROTOCOL)
# f.close()

# Load object
# f = open('linear_y_sine_.pickle', 'rb')
# Y = pickle.load(f)
# f.close()

get_ipython().magic('matplotlib inline')

# function to compute average power spectrum
def avgPS( x, N=256, fs=1):
    M = floor(len(x)/N)
    x_ = reshape(x[:M*N],(M,N)) * np.hamming(N)[None,:]
    X = np.fft.fftshift(np.fft.fft(x_,axis=1),axes=1)
    return r_[-N/2.0:N/2.0]/N*fs, mean(abs(X)**2,axis=0)


# Plot an image of the spectrogram y, with the axis labeled with time tl,
# and frequency fl
#
# t_range -- time axis label, nt samples
# f_range -- frequency axis label, nf samples
# y -- spectrogram, nf by nt array
# dbf -- Dynamic range of the spect

def sg_plot( t_range, f_range, y, dbf = 60, fig = None) :
    eps = 10.0**(-dbf/20.0)  # minimum signal
    
    # find maximum
    y_max = abs(y).max()
    
    # compute 20*log magnitude, scaled to the max
    y_log = 20.0 * np.log10( (abs( y ) / y_max)*(1-eps) + eps )
    
    # rescale image intensity to 256
    img = 256*(y_log + dbf)/dbf - 1
    
    fig=figure(figsize=(16,6))
    
    plt.imshow( np.flipud( 64.0*(y_log + dbf)/dbf ), extent= t_range  + f_range ,cmap=plt.cm.gray, aspect='auto')
    plt.xlabel('Time, s')
    plt.ylabel('Frequency, Hz')
    plt.tight_layout()
    
    return fig


def myspectrogram_hann_ovlp(x, m, fs, fc,dbf = 60):
    # Plot the spectrogram of x.
    # First take the original signal x and split it into blocks of length m
    # This corresponds to using a rectangular window %
    
    
    isreal_bool = isreal(x).all()
    
    # pad x up to a multiple of m 
    lx = len(x);
    nt = (lx + m - 1) // m
    x = append(x,zeros(-lx+nt*m))
    x = x.reshape((m/2,nt*2), order='F')
    x = concatenate((x,x),axis=0)
    x = x.reshape((m*nt*2,1),order='F')
    x = x[r_[m//2:len(x),ones(m//2)*(len(x)-1)].astype(int)].reshape((m,nt*2),order='F')
    
    
    xmw = x * hanning(m)[:,None];
    
    
    # frequency index
    t_range = [0.0, lx / fs]
    
    if isreal_bool:
        f_range = [ fc, fs / 2.0 + fc]
        xmf = np.fft.fft(xmw,len(xmw),axis=0)
        sg_plot(t_range, f_range, xmf[0:m/2,:],dbf=dbf)
        print 1
    else:
        f_range = [-fs / 2.0 + fc, fs / 2.0 + fc]
        xmf = np.fft.fftshift( np.fft.fft( xmw ,len(xmw),axis=0), axes=0 )
        sg_plot(t_range, f_range, xmf,dbf = dbf)
    
    return t_range, f_range, xmf

def play_audio( Q,ctrlQ ,p, fs , dev, ser="", keydelay=0):
    # play_audio plays audio with sampling rate = fs
    # Q - A queue object from which to play
    # ctrlQ - A queue object for ending the thread
    # p   - pyAudio object
    # fs  - sampling rate
    # dev - device number
    # ser - pyserial device to key the radio
    # keydelay - delay after keying the radio
    #
    #
    # There are two ways to end the thread: 
    #    1 - send "EOT" through  the control queue. This is used to terminate the thread on demand
    #    2 - send "EOT" through the data queue. This is used to terminate the thread when data is done. 
    #
    # You can also key the radio either through the data queu and the control queue
    
    
    # open output stream
    ostream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),output=True,output_device_index=dev)
    # play audio
    while (1):
        if not ctrlQ.empty():
            
            # control queue 
            ctrlmd = ctrlQ.get()
            if ctrlmd is "EOT"  :
                    ostream.stop_stream()
                    ostream.close()
                    print("Closed  play thread")
                    return;
            elif (ctrlmd is "KEYOFF"  and ser!=""):
                ser.setDTR(0)
                #print("keyoff\n")
            elif (ctrlmd is "KEYON" and ser!=""):
                ser.setDTR(1)  # key PTT
                #print("keyon\n")
                time.sleep(keydelay) # wait 200ms (default) to let the power amp to ramp up
                
        
        data = Q.get()
        
        if (data is "EOT") :
            ostream.stop_stream()
            ostream.close()
            print("Closed  play thread")
            return;
        elif (data is "KEYOFF"  and ser!=""):
            ser.setDTR(0)
            #print("keyoff\n")
        elif (data is "KEYON" and ser!=""):
            ser.setDTR(1)  # key PTT
            #print("keyon\n")
            time.sleep(keydelay) # wait 200ms (default) to let the power amp to ramp up
            
        else:
            try:
                ostream.write( data.astype(np.float32).tostring() )
            except:
                print("Exception")
                break
            
def record_audio( queue,ctrlQ, p, fs ,dev,chunk=1024):
    # record_audio records audio with sampling rate = fs
    # queue - output data queue
    # p     - pyAudio object
    # fs    - sampling rate
    # dev   - device number 
    # chunk - chunks of samples at a time default 1024
    #
    # Example:
    # fs = 44100
    # Q = Queue.queue()
    # p = pyaudio.PyAudio() #instantiate PyAudio
    # record_audio( Q, p, fs, 1) # 
    # p.terminate() # terminate pyAudio
    
   
    istream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),input=True,input_device_index=dev,frames_per_buffer=chunk)

    # record audio in chunks and append to frames
    frames = [];
    while (1):
        if not ctrlQ.empty():
            ctrlmd = ctrlQ.get()          
            if ctrlmd is "EOT"  :
                istream.stop_stream()
                istream.close()
                print("Closed  record thread")
                return;
        try:  # when the pyaudio object is distroyed stops
            data_str = istream.read(chunk) # read a chunk of data
        except:
            break
        data_flt = np.fromstring( data_str, 'float32' ) # convert string to float
        queue.put( data_flt ) # append to list



def printDevNumbers(p):
    N = p.get_device_count()
    for n in range(0,N):
        name = p.get_device_info_by_index(n).get('name')
        print n, name

p = pyaudio.PyAudio()
printDevNumbers(p)
p.terminate()

###### CHANGE #######
daudio_in = 1
daudio_out = 2
dusb_in = 3
dusb_out = 3

s = serial.Serial(port='/dev/tty.SLAB_USBtoUART')

# create an input output FIFO queues
Qin = Queue.Queue()
Qout = Queue.Queue()

# create a control fifo to kill threads when done
cQin = Queue.Queue()
cQout = Queue.Queue()

# create a pyaudio object
p = pyaudio.PyAudio()

# get sampling rate from pyaudio
fs_usb = p.get_device_info_by_index(dusb_in)['defaultSampleRate']


# initialize a recording thread. 
t_rec = threading.Thread(target = record_audio,   args = (Qin, cQin,p, fs_usb, dusb_in))

# initialize a playing thread. 
t_play = threading.Thread(target = play_audio,   args = (Qout,cQout,  p, fs_usb, daudio_out))

# start the recording and playing threads
t_rec.start()
t_play.start()

# give some time before starting
time.sleep(1)


# record and play about 10 seconds of audio 430*1024/44100 = 9.98 s
mxpwr = zeros(430)
rmspwr = zeros(430)

for n in range(0,430):
    
    samples = Qin.get()
    mxpwr[n] = max(abs(samples))
    rmspwr[n] = np.sqrt(np.sum(np.square(samples)))
    # You can add code here to do processing on samples in chunks of 1024
    # you will have to implement an overlap an add, or overlap an save to get
    # continuity between chunks
    
    Qout.put(samples)
    
    
#Close threads 
cQout.put('EOT')
cQin.put('EOT')
time.sleep(3) # give time for the thread to get killed


# clear Queues 

with Qin.mutex:
    Qin.queue.clear()
with Qout.mutex:
    Qout.queue.clear()
with cQin.mutex:
    cQin.queue.clear()
with cQout.mutex:
    cQout.queue.clear()
    
p.terminate()

def play( data, p, fs):
    # data - audio data array
    # p    - pyAudio object
    # fs    - sampling rate
    
    # open output stream
    ostream = p.open(format=pyaudio.paFloat32, channels=1, rate=fs,output=True)
    # play audio
    ostream.write( data.astype(np.float32).tostring() )

fig = figure(figsize=(16,4))
t = r_[0:430]*1024.0/44100
plt.plot(t,mxpwr)
plt.plot(t,rmspwr/sqrt(1024))
plt.title('Maximum/RMS power')
plt.legend(('Max signal','rms power'))

print('Avg maxpwr',np.mean(mxpwr))
if any(mxpwr > 0.95):
    print("Warning! Signal is clipped. Reduce radio volume, and/or usb device input volume")
if max(mxpwr) < 0.3:
    print("Audio Volume may be too low. Increase it, for better lab performance")

# s = serial.Serial(port='/dev/tty.SLAB_USBtoUART')
s.setDTR(0)

for n in range(0,10):
    s.setDTR(1)
    time.sleep(0.25)
    s.setDTR(0)
    time.sleep(0.25)

# creates a queue
Qout = Queue.Queue()
cQout = Queue.Queue()

# initialize a serial port (use COM1-9 for windows) 
# s = serial.Serial(port='/dev/tty.SLAB_USBtoUART')
s.setDTR(0)

# create a pyaudio object
p = pyaudio.PyAudio()

# get sampling rate
fs_usb = p.get_device_info_by_index(dusb_out)['defaultSampleRate']

# generate sinusoids
t = r_[0:2*fs_usb]/fs_usb
sig2 = 0.5*sin(2*pi*2000*t)
sig1 = 0.5*sin(2*pi*1000*t)

Qout.put("KEYON")
Qout.put(sig2)
Qout.put("KEYOFF")
Qout.put(sig2)
Qout.put("KEYON")
Qout.put(sig1)
Qout.put("KEYOFF")
Qout.put("EOT")


# play audio from Queue 

play_audio(Qout, cQout, p, fs_usb, dusb_out, s,0.2)

time.sleep(2)
p.terminate()

# creates a queue
Qout = Queue.Queue()
cQout = Queue.Queue()

# initialize a serial port (use COM1-9 for windows) 
# s = serial.Serial(port='/dev/tty.SLAB_USBtoUART')
s.setDTR(0)

# create a pyaudio object
p = pyaudio.PyAudio()

# get sampling rate
fs_usb = p.get_device_info_by_index(dusb_out)['defaultSampleRate']

t = r_[0:2*fs_usb]/fs_usb
sig2 = 0.25*sin(2*pi*2000*t)
sig1 = 0.25*sin(2*pi*1000*t)

t_play = threading.Thread(target = play_audio, args = (Qout, cQout,  p, fs_usb, dusb_out, s ,0.2 ))


# play audio from Queue 
t_play.start()

Qout.put("KEYON")
Qout.put(sig2)
Qout.put("KEYOFF")
Qout.put(sig2)
Qout.put("KEYON")
Qout.put(sig1)
Qout.put("KEYOFF")
Qout.put("EOT")

# must wait for the queue to empty before terminating pyaudio
while not(Qout.empty()) :
    time.sleep(1)


time.sleep(1)
p.terminate()

# initialize a serial port (use COM1-9 for windows) 
# s = serial.Serial(port='/dev/tty.SLAB_USBtoUART')
s.setDTR(0)

# Setup SDR
fs_sdr = fs = 240000
fc =    144.970e6 # set your frequency!
ppm =   26 # set estimate ppm
gain =  26 # set gain

sdr = RtlSdr()
sdr.sample_rate = fs_sdr    # sampling rate
sdr.gain = gain
sdr.center_freq = fc
sdr.set_freq_correction(ppm)

# start transmitting
s.setDTR(1)
y = sdr.read_samples(256000*6)
# stop transmitting
s.setDTR(0)
sdr.close()

# Code to plot magnitude signal and compute frequency offset
# Here:
f, sp = avgPS(y,N=256,fs=fs_usb)
fig = figure(figsize=(8,4))
plot(f,10*log10(sp))
title('average power spectrum of signal')
xlabel('frequency offset [KHz]')
offset = f[np.argmax(10*log(sp))]
print('Offset:', offset)

# plot 
fig = figure(figsize=(16,4))
figure()
plot(abs(y))
title('Magnitude signal')
xlabel('t [ms]')

f0 = offset = f[np.argmax(10*log(sp))]
ppmcalib = offset * 1e6  / fc

print 'shift in Hz:', f0
print 'shift in ppm:',ppmcalib

# generate the tone
# your code here:

# Initial pyaudio and Queues
p = pyaudio.PyAudio()
Q = Queue.Queue()
cQ = Queue.Queue()

# initialize a serial port (use COM1-9 for windows) 
s = serial.Serial(port='/dev/tty.SLAB_USBtoUART')
s.setDTR(0)

# get sampling rate
fs_usb = p.get_device_info_by_index(dusb_out)['defaultSampleRate']

# generate sinusoids
t = r_[0:4*fs_usb]/fs_usb
tone_freq = 2200*2*pi
sig = (t/max(t))*sin(tone_freq*t)*(0.1)

# Setup SDR
fs_sdr = 240000
fc = fc    # center     # set your frequency!
ppm =   22 # set estimate ppm
gain =  20 # set gain


sdr = RtlSdr()
sdr.sample_rate = fs_sdr    # sampling rate
sdr.gain = gain
sdr.center_freq = fc
sdr.set_freq_correction(ppm)


# Fill the queue
Q.put(zeros(fs_usb/2))  # wait 1/2 second
Q.put('KEYON')          # start transmitting
Q.put(zeros(fs_usb/2))  # wait 1/2 second 
Q.put(sig)              # start playing audio
Q.put('KEYOFF')         # stop transmitting
Q.put('EOT')         # exit thread

# initialize thread
t_play = threading.Thread(target = play_audio, args = (Q,cQ, p, fs_usb, dusb_out,s  ))

# start transmitting
t_play.start()

# read samples from SDR
y = sdr.read_samples(256000*6)

# stop transmitting when done
s.setDTR(0)
sdr.close()

# empty queue
while not(Q.empty()):
    time.sleep(1)

# terminate
time.sleep(2)
p.terminate()

#your code here:
fc = fc
fs_usb = 44100
f, sp = avgPS(y,N=256,fs=fs_usb)
fig = figure(figsize=(8,4))
plot(f,10*log10(sp))
title('average power spectrum')
xlabel('frequency offset [Hz]')

# plot 
fig = figure(figsize=(16,4))
plot(abs(y))
title('Magnitude signal')

f0 = offset = f[np.argmax(10*log(sp))]
ppmcalib = offset * 1e6  / fc

y_cropped = y[len(y)//10 : len(y)*7//10]
fig = figure(figsize=(16,4))
plot(abs(y_cropped))
title('Magnitude of cropped signal')

print 'shift in Hz:', f0
print 'shift in ppm:',ppmcalib

# Your code here
m = 256 # window length
myspectrogram_hann_ovlp(y,m,fs_sdr,0);

# # # Serialize object
# f = open('linear_y_sine_.pickle', 'wb')
# pickle.dump(y, f, protocol=pickle.HIGHEST_PROTOCOL)
# f.close()

# Load object
# f = open('linear_y_sine_.pickle', 'rb')
# yyyy = pickle.load(f)
# f.close()

hdiff = lambda s: signal.fftconvolve(s,
                                    signal.remez(31, [0.0,10000.0,12000.0,12000.0],[1.05/1.2,0], Hz = 24e3, type='differentiator'),
                                    mode='same')

demod = lambda s: np.imag(np.conj(s)*hdiff(s))

# your code here
m = 256
dd = 10
h = signal.firwin(513, cutoff=15e3/120e3)
filtered_data = signal.fftconvolve(y, h, 'same')

myspectrogram_hann_ovlp(filtered_data,m,fs_sdr,0);

ds = filtered_data[::dd]
myspectrogram_hann_ovlp(ds,m,fs_sdr/dd,0);

# demoded = demod(filtered_data)[::5]

demoded = demod(ds/np.abs(ds))
myspectrogram_hann_ovlp(demoded,m,fs_sdr/dd,0);

# p2 = pyaudio.PyAudio()
# play(demoded_norm, p2, 48000)
# p2.terminate()

# your code here
freq = 2200
t = np.linspace(0,513/(fs_sdr/10),513)
modulator = np.exp(2j*pi*freq*t)
ssb_filter = np.hanning(513)*modulator

y_ssb_filt = signal.fftconvolve(demoded, ssb_filter, 'same')

plot(np.abs(y_ssb_filt))
plt.title('Magnitude of SSB filtered data')
plt.xlabel('time [samples]')

myspectrogram_hann_ovlp(y_ssb_filt,m,fs_sdr/10,0);

mxamp = np.abs(y_ssb_filt)[np.argmax(np.abs(y_ssb_filt))]/4
print(mxamp)

def genChirpPulse(Npulse, f0, f1, fs):
    #     Function generates an analytic function of a chirp pulse
    #     Inputs:
    #             Npulse - pulse length in samples
    #             f0     - starting frequency of chirp
    #             f1     - end frequency of chirp
    #             fs     - sampling frequency
    
    t1 = r_[0.0:Npulse]/fs
    Tpulse = float32(Npulse) / fs 
    f_of_t = f0 + (t1) / Tpulse * (f1 - f0)
    phi_of_t = 2*pi*np.cumsum(f_of_t)/fs
    pulse = exp(1j* phi_of_t )
    return f_of_t, pulse

# Generating chirp.

f0 = 20 # should be e3
f1 = 5e3 # should be e3
f_, chirp = genChirpPulse(4*fs_usb, f0, f1, fs=fs_usb)
chirp = chirp/40

# Playing chirp.

# p2 = pyaudio.PyAudio()
# print('chirp len', len(chirp)/48000, 'seconds')
# play(chirp, p2, 48000)
# p2.terminate()

_, fft_chirp = signal.freqz(chirp)
phase = np.angle(fft_chirp)
# f = np.linspace(min(f0,f1), max(f0,f1), len(fft_chirp))
plt.plot(_*(fs_sdr/np.pi), abs((fft_chirp)))
plt.title("Received chirp magnitude frequency response")
plt.xlabel("Frequency (Hz)")

figure()
plt.plot(_*(fs_sdr/np.pi), phase)
plt.title("Received chirp phase response")
plt.xlabel("Frequency (Hz)")

plot(f_[9000:9100], chirp[9000:9100])

### START: transmit and record ###

# Initial pyaudio and Queues
p = pyaudio.PyAudio()
Q = Queue.Queue()
cQ = Queue.Queue()

# initialize a serial port (use COM1-9 for windows) 
s = serial.Serial(port='/dev/tty.SLAB_USBtoUART')
s.setDTR(0)

# get sampling rate
fs_usb = p.get_device_info_by_index(dusb_out)['defaultSampleRate']

# Setup SDR
fs_sdr = 240000
fc = fc        # center     # set your frequency!
gain = 10
ppm = 27

sdr = RtlSdr()
sdr.sample_rate = fs_sdr    # sampling rate
sdr.gain = gain
sdr.center_freq = fc
sdr.set_freq_correction(ppm)

# Fill the queue
Q.put(zeros(fs_usb/2))  # wait 1/2 second
Q.put('KEYON')          # start transmitting
Q.put(zeros(fs_usb/2))  # wait 1/2 second 
Q.put(chirp)                                                                         # start playing: chirp
Q.put('KEYOFF')         # stop transmitting
Q.put('EOT')         # exit thread
 
# initialize thread
t_play = threading.Thread(target = play_audio, args = (Q,cQ, p, fs_usb, dusb_out,s  ))

# start transmitting
t_play.start()

# read samples from SDR
y = sdr.read_samples(256000*7)

# stop transmitting when done
s.setDTR(0)
sdr.close()

# empty queue
while not(Q.empty()) :
    time.sleep(1)

# terminate
time.sleep(2)
p.terminate()

### END: transmit and record ###

m = 256 # window length
myspectrogram_hann_ovlp(y,m,fs_sdr,0);

dd = 10
h = signal.firwin(513, cutoff=15e3/120e3)
filtered_data = signal.fftconvolve(y, h, 'same')

myspectrogram_hann_ovlp(filtered_data,m,fs_sdr,0);

ds = filtered_data[::dd]
myspectrogram_hann_ovlp(ds,m,fs_sdr/dd,0);

# demoded = demod(filtered_data)[::5]

demoded = demod(ds/np.abs(ds))
myspectrogram_hann_ovlp(demoded,m,fs_sdr/dd,0);

# your code here
freq = 20 # changed from 2.2e3 ! because of chirp freq
t = np.linspace(0,513/(fs_sdr/10),513)
modulator = np.exp(2j*pi*freq*t)
ssb_filter = np.hanning(513)*modulator

y_ssb_filt = signal.fftconvolve(demoded, ssb_filter, 'same')[30e3:-2000]

plot(np.abs(y_ssb_filt))
plt.title('Magnitude of SSB filtered data')
plt.xlabel('time [samples]')

myspectrogram_hann_ovlp(y_ssb_filt,m,fs_sdr/10,0);

_, fft_chirp = signal.freqz(y_ssb_filt)
phase = np.angle(fft_chirp)

plt.plot(_*(fs_sdr/np.pi), abs((fft_chirp)))
plt.title("Received chirp magnitude frequency response")
plt.xlabel("Frequency (Hz)")

figure()
plt.plot(_*(fs_sdr/np.pi), phase)
plt.title("Received chirp phase response")
plt.xlabel("Frequency (Hz)")

#Your code here:

# generate the tone with appropriate gain
# your code here:

scale = 1
noise_sig = np.random.randn(44.1e3*4)/scale

### START: transmit and record ###

# Initial pyaudio and Queues
p = pyaudio.PyAudio()
Q = Queue.Queue()
cQ = Queue.Queue()

# initialize a serial port (use COM1-9 for windows) 
s = serial.Serial(port='/dev/tty.SLAB_USBtoUART')
s.setDTR(0)

# get sampling rate
fs_usb = p.get_device_info_by_index(dusb_out)['defaultSampleRate']

# Setup SDR
fs_sdr = 240000
fc = fc        # center     # set your frequency!
gain = 10
ppm = 27

sdr = RtlSdr()
sdr.sample_rate = fs_sdr    # sampling rate
sdr.gain = gain
sdr.center_freq = fc
sdr.set_freq_correction(ppm)

# Fill the queue
Q.put(zeros(fs_usb/2))  # wait 1/2 second
Q.put('KEYON')          # start transmitting
Q.put(zeros(fs_usb/2))  # wait 1/2 second 
Q.put(noise_sig)        # start playing: noise
Q.put('KEYOFF')         # stop transmitting
Q.put('EOT')         # exit thread
 
# initialize thread
t_play = threading.Thread(target = play_audio, args = (Q,cQ, p, fs_usb, dusb_out,s  ))

# start transmitting
t_play.start()

# read samples from SDR
y = sdr.read_samples(256000*7)

# stop transmitting when done
s.setDTR(0)
sdr.close()

# empty queue
while not(Q.empty()) :
    time.sleep(1)

# terminate
time.sleep(2)
p.terminate()

### END: transmit and record ###

# Your demodulation code

m = 256 # window length
myspectrogram_hann_ovlp(y,m,fs_sdr,0);

dd = 10
h = signal.firwin(513, cutoff=15e3/120e3)
filtered_data = signal.fftconvolve(y, h, 'same')

myspectrogram_hann_ovlp(filtered_data,m,fs_sdr,0);

ds = filtered_data[::dd]
myspectrogram_hann_ovlp(ds,m,fs_sdr/dd,0);

# demoded = demod(filtered_data)[::5]

demoded = demod(ds/np.abs(ds))
myspectrogram_hann_ovlp(demoded,m,fs_sdr/dd,0);

# your code here
freq = 2200
t = np.linspace(0,513/(fs_sdr/10),513)
modulator = np.exp(2j*pi*freq*t)
ssb_filter = np.hanning(513)*modulator

y_ssb_filt = signal.fftconvolve(demoded, ssb_filter, 'same')[30e3:-2000]

plot(np.abs(y_ssb_filt))
plt.title('Magnitude of SSB filtered data')
plt.xlabel('time [samples]')

myspectrogram_hann_ovlp(y_ssb_filt,m,fs_sdr/10,0);

def text2Morse(text,fc,fs,dt):
    CODE = {'A': '.-',     'B': '-...',   'C': '-.-.', 
        'D': '-..',    'E': '.',      'F': '..-.',
        'G': '--.',    'H': '....',   'I': '..',
        'J': '.---',   'K': '-.-',    'L': '.-..',
        'M': '--',     'N': '-.',     'O': '---',
        'P': '.--.',   'Q': '--.-',   'R': '.-.',
     	'S': '...',    'T': '-',      'U': '..-',
        'V': '...-',   'W': '.--',    'X': '-..-',
        'Y': '-.--',   'Z': '--..',
        
        '0': '-----',  '1': '.----',  '2': '..---',
        '3': '...--',  '4': '....-',  '5': '.....',
        '6': '-....',  '7': '--...',  '8': '---..',
        '9': '----.',

        ' ': ' ', "'": '.----.', '(': '-.--.-',  ')': '-.--.-',
        ',': '--..--', '-': '-....-', '.': '.-.-.-',
        '/': '-..-.',   ':': '---...', ';': '-.-.-.',
        '?': '..--..', '_': '..--.-'
        }
    
    Ndot= 1.0*fs*dt
    Ndah = 3*Ndot
    
    sdot = sin(2*pi*fc*r_[0.0:Ndot]/fs)
    sdah = sin(2*pi*fc*r_[0.0:Ndah]/fs)
    
    # convert to dit dah
    mrs = ""
    for char in text:
        mrs = mrs + CODE[char.upper()] + "*"
    
    sig = zeros(1)
    for char in mrs:
        if char == " ":
            sig = concatenate((sig,zeros(Ndot*7)))
        if char == "*":
            sig = concatenate((sig,zeros(Ndot*3)))
        if char == ".":
            sig = concatenate((sig,sdot,zeros(Ndot)))
        if char == "-":
            sig = concatenate((sig,sdah,zeros(Ndot)))
    return sig
            
    

p = pyaudio.PyAudio()

fs = 240000
fc =  fc
sdr = RtlSdr()
sdr.sample_rate = fs    # sampling rate
sdr.gain = gain
sdr.center_freq = fc
sdr.set_freq_correction(ppm)


Q = Queue.Queue()
cQ = Queue.Queue()


# initialize a serial port (use COM1-9 for windows) 
s = serial.Serial(port='/dev/tty.SLAB_USBtoUART')
s.setDTR(0)

# get sampling rate
fs_usb = p.get_device_info_by_index(dusb_out)['defaultSampleRate']


callsign = text2Morse("KM6BJU KM6BJU TESTING",850,44100,75e-3)*0.1


Q.put(r_[0:44100.0*0.3]*0)
Q.put(callsign)
Q.put('KEYOFF')
Q.put('EOT')

t_play = threading.Thread(target = play_audio,   args = (Q, cQ,  p, 44100, dusb_out ,s ))

s.setDTR(1)
t_play.start()

y = sdr.read_samples(256000*20)
sdr.close()
s.setDTR(0)

while not(Q.empty()) :
    time.sleep(1)
    
p.terminate()
tt,ff,xmf = myspectrogram_hann_ovlp(y, 512, fs, fc,dbf = 60)



