#required libraries
import urllib
import scipy.io.wavfile
import pydub

#a temp folder for downloads
temp_folder="/Users/flavio.clesio/Downloads/audio_analysis/"
temp_folder

#spotify mp3 sample file
web_file="http://p.scdn.co/mp3-preview/35b4ce45af06203992a86fa729d17b1c1f93cac5"

#download file
urllib.urlretrieve(web_file,temp_folder+"file.mp3")
#read mp3 file
mp3 = pydub.AudioSegment.from_mp3(temp_folder+"file.mp3")
#convert to wav
mp3.export(temp_folder+"file.wav", format="wav")
#read wav file
rate,audData=scipy.io.wavfile.read(temp_folder+"file.wav")

# There's a bug in the code, reported in https://github.com/jiaaro/pydub/issues/189

#read mp3 file
mp3 = pydub.AudioSegment.from_mp3("Users/flavio.clesio/Downloads/file.mp3")

#convert to wav
mp3.export(temp_folder+"file.wav", format="wav")

#read wav file
rate,audData=scipy.io.wavfile.read(temp_folder+"file.wav")

print(rate)

print(audData)

#wav length
audData.shape[0] / rate

#wav number of channels mono/stereo
audData.shape[1]

#if stereo grab both channels
channel1=audData[:,0] #left
channel2=audData[:,1] #right

audData.dtype

#save wav file
scipy.io.wavfile.write(temp_folder+"file2.wav", rate, audData)

#save a file at half and double speed
scipy.io.wavfile.write(temp_folder+"file2.wav", rate/2, audData)
scipy.io.wavfile.write(temp_folder+"file2.wav", rate*2, audData)

#save a single channel
scipy.io.wavfile.write(temp_folder+"file2.wav", rate, channel1)

import numpy as np

#averaging the channels damages the music
mono=np.sum(audData.astype(float), axis=1)/2
scipy.io.wavfile.write(temp_folder+"file2.wav", rate, mono)

#Energy of music
np.sum(channel1.astype(float)**2)

#power - energy per unit of time
1.0/(2*(channel1.size)+1)*np.sum(channel1.astype(float)**2)/rate

import matplotlib.pyplot as plt

#create a time variable in seconds
time = np.arange(0, float(audData.shape[0]), 1) / rate

#plot amplitude (or loudness) over time
plt.figure(1)
plt.subplot(211)
plt.plot(time, channel1, linewidth=0.01, alpha=0.7, color='#ff7f00')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.subplot(212)
plt.plot(time, channel2, linewidth=0.01, alpha=0.7, color='#ff7f00')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

from numpy import fft as fft

fourier=fft.fft(channel1)

plt.plot(fourier, color='#ff7f00')
plt.xlabel('k')
plt.ylabel('Amplitude')

n = len(channel1)
fourier = fourier[0:(n/2)]

# scale by the number of points so that the magnitude does not depend on the length
fourier = fourier / float(n)

#calculate the frequency at each point in Hz
freqArray = np.arange(0, (n/2), 1.0) * (rate*1.0/n);

plt.plot(freqArray/1000, 10*np.log10(fourier), color='#ff7f00', linewidth=0.02)
plt.xlabel('Frequency (kHz)')
plt.ylabel('Power (dB)')

plt.figure(2, figsize=(8,6))
plt.subplot(211)
Pxx, freqs, bins, im = plt.specgram(channel1, Fs=rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))
cbar=plt.colorbar(im)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar.set_label('Intensity dB')
plt.subplot(212)
Pxx, freqs, bins, im = plt.specgram(channel2, Fs=rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))
cbar=plt.colorbar(im)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar.set_label('Intensity (dB)')
#plt.show()

np.where(freqs==10034.47265625)
MHZ10=Pxx[233,:]
plt.plot(bins, MHZ10, color='#ff7f00')

