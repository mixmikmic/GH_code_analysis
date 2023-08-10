import numpy, scipy, matplotlib.pyplot as plt, pandas, librosa,sklearn,IPython.display as ipd
import matplotlib.style as ms
ms.use('seaborn-muted')
import config,functions
get_ipython().magic('matplotlib inline')

float_formatter = lambda x: "%.4f" % x
numpy.set_printoptions(formatter={'float_kind':float_formatter})

ipd.Image(filename='kulak.jpg')

ipd.Image(filename='low.jpg')

ipd.Image(filename='high.jpg')

sampling_rate=22050

x_metal,sr=librosa.load("./genres/metal/metal.00001.au",sr=sampling_rate,mono=True,duration=5.0)
x_classical,sr=librosa.load("./genres/classical/classical.00001.au",sr=sampling_rate,mono=True,duration=5.0)

print("Overview of the array of samples for Metal song:(size={})\n{}:\n ".format(x_metal.shape,x_metal))
print("Overview of the array of samples for Classical song:(size={})\n{}".format(x_classical.shape,x_classical))

#Waveplot - Tİme Domain

fig=plt.figure(figsize=(12,5))
fig.subplots_adjust(hspace=.5)

plt.subplot(2, 1, 1)
plt.xlabel("Time")
plt.ylabel("Amplitude")
librosa.display.waveplot(x_metal,sampling_rate)
plt.title('Metal')

plt.subplot(2, 1, 2)
plt.ylim(-0.4,0.4)
plt.xlabel("Time")
plt.ylabel("Amplitude")
librosa.display.waveplot(x_classical,sampling_rate)
plt.title('Clasical')

ipd.Audio(x_metal, rate=sampling_rate)

ipd.Audio(x_classical, rate=sampling_rate)

hop_size=512
frame_size=2048

#Segmentation
F=librosa.util.frame(x_metal,frame_length=frame_size,hop_length=hop_size)
x,y=F.shape

print("Number of frames={} with frame size={} for Metal song".format(y,x))
print("Total number of samples= {}\n".format(y*x+hop_size))
print(F)

F2=librosa.util.frame(x_classical,frame_length=frame_size,hop_length=hop_size)
x2,y2=F2.shape

print("\nNumber of frames={} with frame size={} for Classical song".format(y2,x2))
print("Total number of samples= {}\n".format(y2*x2+hop_size))
print(F2)

#Onset Detection
onset_frames = librosa.onset.onset_detect(x_metal, sr=sampling_rate,hop_length=hop_size)
onset_times = librosa.frames_to_time(onset_frames)
x=onset_frames.size
print("Number of detected onsets for Metal song={}".format(x))
print(onset_frames)


onset_frames2 = librosa.onset.onset_detect(x_classical, sr=sampling_rate,hop_length=hop_size)
onset_times2 = librosa.frames_to_time(onset_frames2)
x2=onset_frames2.size
print("\nNumber of detected onsets for Classical song={}".format(x2))
print(onset_frames2)

fig=plt.figure(figsize=(12,5))
fig.subplots_adjust(hspace=.5)

plt.subplot(2,1,1)
librosa.display.waveplot(x_metal, sr=sampling_rate)
plt.vlines(onset_times*1000, -0.8, 0.79, color='r', alpha=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title('Metal')

plt.subplot(2,1,2)
librosa.display.waveplot(x_classical, sr=sampling_rate)
plt.vlines(onset_times2*1000, -0.8, 0.79, color='r', alpha=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title('Classical')

clicks = librosa.clicks(frames=onset_frames, sr=sampling_rate, length=len(x_metal))
ipd.Audio(x_metal + clicks, rate=sampling_rate)

clicks = librosa.clicks(frames=onset_frames2, sr=sampling_rate, length=len(x_classical))
ipd.Audio(x_classical + clicks, rate=sampling_rate)

ipd.Image(filename='time_and_freq.jpg')

ipd.display(ipd.Image(url='https://upload.wikimedia.org/wikipedia/commons/7/72/Fourier_transform_time_and_frequency_domains_%28small%29.gif'))

# Fourier Transform and Spectrum

fig=plt.figure(figsize=(12,5))
fig.subplots_adjust(hspace=.5)

plt.subplot(2,1,1)
FFT_Metal=scipy.fft(x_metal)
FFT_Metal_Mag=numpy.absolute(FFT_Metal)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title('Metal')
plt.plot(FFT_Metal_Mag)

plt.subplot(2,1,2)
FFT_Classical=scipy.fft(x_classical)
FFT_Classical_Mag=numpy.absolute(FFT_Classical)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title('Classical')
plt.plot(FFT_Classical_Mag)

#Short-Time Fourier Transform and Spectogram

window_size=2048

fig=plt.figure(figsize=(12,5))
fig.subplots_adjust(hspace=.5)

plt.subplot(2,1,1)
S = librosa.feature.melspectrogram(x_metal, sr=sampling_rate, n_fft=window_size)
logS = librosa.logamplitude(S)
plt.title('Metal')
librosa.display.specshow(logS, sr=sampling_rate, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')

plt.subplot(2,1,2)
S2 = librosa.feature.melspectrogram(x_classical, sr=sampling_rate, n_fft=window_size)
logS2 = librosa.logamplitude(S2)
plt.title('Classical')
librosa.display.specshow(logS2, sr=sampling_rate, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')

fig=plt.figure(figsize=(12,5))
fig.subplots_adjust(hspace=.5)

plt.subplot(2,1,1)
S = librosa.feature.melspectrogram(x_metal, sr=sampling_rate, n_fft=window_size)
logS = librosa.logamplitude(S)
plt.title('Metal')
librosa.display.specshow(logS, sr=sampling_rate, x_axis='time', y_axis='log')
plt.vlines(onset_frames, 0, logS.shape[0], color='k', alpha=0.8)
plt.colorbar(format='%+2.0f dB')

plt.subplot(2,1,2)
S2 = librosa.feature.melspectrogram(x_classical, sr=sampling_rate, n_fft=window_size)
logS2 = librosa.logamplitude(S2)
plt.title('Classical')
librosa.display.specshow(logS2, sr=sampling_rate, x_axis='time', y_axis='log')
plt.vlines(onset_frames2, 0, logS.shape[0], color='k', alpha=0.8)
plt.colorbar(format='%+2.0f dB')

#Canstant-Q Transform and Chromagram

fig=plt.figure(figsize=(12,5))
fig.subplots_adjust(hspace=.5)

plt.subplot(2,1,1)
C = librosa.cqt(x_metal, sr=sampling_rate,hop_length=hop_size,real=False,fmin=librosa.note_to_hz('C2'))
logC = librosa.logamplitude(C)
plt.title('Metal')
librosa.display.specshow(logC, sr=sampling_rate, x_axis='time', y_axis='cqt_note',cmap='coolwarm',)
plt.colorbar(format='%+2.0f dB')

plt.subplot(2,1,2)
C2 = librosa.cqt(x_classical, sr=sampling_rate,hop_length=hop_size,real=False,fmin=librosa.note_to_hz('C2'))
logC2 = librosa.logamplitude(C2)
plt.title('Classical')
librosa.display.specshow(logC2, sr=sampling_rate, x_axis='time', y_axis='cqt_note',cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')

#Canstant-Q Transform and Chromagram with Pitch Classes

fig=plt.figure(figsize=(12,5))
fig.subplots_adjust(hspace=.5)

plt.subplot(2,1,1)
C = librosa.feature.chroma_cqt(x_metal, sr=sampling_rate,hop_length=hop_size,fmin=librosa.note_to_hz('C2'))
plt.title('Metal')
librosa.display.specshow(C, sr=sampling_rate, x_axis='time', y_axis='chroma',cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')

plt.subplot(2,1,2)
C2 = librosa.feature.chroma_cqt(x_classical, sr=sampling_rate,fmin=librosa.note_to_hz('C2'))
plt.title('Classical')
librosa.display.specshow(C2, sr=sampling_rate, x_axis='time', y_axis='chroma',cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')

