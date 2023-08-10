import random
import sys
import glob
import os
import time

import IPython
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram

import librosa
import librosa.display

from sklearn.preprocessing import normalize
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

# Note: SR stands for sampling rate, the rate at which my audio files were recorded and saved. 
SR = 22050 # All audio files are saved like this

def load_sounds_in_folder(foldername):
    """ Loads all sounds in a folder"""
    sounds = []
    for filename in os.listdir(foldername):
        X, sr = librosa.load(os.path.join(foldername,filename))
        assert sr == SR
        sounds.append(X)
    return sounds

## Sounds in which you can hear a bat are in the folder called "1". Others are in a folder called "0". 
batsounds = load_sounds_in_folder('labeled/1')
noisesounds = load_sounds_in_folder('labeled/0')

print("With bat: %d without: %d total: %d " % (len(batsounds), len(noisesounds), len(batsounds)+len(noisesounds)))
print("Example of a sound with a bat:")
IPython.display.display(IPython.display.Audio(random.choice(batsounds), rate=SR,autoplay=True))
print("Example of a sound without a bat:")
IPython.display.display(IPython.display.Audio(random.choice(noisesounds), rate=SR,autoplay=True))

def get_short_time_fourier_transform(soundwave):
    return librosa.stft(soundwave, n_fft=256)

def short_time_fourier_transform_amplitude_to_db(stft):
    return librosa.amplitude_to_db(stft)

def soundwave_to_np_spectogram(soundwave):
    step1 = get_short_time_fourier_transform(soundwave)
    step2 = short_time_fourier_transform_amplitude_to_db(step1)
    step3 = step2/100
    return step3

def inspect_data(sound):
    plt.figure()
    plt.plot(sound)
    IPython.display.display(IPython.display.Audio(sound, rate=SR))
    a = get_short_time_fourier_transform(sound)
    Xdb = short_time_fourier_transform_amplitude_to_db(a)
    plt.figure()
    plt.imshow(Xdb)
    plt.show()
    print("Length per sample: %d, shape of spectogram: %s, max: %f min: %f" % (len(sound), str(Xdb.shape), Xdb.max(), Xdb.min()))

inspect_data(batsounds[0])
inspect_data(noisesounds[0])

WINDOW_WIDTH = 10
AUDIO_WINDOW_WIDTH = 1000 # With sampling rate of 22050 we get 22 samples for our second of audio
def audio_to_metadata(audio):
    """ Takes windows of audio data, per window it takes the max value, min value, mean and stdev values"""
    features = []
    for start in range(0,len(audio)-AUDIO_WINDOW_WIDTH,AUDIO_WINDOW_WIDTH):
        subpart = audio[start:start+AUDIO_WINDOW_WIDTH]
        maxval = max(subpart)
        minval = min(subpart)
        mean = np.mean(subpart)
        stdev = np.std(subpart)
        features.extend([maxval,minval,mean,stdev,maxval-minval])
    return features

metadata = audio_to_metadata(batsounds[0])
print(metadata)
print(len(metadata))

# Meta-feature based batsounds and their labels
preprocessed_batsounds = list()
preprocessed_noisesounds = list()

for sound in batsounds:
    expandedsound = audio_to_metadata(sound)
    preprocessed_batsounds.append(expandedsound)
for sound in noisesounds:
    expandedsound = audio_to_metadata(sound)
    preprocessed_noisesounds.append(expandedsound)

labels = [0]*len(preprocessed_noisesounds) + [1]*len(preprocessed_batsounds)
assert len(labels) == len(preprocessed_noisesounds) + len(preprocessed_batsounds)
allsounds = preprocessed_noisesounds + preprocessed_batsounds
allsounds_normalized = normalize(np.array(allsounds),axis=1)
one_hot_labels = keras.utils.to_categorical(labels)
print(allsounds_normalized.shape)
print("Total noise: %d total bat: %d total: %d" % (len(allsounds_normalized), len(preprocessed_batsounds), len(allsounds)))

## Now zip the sounds and labels, shuffle them, and split into a train and testdataset
zipped_data = zip(allsounds_normalized, one_hot_labels)
np.random.shuffle(zipped_data)
random_zipped_data = zipped_data
VALIDATION_PERCENT = 0.8 # use X percent for training, the rest for validation
traindata = random_zipped_data[0:int(VALIDATION_PERCENT*len(random_zipped_data))]
valdata = random_zipped_data[int(VALIDATION_PERCENT*len(random_zipped_data))::]
indata = [x[0] for x in traindata]
outdata = [x[1] for x in traindata]
valin = [x[0] for x in valdata]
valout = [x[1] for x in valdata]

LEN_SOUND = len(preprocessed_batsounds[0])
NUM_CLASSES = 2 # Bat or no bat

model = Sequential()
model.add(Dense(128, activation='relu',input_shape=(LEN_SOUND,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2))
model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mae','accuracy'])
model.summary()
model.fit(np.array(indata), np.array(outdata), batch_size=64, epochs=10,verbose=2, shuffle=True) 
valresults = model.evaluate(np.array(valin), np.array(valout), verbose=0)
res_and_name = zip(valresults, model.metrics_names)
for result,name in res_and_name: 
    print("Validation " + name + ": " + str(result))

soundarray, sr = librosa.load("batsounds/bats9.m4a")
maxseconds = int(len(soundarray)/sr)
for second in range(maxseconds-1):
    audiosample = np.array(soundarray[second*sr:(second+1)*sr])
    metadata = audio_to_metadata(audiosample)
    testinput = normalize(np.array([metadata]),axis=1)
    prediction = model.predict(testinput)

    if np.argmax(prediction) ==1:
        IPython.display.display(IPython.display.Audio(audiosample, rate=sr,autoplay=True))
        librosa.output.write_wav('onedetectedbat.wav', audiosample, sr)

        time.sleep(2)
        print("Detected a bat at " + str(second) + " out of " + str(maxseconds) + " seconds")
        print(prediction)

