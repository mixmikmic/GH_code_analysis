# Required packages

import pandas as pd
import numpy as np
import librosa
from os import listdir
import os
import sys
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import scipy

# Getting data

path = 'scenes_stereo_trainset/scenes_stereo_trainset/scenes_stereo/'
filenames = os.listdir(path)

filenames

# Creating columns for summary statistics

cols = []
for sum_key in ['mean','var','median','min','max','kurtosis','mean_diff','var_diff']:
    for i in range(20):
        cols+=[sum_key+str(i)]

print cols    

len(cols)

# Creating dataframe for training data

train = pd.DataFrame(index=range(len(filenames)),columns = cols)
train = train.fillna(0)

lst = map(lambda x:str(x),range(20))
print lst

len(filenames)

# Extracting MFCCs, calculating the summary stats 

t = []
for l in range(len(filenames)):
    #print l,'l'
    try:
        filename = path+filenames[l]
        y, sr = librosa.core.load(filename)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        S = librosa.feature.melspectrogram(y=y, n_mels=40,fmin=0,
                                       fmax= 22050)
        mfccs = librosa.feature.mfcc(S=librosa.logamplitude(S))
        #print 'mfccs',np.mean(mfccs[0])
        t.append(tempo)
        i = 0
        for ind in mfccs:
            #print ind,'ind'
            #print np.mean(ind),'realv'
            train.loc[l,'mean'+str(i)] = np.mean(ind)
            #print train.loc[l,'mffcs'+str(i)],'value'
            train.loc[l,'min'+str(i)] = np.min(ind)
            train.loc[l,'max'+str(i)] = np.max(ind)
            train.loc[l,'var'+str(i)] = np.var(ind)
            train.loc[l,'median'+str(i)] = np.median(ind)
            train.loc[l,'mean_diff'+str(i)] = np.mean(np.diff(ind))
            train.loc[l,'var_diff'+str(i)] = np.var(np.diff(ind))
            train.loc[l,'kurtosis'+str(i)] = scipy.stats.kurtosis(ind)
            i = i+1
            #print i,'i'
            
    except:
        pass      

len(train)

from sklearn import preprocessing
train_scaled = preprocessing.scale(train)

len(train)

files = filenames

scenes = []
for f in files:
    scenes.append(f[:-6])

train['scenes'] = scenes

train_scaled = pd.DataFrame(data = train_scaled)
train_scaled = train.columns

len(train.columns)

train['scenes'] = scenes

train

train_scaled.to_csv('training_dataset_scaled.csv')

train.to_csv('training_dataset.csv')

