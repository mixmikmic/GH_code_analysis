import glob
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import time
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')

def windows(data, window_size):
    start = 0
    while True:
        yield start, start + window_size
        if(start+ window_size >= len(data)):
            break
        start += (window_size // 2)

#for very short wav file, I still want to keep it.
def extract_features(path_dir, ds_filename, fold_num, n_mfcc = 20, frames = 41):
    window_size = 512 * (frames - 1)
    mfccs = []
    labels = []
    labels_name = []
    file_name = []
    df = pd.read_csv(ds_filename, index_col = False)
    
    for row in df.itertuples():
        if(row[6]==fold_num):
            sound_clip,s = librosa.load(path_dir + str(row[6])+"/" + str(row[1]))
            
            for (start,end) in windows(sound_clip,window_size):
                if(len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                elif (start==0):
                    signal = np.lib.pad(sound_clip[start:], (0, window_size - len(sound_clip[start: end])), 'constant', constant_values = (0))
                else: 
                    break
                mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc = n_mfcc).T.flatten()[:, np.newaxis].T
                mfccs.append(mfcc)
                labels.append(row[7])
                labels_name.append(row[8])
                file_name.append(row[1])
    
    features = np.asarray(mfccs).reshape(len(mfccs),frames, n_mfcc)
    return np.array(features), np.array(labels,dtype = np.int), np.array(labels_name), np.array(file_name)

def extract_features_delta(path_dir, ds_filename, fold_num, n_mels = 60, frames = 41):
    window_size = 512 * (frames - 1)
    #mfccs = []
    log_specgrams = []
    labels = []
    labels_name = []
    file_name = []
    df = pd.read_csv(ds_filename, index_col = False)
    
    for row in df.itertuples():
        if(row[6]==fold_num):
            sound_clip,s = librosa.load(path_dir + str(row[6])+"/" + str(row[1]))
            
            for (start,end) in windows(sound_clip,window_size):
                if(len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                elif (start==0):
                    signal = np.lib.pad(sound_clip[start:], (0, window_size - len(sound_clip[start: end])), 'constant', constant_values = (0))
                else: 
                    break
                melspec = librosa.feature.melspectrogram(signal, n_mels = n_mels)
                logspec = librosa.logamplitude(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)

                labels.append(row[7])
                labels_name.append(row[8])
                file_name.append(row[1])
                
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),n_mels,frames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    
    #features = np.asarray(mfccs).reshape(len(mfccs),frames, n_mfcc)
    return np.array(features), np.array(labels,dtype = np.int), np.array(labels_name), np.array(file_name)

mega_file = './UrbanSound8K/metadata/UrbanSound8K.csv' 
path_dir = "./UrbanSound8K/audio/fold"

start_time = time.time()
v_n_mels = 60
v_frames = 41
for i in range(1, 11):
    features, labels, labels_name, file_name = extract_features_delta(path_dir, mega_file, i, n_mels = v_n_mels, frames = v_frames)
    feature_pd = pd.DataFrame(features.reshape((-1, v_n_mels * v_frames * 2))) 
    feature_pd["label"] = labels
    feature_pd["labels_name"] = labels_name
    feature_pd["file_name"]= file_name
    feature_pd.to_csv("./UrbanSound8K/audio/delta" + str(i) + ".csv", index = False)
    print ("fold"+str(i) +" finished transferring.")
    
end_time = time.time()
print("seconds:", end_time-start_time)

start_time = time.time()
for i in range(1, 11):
    features, labels, labels_name, file_name = extract_features(path_dir, mega_file, i, n_mfcc = 20)
    feature_pd = pd.DataFrame(features.reshape((-1, 820))) 
    feature_pd["label"] = labels
    feature_pd["labels_name"] = labels_name
    feature_pd["file_name"]= file_name
    feature_pd.to_csv(path_dir + str(i) + "/" + "mfcc_f" + str(i) + ".csv", index = False)
    print ("fold"+str(i) +" finished transferring.")
    
end_time = time.time()
print("seconds:", end_time-start_time)

def extract_feature_193(file_name):
    X, sample_rate = librosa.load(file_name)
    
    """
    The short-time Fourier transform (STFT), or alternatively short-term Fourier transform, 
    is a Fourier-related transform used to determine the sinusoidal frequency and phase content 
    of local sections of a signal as it changes over time. In practice, the procedure for 
    computing STFTs is to divide a longer time signal into shorter segments of equal length 
    and then compute the Fourier transform separately on each shorter segment. This reveals 
    the Fourier spectrum on each shorter segment. One then usually plots the changing spectra 
    as a function of time.
    """
    
    stft = np.abs(librosa.stft(X))
    """
    The most commonly used feature extraction method in automatic speech recognition (ASR) is Mel-Frequency 
    Cepstral Coefficients (MFCC) [1]. This feature extraction method was first mentioned by Bridle and Brown in 1974 
    and further developed by Mermelstein in 1976 and is based on experiments of the human misconception of words.

    To extract a feature vector containing all information about the linguistic message, MFCC mimics some parts of the 
    human speech production and speech perception. MFCC mimics the logarithmic perception of loudness and pitch of human 
    auditory system and tries to eliminate speaker dependent characteristics by excluding the fundamental frequency and 
    their harmonics. To represent the dynamic nature of speech the MFCC also includes the change of the feature vector 
    over time as part of the feature vector.
    """
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    
    """
    Chroma features are an interesting and powerful representation for music audio in which the entire spectrum 
    is projected onto 12 bins representing the 12 distinct semitones (or chroma) of the musical octave. Since, 
    in music, notes exactly one octave apart are perceived as particularly similar, knowing the distribution 
    of chroma even without the absolute frequency (i.e. the original octave) can give useful musical information 
    about the audio -- and may even reveal perceived musical similarity that is not apparent in the original spectra.
    """
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    
    """
    In sound processing, the mel-frequency cepstrum (MFC) is a representation of the short-term power spectrum 
    of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency.
    
    mel-scaled spectrogram
    """
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    
    """
    Compute spectral contrast
    """
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    """
    Computes the tonal centroid features (tonnetz)
    """
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

#concatenate the features together as the input to training algorithm

def parse_audio_files(path_dir, ds_filename, fold_num):  
    #knowledge_based feature transformation. Transform each .wav file to a list of audio features
    features, labels, labels_name, file_name = np.empty((0,193)), np.empty(0, dtype = int), np.empty(0), np.empty(0)
    df = pd.read_csv(ds_filename, index_col = False)
    
    for row in df.itertuples():
        if(row[6]==fold_num):
            mfccs, chroma, mel, contrast,tonnetz = extract_feature_193(path_dir + str(row[6])+"/" + str(row[1]))
            
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, row[7])
            labels_name = np.append(labels_name, row[8])
            file_name = np.append(file_name, row[1])
 
    return features, labels, labels_name, file_name #np.array(features), np.array(labels, dtype = np.int)

start_time = time.time()
for i in range(1, 11):
    f, l, ln, fn = parse_audio_files(path_dir, mega_file, i)
    feature_pd_193 = pd.DataFrame(f) #np.array(mfcc_data).reshape(len(mfcc_data), 820))
    feature_pd_193["label"] = l
    feature_pd_193["labels_name"] = ln
    feature_pd_193["file_name"] = fn
    feature_pd_193.to_csv("./UrbanSound8K/audio/features193" + str(i) + ".csv", index = False)
    print ("fold"+str(i) +" finished transferring.")
end_time = time.time()
print("seconds:", end_time-start_time)



