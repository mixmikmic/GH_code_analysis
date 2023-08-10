import tensorflow as tf

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import wave, os, glob
get_ipython().run_line_magic('matplotlib', 'inline')

import librosa
import librosa.display
import librosa.feature
from sklearn.mixture import GaussianMixture as GMM

def preprocess_data(subject_num, block_size, threshold):
    """
    Given a subject number:
        - retrieve data from directory 
        - remove low power components
        - divide the continuous sound vector into uniform blocks
        - compute MFCC for each block
    
    Returns:
        - S, tensor of shape (num_samp, n_mels)
    
    """ 
    
    # Create empty vector to store data
    subject = []
    
    # Define path
    path = "/Users/willi/EE4563/speech_samples/subject_"+subject_num+"/left-headed"
    
    # Retrieve data from directory, concatenate all wav file data in the target directory
    for filename in glob.glob(os.path.join(path, '*.wav')):
        y, sr = librosa.load(filename)    
        subject = np.append(subject,y)

    # Remove low power components
    low_ind = (abs(subject) <= threshold)
    subject[low_ind] = 0
    
    subject_active = subject[subject != 0]
    
    num_samples = int(np.floor(len(subject_active)/block_size))
    
    # Reshape data into blocks of samples
    subject_seg = subject_active[0:(num_samples*block_size)]
    subject_seg = np.reshape(subject_seg,(num_samples,-1))
    
    block_time = (librosa.feature.melspectrogram(y=subject_seg[0,:], sr=sr, n_mels=128, fmax=8000)).shape[1]
    S = np.empty((0,128,block_time))
    
    print("SUBJECT ",subject_num)
    print("Generating MFCC for %d total samples." %num_samples)
    
    # Generate MFCC over each sample
    for n in range(0,num_samples):

        S = np.append(S,librosa.feature.melspectrogram(y=subject_seg[n,:], sr=sr, n_mels=128, fmax=8000)[None,:,:],axis = 0)
        if n % 500 == 0:
            print(n," samples done")
            
    print("* COMPLETE *")
    
    # Plot the spectrogram of the first block/sample
    librosa.display.specshow(librosa.logamplitude(S[0,:,:],ref_power=np.max),
                             y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram\nSubject %s Block 0' %subject_num)
    plt.tight_layout()
        
    return S[:,:,0]

def check_user(speech_samp,model1, model2, model3, model4):
    """
    Given a speech sample fragment and four possible speaker models:
        - determine how likely the sample came from each of the models
        - print the predicted speaker the speech sample belongs to
            
    Returns: NONE
    
    """ 
    
    # Score the speech sample using each given model
    scores = np.array(([model1.score(speech_samp),model2.score(speech_samp),
                        model3.score(speech_samp),model4.score(speech_samp)]))
    
    # The largest value in the score vector corresponds to the most likely speaker
    predicted_speaker = np.argmax(scores)+1
    print("The predicted speaker is subject %d" %predicted_speaker)
    

# Define basic parameters

block_size = 256
threshold = 0.05
nts = 1000
ncomp = 8

# Preprocess data for subject 1

subject = '1'
S1 = preprocess_data(subject,block_size,threshold)

# Preprocess data for subject 2

subject = '2'
S2 = preprocess_data(subject,block_size,threshold)

# Preprocess data for subject 3

subject = '3'
S3 = preprocess_data(subject,block_size,threshold)

# Preprocess data for subject 4

subject = '4'
S4 = preprocess_data(subject,block_size,threshold)

# Split training/test data and fit Gaussian Mixture Model for subject 1

ntr1 = S1.shape[0]-nts

Xtr1 = S1[:ntr1,:]
Xts1 = S1[ntr1:,:]
gmm1 = GMM(n_components=ncomp)

# Fit model to training data
gmm1.fit(Xtr1)

# Split training/test data and fit Gaussian Mixture Model for subject 2

ntr2 = S2.shape[0]-nts

Xtr2 = S2[:ntr2,:]
Xts2 = S2[ntr2:,:]
gmm2 = GMM(n_components=ncomp)

# Fit model to training data
gmm2.fit(Xtr2)

# Split training/test data and fit Gaussian Mixture Model for subject 3

ntr3 = S3.shape[0]-nts

Xtr3 = S3[:ntr3,:]
Xts3 = S3[ntr3:,:]
gmm3 = GMM(n_components=ncomp)

# Fit model to training data
gmm3.fit(Xtr3)

# Split training/test data and fit Gaussian Mixture Model for subject 4

ntr4 = S4.shape[0]-nts

Xtr4 = S4[:ntr4,:]
Xts4 = S4[ntr4:,:]
gmm4 = GMM(n_components=ncomp)

# Fit model to training data
gmm4.fit(Xtr4)

# Compute a matrix on the scores 
# Ideally, the values on the diagonals should be high, while the off-diagonal values low

scores = np.array(([gmm1.score(Xts1),gmm2.score(Xts1),gmm3.score(Xts1),gmm4.score(Xts1)],
                   [gmm1.score(Xts2),gmm2.score(Xts2),gmm3.score(Xts2),gmm4.score(Xts2)],
                   [gmm1.score(Xts3),gmm2.score(Xts3),gmm3.score(Xts3),gmm4.score(Xts3)],
                   [gmm1.score(Xts4),gmm2.score(Xts4),gmm3.score(Xts4),gmm4.score(Xts4)]))

print(scores)

# Use the test data from each speaker to test the models

check_user(Xts4,gmm1,gmm2,gmm3,gmm4) # Speaker 4
check_user(Xts1,gmm1,gmm2,gmm3,gmm4) # Speaker 1
check_user(Xts3,gmm1,gmm2,gmm3,gmm4) # Speaker 3
check_user(Xts2,gmm1,gmm2,gmm3,gmm4) # Speaker 2

