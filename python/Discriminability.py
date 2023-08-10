#%matplotlib inline
#import matplotlib.image as mpimage
#img = mpimage.imread("Selection_006.png")
#import matplotlib.pyplot as plt
#plt.axis("off")
#plt.imshow(img)

def rdf(eeg_data, delta, s):
    T = eeg_data.shape[2] # Number of trials
    S = eeg_data.shape[3] # Number of subjects
    total_true = 0
    for s_ in range(S):
        if not (s == s_):
            for t in range(T):
                for t_ in range(T):
                    if not (t == t_):
                        intra = delta(eeg_data[:, :, t, s], eeg_data[:, :, t_, s])
                        inter = delta(eeg_data[:, :, t, s], eeg_data[:, :, t, s_])
                        #print intra, inter
                        total_true += int(intra < inter)
    #print total_true
    return float(total_true) / ((S-1) * T * (T-1))

def disc(eeg_data, delta):
    T = eeg_data.shape[2] # Number of trials
    S = eeg_data.shape[3] # Number of subjects
    tot = 0
    for s in range(S):
        tot += rdf(eeg_data, delta, s)
    return float(tot) / (S)

import numpy as np
one = np.zeros([4, 4, 4])
two = np.ones([4, 4, 4])
eeg_data = np.concatenate([one[...,np.newaxis], two[...,np.newaxis]], axis=3)
#print eeg_data
#print eeg_data.shape
print eeg_data[:,:,:,0]

def distance(arr1, arr2):
    if np.array_equal(arr1, arr2):
        return 0
    return 1

print disc(eeg_data, distance)

def corrdistance(arr1, arr2):
    return abs(np.corrcoef(arr1, arr2)[0,1])

np.corrcoef(eeg_data[:,1,0,0], eeg_data[:,1,0,1])

#print disc(eeg_data, distance)



