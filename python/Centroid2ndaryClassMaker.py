import os 
import re

import numpy as np
import math
import tiffspect

import librosa
import librosa.display

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# First some utilities, and the function we want to use to classify our spectrograms 

def weightedCentroid(spect) :
    """
    param: spect - a magnitude spectrum
    Returns the spectral centroid averaged over frames, and weighted by the rms of each frame
    """
    cent = librosa.feature.spectral_centroid(S=spect)
    rms = librosa.feature.rmse(S=spect)
    avg = np.sum(np.multiply(cent, rms))/np.sum(rms)
    return avg

def log2mag(S) : 
    """ Get your log magnitude spectrum back to magnitude"""
    return np.power(10, np.divide(S,20.))

def spectFile2Centroid(fname) :
    """ Our spect files are in log magnitude, and in tiff format"""
    D1, _ = tiffspect.Tiff2LogSpect(fname)
    D2 = log2mag(D1)
    return weightedCentroid(D2)

# just testing code, demonstrating how to visualize som of the data
wc = spectFile2Centroid('esc50spect//205 - Chirping birds/5-242491-A.tif')
print ('wc is ' + str(wc))

D1, _ = tiffspect.Tiff2LogSpect('esc50spect//205 - Chirping birds/5-242491-A.tif')
D2 = log2mag(D1)
wcs = weightedCentroid(D2)
cent = librosa.feature.spectral_centroid(S=D2)
plt.plot(cent[0])

rms = librosa.feature.rmse(S=D2)
np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)
print(np.column_stack((np.transpose(np.round(cent,2)),np.transpose(np.round(rms,2)))))

# Next, some utilities for managing files
#----------------------------------------

def fullpathfilenames(directory): 
    '''Returns the full path to all files living in directory (the leaves in the directory tree)
    '''
    fnames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(directory)) for f in fn]
    return fnames

def esc50files(directory, regexString) :
    filenames = fullpathfilenames(directory)
    return [fname for fname in filenames if re.match(regexString, fname)]

def addClass2Filename(fname, cname, action="move") : 
    newname = re.sub('.tif', '_'+ str(cname) + '.tif', fname)
    if (action == "move") :
        os.rename(fname, newname)
    else :
        print(newname)
    
def filestats (filenames, func) :
    stats = [[fname, func(fname)] for fname in filenames]
    return stats

def createBalancedClassesWithFunc(topDirectory, regexString, func, numPerClass, action="move") :
    """
    Groups files in topDirectory matching regexString by the single number returned by func.
    Each group will have numPerClass files in it (the total number of files must be divisible by numPerClass)
    Renames them using their group index, gidx: origFilename.tif -> origFilename._gidx_.tif
    if action="move, files are renames. Otherwise, the new names are just printed to console.
    """
    wholelist=esc50files(topDirectory, regexString)
    print(wholelist)
    stats = filestats(wholelist, func)
    stats_ordered = sorted(stats, key=lambda a_entry: a_entry[1])
    classes=np.array(stats_ordered)[:,0].reshape(-1, numPerClass)
    for i in range(len(classes)) :
        for j in range(len(classes[i])) :
            addClass2Filename(classes[i,j],i, action)

    return stats, stats_ordered #returns stuff just for viewing 

#--------------------------------------------------------------------------------
#if you got yourself in trouble, and need to remove all the secondary classnames:
def removeAllSecondaryClassNames(directory) :
    """Remove ALL the 2ndary class names (of the form ._cname_) from ALL files in the directory restoring them to their original"""
    for fname in fullpathfilenames(directory) :
        m = re.match('.*?(\._.*?_)\.tif$', fname)  #grabs the string of all secondary classes if there is a seq of them
        if (m) :
            newname = re.sub(m.group(1), '', fname)
            print('Will move ' + fname + '\n to ' + newname)
            os.rename(fname, newname)
        else :
            print('do nothing with ' + fname)

#removeAllSecondaryClassNames('esc50spect')
#stats, stats_ordered  = createBalancedClassesWithFunc('esc50spect', '.*/([1-5]).*', spectFile2Centroid, 250, action="print")
#stats, stats_ordered  = createBalancedClassesWithFunc('/Volumes/Bothways/ZCODE/TENSORFLOW/dcn_soundclass/data50', '.*/(train|validate)/([1-5]).*', spectFile2Centroid, 250, action="print")

#change "print" to "move" to actually change the filenames:
stats, stats_ordered  = createBalancedClassesWithFunc('C:/Users/Huz/Documents/python_scripts/ESC50_multitask/ESC-50-cqt', '.*', spectFile2Centroid, 250, action="move")

# Analysis of the results of your classification
#------------------------------------------------

plt.plot(np.array(stats)[:,1])         # centroids of files in order read in (plotted in blue below)
plt.plot(np.array(stats_ordered)[:,1]) #centroids from min to max (plotted in orange below)

#Just for our own info, and paper writeups:
breaks=[baz[i][1] for i in range(0, 2000, 250)] # [0, 250, 500, 750, 1000, 1250, 1500, 1750]
breaks





