import wave, struct, math # To calculate the WAV file content
import numpy as np # To handle matrices
from PIL import Image # To open the input image and convert it to grayscale

import scipy                     # To plot the spectrogram
import matplotlib.pyplot as plt  # To plot the spectrogram
import scipy.io.wavfile          # To plot the spectrogram

import scipy.ndimage # To resample using nearest neighbour
import IPython.display  # Jupyter notebook ...

def plotSpectrogram(file="sound.wav"):
    sample_rate, X = scipy.io.wavfile.read(file)
    plt.specgram(X, Fs=sample_rate, xextent=(0,60))
    print("File: ", file)
    print("Sample rate (Hz): ",sample_rate)

def plotMat(mat):
    mat = np.flip(mat,0)
    X, Y = np.meshgrid(range(mat.shape[0]), range(mat.shape[1]))
    Z = mat[X,Y]

    plt.pcolormesh(Y,X,Z)
    plt.show()

x = np.arange(9).reshape(3,3)
print("Original array")
x

print("After resampling by factor of 2 along both axis, using nearest neighbour")
scipy.ndimage.zoom(x, 2, order=0)

'''
    Loads a picture, converts it to greyscale, then to numpy array, normalise it so that the max value is 1 
    the min is 0, increase the contrast a bit, remove every pixel which intensity is lower that 0.5, 
    then resize the picture using nearest neighbour resampling and outputs the numpy matrix.
    
    FYI: imgArr[0,0] is the top left corner of the image, cheers matrix indexing
    
    Returns: the resized image as a high contrast, normalised between 0 and 1, numpy matrix
'''
def loadPicture(size, file, verbose=1):
    img = Image.open(file)
    img = img.convert("L")
    #img = img.resize(size) # DO NOT DO THAT OR THE PC WILL CRASH
    
    imgArr = np.array(img)
    if verbose:
        print("Image original size: ", imgArr.shape)
        
    # Increase the contrast of the image
    imgArr = imgArr/np.max(imgArr)
    imgArr = 1/(imgArr+10**15.2)
    
    # Scale between 0 and 1
    imgArr -= np.min(imgArr)
    imgArr = imgArr/np.max(imgArr)
    
    # Remove low pixel values
    removeLowValues = np.vectorize(lambda x: x if x > 0.5 else 0, otypes=[np.float])
    imgArr = removeLowValues(imgArr)
    
    if size[0] == 0:
        size = imgArr.shape[0], size[1]
    if size[1] == 0:
        size = size[0], imgArr.shape[1]
    resamplingFactor = size[0]/imgArr.shape[0], size[1]/imgArr.shape[1]
    if resamplingFactor[0] == 0:
        resamplingFactor = 1, resamplingFactor[1]
    if resamplingFactor[1] == 0:
        resamplingFactor = resamplingFactor[0], 1
    
    # Order : 0=nearestNeighbour, 1:bilinear, 2:cubic etc...
    imgArr = scipy.ndimage.zoom(imgArr, resamplingFactor, order=0)
    
    if verbose:
        print("Resampling factor", resamplingFactor)
        print("Image resized :", imgArr.shape)
        print("Max intensity: ", np.max(imgArr))
        print("Min intensity: ", np.min(imgArr))
        plotMat(imgArr)
    return imgArr

IPython.display.Image("/home/sam1902/Pictures/WandererAboveTheSeaOfFogResized.jpg")

imgMat = loadPicture(size=(2901,2300), file="/home/sam1902/Pictures/WandererAboveTheSeaOfFog.jpg")

def genSoundFromImage(file, output="sound.wav", duration=5.0, sampleRate=44100.0):
    wavef = wave.open(output,'w')
    wavef.setnchannels(1) # mono
    wavef.setsampwidth(2) 
    wavef.setframerate(sampleRate)
    
    max_frame = int(duration * sampleRate)
    max_freq = 22000 # Hz
    max_intensity = 32767
    
    stepSize = 400 # Hz
    steppingSpectrum = int(max_freq/stepSize)
    
    imgMat = loadPicture((steppingSpectrum, max_frame), file, verbose=0)
    imgMat *= max_intensity
    print("Input: ", file)
    print("Duration (in seconds): ", duration)
    print("Sample rate: ", sampleRate)
    print("Computing each soundframe sum value..")
    for frame in range(max_frame):
        if frame % 60 == 0: # Only print once in a while
            IPython.display.clear_output(wait=True)
            print("Progress: ==> {:.2%}".format(frame/max_frame), end="\r")
        signalValue, count = 0, 0
        for step in range(steppingSpectrum):
            intensity = imgMat[step, frame]
            if intensity == 0:
                continue
            # nextFreq is less than currentFreq
            currentFreq = max_freq - step * stepSize
            nextFreq = max_freq - (step+1) * stepSize
            if nextFreq < 0: # If we're at the end of the spectrum
                nextFreq = 0
            for freq in range(nextFreq, currentFreq, 1000): # substep of 1000 Hz is good
                signalValue += intensity*math.cos(freq * 2 * math.pi * float(frame) / float(sampleRate))
                count += 1
        if count == 0: count = 1
        signalValue /= count
        
        data = struct.pack('<h', int(signalValue))
        wavef.writeframesraw( data )
        
    wavef.writeframes(''.encode())
    wavef.close()
    print("\nProgress: ==> 100%")
    print("Output: ", output)

genSoundFromImage(file="/home/sam1902/Pictures/WandererAboveTheSeaOfFog.jpg")

plotSpectrogram()

