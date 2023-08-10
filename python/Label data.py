import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import IPython


directory = "batsounds"
filename = "bats8.m4a"
openfile = os.path.join(directory, filename)

print("Loading " + openfile)
soundarray, sr = librosa.load(openfile) # This operation can take a long time with large sound files

# We take 1 second per sample, which comes down to sampling-rate variables per sample. 
maxseconds = int(len(X)/sr)
for second in range(maxseconds-1):
    print(str(second) + " out of " + str(maxseconds))
    audiosample = np.array(soundarray[second*sr:(second+1)*sr])
    IPython.display.display(IPython.display.Audio(audiosample, rate=sr,autoplay=True))
    label = str(input("bat(1) or not (0)"))

    outputfilename = os.path.join("labeled", label, filename + str(second) + ".wav")
    print("Saving to: " + outputfilename)
    librosa.output.write_wav(outputfilename, audiosample, sr)

