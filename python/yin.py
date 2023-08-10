from yin import yin,yin_signal
import librosa
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from IPython.display import display,Audio

piano = 'piano-c4.wav'
flute = 'flute-a4.wav'

wave,sr = librosa.load(flute,sr=None)
wave/= np.mean(np.abs(wave))
frame = wave[0:2048]
frame /= np.max(np.abs(frame))
plt.plot(frame)
plt.xlabel('sample')
plt.ylabel('amplitude')
plt.show()

print("Flute")
display(Audio(wave,rate=sr))


f0_1,autocorr = yin(frame,step=1)
f0_2,diff_func = yin(frame,step=2)
f0_3,norm_diff_func = yin(frame,step=3)
f0_4,_ = yin(frame,step=4)

plt.plot(autocorr)
plt.xlabel('Samples')
plt.ylabel('Autocorrelation')
plt.show()

plt.plot(diff_func)
plt.xlabel('Samples')
plt.ylabel('Difference Function')
plt.show()

plt.plot(norm_diff_func)
plt.xlabel('Samples')
plt.ylabel('Normalized Difference Function')
plt.show()

print("""Step 1: %d \nStep 2: %d\nStep 3: %d\nStep 4: %d""" 
      %(f0_1,f0_2,f0_3,f0_4))

f0s = yin_signal(wave,hop=128)


plt.plot(f0s,'ro')
ply.show()



