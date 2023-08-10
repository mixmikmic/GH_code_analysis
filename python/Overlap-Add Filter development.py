get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sps
from scipy.io.wavfile import read as wavread

from IPython.display import Audio

fs, testsig = wavread('item_01_MA.wav')
Audio(data=testsig, rate=fs)

L_I = 1000

testfilt = sps.firwin(L_I, [300, 3300], nyq=fs/2, pass_zero=False)
plt.plot(testfilt)

get_ipython().magic('timeit sps.lfilter(testfilt, [1, ], testsig)')
result1 = sps.lfilter(testfilt, [1, ], testsig)
Audio(data=result1, rate=fs)

L_F = 2<<(L_I-1).bit_length()
L_S = L_F - L_I + 1
FDir = np.fft.rfft(testfilt, n=L_F)

L_sig = testsig.shape[0]
offsets = range(0, L_sig, L_S)

print('FFT size {}, filter segment size {}, overall signal length {}.'.format(L_F, L_S, L_sig))

tempresult = [np.fft.irfft(np.fft.rfft(testsig[n:n+L_S], n=L_F)*FDir) for n in offsets]
result2 = np.zeros(L_sig+L_F)
for i, n in enumerate(offsets):
    result2[n:n+L_F] += tempresult[i]
result2 = result2[:L_sig]

plt.plot(result1-result2)
print('SNR is {}.'.format(10*np.log10(np.sum(result1**2)/np.sum((result1-result2)**2))))

get_ipython().magic('run olafilt.py')

get_ipython().magic('timeit olafilt(testfilt, testsig)')
result3 = olafilt(testfilt, testsig)
print('SNR is {}.'.format(10*np.log10(np.sum(result1**2)/np.sum((result1-result3)**2))))

get_ipython().magic('timeit sps.fftconvolve(testfilt, testsig)')
result4 = sps.fftconvolve(testfilt, testsig)
result4 = result4[:result1.shape[0]]

plt.plot(result1-result4)
print('SNR is {}.'.format(10*np.log10(np.sum(result1**2)/np.sum((result1-result4)**2))))

zi = np.array([0])
splitpoint = L_sig//2
result5_part1, zi = olafilt(testfilt, testsig[:splitpoint], zi)
result5_part2, zi = olafilt(testfilt, testsig[splitpoint:], zi)
result5 = np.hstack((result5_part1, result5_part2))
print('Average RMSE full signal processing vs partial processing:', 
      np.sqrt(np.sum((result3-result5)**2))/L_sig)
plt.plot(result3-result5)

cfilt = sps.hilbert(testfilt)
result1c = sps.lfilter(cfilt, [1, ], testsig)
result3c = olafilt(cfilt, testsig)
print('SNR is {}.'.format(10*np.log10(np.sum(np.abs(result1**2))/np.sum(np.abs((result1-result3)**2)))))

