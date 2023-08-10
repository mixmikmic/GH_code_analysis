## You need to install the PyWavelets package before you use it.
## Only need to do this once on the server.
# I commented this out, so it doesn't run unless you need it -- take the # signs out below

#%%bash
#pip install  PyWavelets --user

## We load the tools we need, including the wavelet package here
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import pywt

x = [1, 2, 3, 4, 5, 6, 7, 8]
cA, cD = pywt.dwt(x, 'db1')

# the sums (remove the root 2 factor)
np.sqrt(2)*cA

# the differences (remove the root 2 factor)
np.sqrt(2)*cD

plt.subplot(1, 3, 1)
plt.plot(x)
plt.subplot(1, 3, 2)
plt.plot(cA)
plt.subplot(1, 3, 3)
plt.plot(cD)

# Lets to a more interesting signal, a simple sine wave
n = 1024
t = np.linspace(0,1,n)
x = np.sin(2*np.pi*t)
cA, cD = pywt.dwt(x,'db1')
plt.subplot(1, 3, 1)
plt.plot(x)
plt.title("original")
plt.subplot(1, 3, 2)
plt.plot(cA)
plt.title("approximation")
plt.subplot(1, 3, 3)
plt.plot(cD)
plt.title("detail")

# Lets to a more interesting signal, a ramping sine wave
n = 1024
t = np.linspace(0,1,n)
x = np.sin(2*np.pi*t*t)
cA, cD = pywt.dwt(x,'db1')
plt.subplot(1,3, 1)
plt.plot(x)
plt.title("original")
plt.subplot(1,3, 2)
plt.plot(cA)
plt.title("approximation")
plt.subplot(1,3, 3)
plt.plot(cD)
plt.title("detail")

# Lets to a more interesting signal, a ramping sine wave
n = 1024
t = np.linspace(0,1,n)
x = np.sin(2*np.pi*t*t)
cA, cD2, cD1 = pywt.wavedec(x,'db1',level=2)
fig = plt.figure(figsize=(10,10))
plt.subplot(4,1, 1)
plt.plot(x)
plt.title("original")
plt.subplot(4,1, 2)
plt.plot(cA)
plt.title("approximation")
plt.subplot(4,1, 3)
plt.plot(cD2)
plt.title("detail 2")
plt.subplot(4,1, 4)
plt.plot(cD1)
plt.title("detail 1")
plt.tight_layout()

# Lets to a more interesting signal, a ramping sine wave
n = 1024
t = np.linspace(0,4,n)
x = np.sin(2*np.pi*t*t)
cA, cD3, cD2, cD1 = pywt.wavedec(x,'db1',level=3)
fig = plt.figure(figsize=(10,10))
plt.subplot(5,1, 1)
plt.plot(x)
plt.ylabel("original")
plt.subplot(5,1, 2)
plt.plot(cA)
plt.ylabel("approximation")
plt.subplot(5,1, 3)
plt.plot(cD3)
plt.ylabel("detail 3")
plt.subplot(5,1, 4)
plt.plot(cD2)
plt.ylabel("detail 2")
plt.subplot(5,1, 5)
plt.plot(cD1)
plt.ylabel("detail 1")
plt.tight_layout()

# Lets to a more interesting signal, a ramping sine wave
n = 1024
t = np.linspace(0,4,n)
x = np.sin(2*np.pi*t*t)
cA, cD3, cD2, cD1 = pywt.wavedec(x,'db3',level=3)
fig = plt.figure(figsize=(10,10))
plt.subplot(5,1, 1)
plt.plot(x)
plt.ylabel("original")
plt.subplot(5,1, 2)
plt.plot(cA)
plt.ylabel("approximation")
plt.subplot(5,1, 3)
plt.plot(cD3)
plt.ylabel("detail 3")
plt.subplot(5,1, 4)
plt.plot(cD2)
plt.ylabel("detail 2")
plt.subplot(5,1, 5)
plt.plot(cD1)
plt.ylabel("detail 1")
plt.tight_layout()

# we should note how to use a list of arrays from the wavelet transform. Interesting syntax to use.
n = 1024
t = np.linspace(0,4,n)
x = np.sin(2*np.pi*t*t)
coeffs = pywt.wavedec(x,'db3',level=3)
fig = plt.figure(figsize=(10,10))
plt.subplot(5,1, 1)
plt.plot(x)
plt.ylabel("original")
plt.subplot(5,1, 2)
plt.plot(coeffs[0])
plt.ylabel("approximation")
plt.subplot(5,1, 3)
plt.plot(coeffs[1])
plt.ylabel("detail 3")
plt.subplot(5,1, 4)
plt.plot(coeffs[2])
plt.ylabel("detail 2")
plt.subplot(5,1, 5)
plt.plot(coeffs[3])
plt.ylabel("detail 1")
plt.tight_layout()

# Let's do a reconstruction, dropping the detail 1 level.
n = 1024
t = np.linspace(0,4,n)
x = np.sin(2*np.pi*t*t)
coeffs = pywt.wavedec(x,'db3',level=3)
coeffs[3] = 0*coeffs[3]   # we zero out the detail 1 level
y = pywt.waverec(coeffs,'db3')
fig = plt.figure(figsize=(10,10))
plt.subplot(3,1, 1)
plt.plot(x)
plt.ylabel("original")
plt.subplot(3,1, 2)
plt.plot(y)
plt.ylabel("reconstruction")
plt.subplot(3,1, 3)
plt.plot(y-x)
plt.ylabel("difference")
plt.tight_layout()

# Let's reconstruct some wavelets
n = 1024
t = np.linspace(0,4,n)
x = 0*t  # we start with nothing
coeffs = pywt.wavedec(x,'db5',level=6)
coeffs[1][10] = 1   # stick in one non-zero value
y = pywt.waverec(coeffs,'db5')
fig = plt.figure(figsize=(10,5))
plt.plot(y)
plt.ylabel("reconstruction")
plt.tight_layout()

lp = np.zeros(128)
hp = np.zeros(128)
lp[0] = 1
lp[1] = 1
hp[0] = -1
hp[1] = 1
lpfft = np.abs(np.fft.fft(lp))
hpfft = np.abs(np.fft.fft(hp))


fig = plt.figure(figsize=(10,5))
plt.subplot(1,2, 1)
plt.plot(lpfft[0:64])
plt.title("low pass")
plt.subplot(1,2, 2)
plt.plot(hpfft[0:64])
plt.title("high pass")
plt.tight_layout()

lp = np.zeros(128)
hp = np.zeros(128)
w = pywt.Wavelet('db2')  # change the db1 to sb2m db3 etc
lp[0:len(w.dec_lo)] = w.dec_lo
hp[0:len(w.dec_hi)] = w.dec_hi
lpfft = np.abs(np.fft.fft(lp))
hpfft = np.abs(np.fft.fft(hp))


fig = plt.figure(figsize=(10,5))
plt.subplot(1,2, 1)
plt.plot(lpfft[0:64])
plt.title("low pass")
plt.subplot(1,2, 2)
plt.plot(hpfft[0:64])
plt.title("high pass")
plt.tight_layout()

t = np.linspace(0, 3, 300, endpoint=False)
sig = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7*(t-2.0)**2)*np.exp(1j*2*np.pi*2*(t-2.0)))
widths = np.arange(1, 31)
cwtmatr, freqs = pywt.cwt(sig, widths, 'mexh')
plt.imshow(cwtmatr, extent=[0, 3, 1, 31], cmap='PRGn', aspect='auto',
  vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())

Pxx, freqs, bins, im = plt.specgram(sig, NFFT=64, Fs=100, noverlap=5)

# We use a meshgrid to compute matrices of x and y values, in the square [-1,1]x[-1,1]
# We use Gaussians to build circles and lines
x = np.linspace(-1,1,256)
y = np.linspace(-1,1,256)
X,Y = np.meshgrid(x,y)  # these are matrices!
radius = .05
myimage = np.exp(-np.abs(X*X + Y*Y - .051)/.01) + np.exp(-np.abs(X-Y)/.01) 
#myimage = myimage -np.mean(np.mean(myimage))
myfft = np.abs(np.fft.fftshift(np.fft.fft2(myimage)))
fig = plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(myimage)
plt.subplot(1,2,2)
plt.imshow(myfft)

# We use a meshgrid to compute matrices of x and y values, in the square [-1,1]x[-1,1]
# We use Gaussians to build circles and lines
x = np.linspace(-1,1,256)
y = np.linspace(-1,1,256)
X,Y = np.meshgrid(x,y)  # these are matrices!
radius = .05
myimage = np.exp(-np.abs(X*X + Y*Y - .051)/.01) + np.exp(-np.abs(X-Y)/.01) 
coeffs = pywt.wavedec2(myimage, 'db1',level=1)
mywt2, coeff_slices = pywt.coeffs_to_array(coeffs)
fig = plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(myimage)
plt.subplot(1,2,2)
plt.imshow(mywt2)



