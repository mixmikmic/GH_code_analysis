get_ipython().magic('pylab inline')

import numpy as np
from scipy import signal

x = array([ 0.07173809,  0.08927202,  0.10931984,  0.12888261,  0.14506918,
        0.15528506,  0.15739706,  0.14987065,  0.13187723,  0.10336782,
        0.06510811,  0.01866841, -0.03363801, -0.08887335, -0.14365156,
       -0.1943677 , -0.23746812, -0.26974155, -0.28860473, -0.29235225,
       -0.28034085, -0.25308265, -0.21223148, -0.16045852, -0.10122738,
       -0.03849198,  0.02364947,  0.08130607,  0.13111293,  0.17048475,
        0.19778127,  0.21237415,  0.21461631,  0.20572562,  0.18760184,
        0.16260017,  0.13328571,  0.10219209,  0.07160499,  0.04338767,
        0.01886157, -0.0012495 , -0.01680865, -0.02819138, -0.03615636,
       -0.04167683, -0.04575288, -0.04922816, -0.05263522, -0.05609121,
       -0.05925984, -0.06138745, -0.06141052, -0.05812188, -0.050373  ,
       -0.03728382, -0.01842859,  0.00603124,  0.03528878,  0.06791968,
        0.1019943 ,  0.13526002,  0.16536863,  0.19011767,  0.20767296,
        0.21674355,  0.21668832,  0.20754462,  0.18998106,  0.16518718,
        0.13472059,  0.10033637,  0.06382262,  0.02686264, -0.00906342,
       -0.04273232, -0.07319031, -0.09973401, -0.12187301, -0.13928831,
       -0.15179841, -0.15934055, -0.16196921, -0.15986845, -0.15337059,
       -0.14297148, -0.12933269, -0.11326353, -0.09567982, -0.07754179,
       -0.0597784 , -0.04320901, -0.02847528, -0.01599463, -0.00594336,
        0.0017284 ,  0.00725373,  0.01098699,  0.01333559,  0.01469781])

#x = np.random.rand(100)
#x = signal.detrend(x)
#Now create a lowpass Butterworth filter with a cutoff of 0.125 times
#the Nyquist rate, or 125 Hz, and apply it to x with filtfilt.  The
#result should be approximately xlow, with no phase shift.
fb, fa = signal.butter(8, 0.125)
x = signal.filtfilt(fb, fa, x, padlen=10)
#print np.abs(y - xlow).max()
plot(x)

y = np.concatenate((np.zeros(50), x,np.zeros(50)))

plot(y)

xycor = np.correlate(x, y, mode='full')
plot(xycor)

xycor.argmax()

xycor.size

size(x)

print ((arange(size(x))-size(x))+1)

fig = figure(figsize=(15,20))
ax = fig.add_subplot(521)
ax.plot(y)
ax.plot(((arange(size(x))-size(x))+1), x)
ax.set_xlim(-100, 200)
ax.set_title('Moving and Calculating Correlation: Move(1)')
ax = fig.add_subplot(522)
#ax.plot(range(799), np.zeros(799))
ax.plot(0, x[-1]*y[0], 'or')
ax.plot(xycor, 'k--')
ax.set_xlim(-10, 299)
ax.set_ylim(-1.5, 2.)
ax.set_title('Cross-correlation y*x until Move(1)')

ax = fig.add_subplot(523)
ax.plot(y)
ax.plot(((arange(size(x))-size(x))+1)+30, x)
ax.set_xlim(-100, 200)
ax.set_title('Moving and Calculating Correlation: Move(30)')
ax = fig.add_subplot(524)
ax.plot(xycor[:30], 'or')
ax.plot(xycor, 'k--')
ax.set_xlim(-10, 299)
ax.set_ylim(-1.5, 2.)
ax.set_title('Cross-correlation y*x until Move(30)')

ax = fig.add_subplot(525)
ax.plot(y)
ax.plot(((arange(size(x))-size(x))+1)+95, x)
ax.set_xlim(-100, 200)
ax.set_title('Moving and Calculating Correlation: Move(95)')
ax = fig.add_subplot(526)
ax.plot(xycor[:95], 'or')
ax.plot(xycor, 'k--')
ax.set_xlim(-10, 299)
ax.set_ylim(-1.5, 2.)
ax.set_title('Cross-correlation y*x until Move(95)')

ax = fig.add_subplot(527)
ax.plot(y)
ax.plot(((arange(size(x))-size(x))+1)+149, x)
ax.set_xlim(-100, 200)
ax.set_title('Moving and Calculating Correlation: Move(149)')
ax = fig.add_subplot(528)
ax.plot(xycor[:149], 'or')
ax.plot(xycor, 'k--')
ax.set_xlim(-10, 299)
ax.set_ylim(-1.5, 2.)
ax.set_title('Cross-correlation y*x until Move(149)')

print (xycor.argmax()-size(x))+1

size(x_pad)
size(y)

npad = size(y) - size(x)
x_pad = np.concatenate((x, np.zeros(npad)))

a = xcorr(x_pad, y,  maxlags=None, normed=True)

print a[1].argmax()
print a[0][a[1].argmax()]



