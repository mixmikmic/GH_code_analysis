get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
import numpy as np
import time
from IPython.core.display import clear_output
from IPython.core.display import display

# create list of x values    
t=np.arange(-np.pi,np.pi,0.01)  

# frequency and amplitudes to build up our mystery wave
freqs = np.arange(1.0,20.,2.)
Cn = np.array([(3.46565342017-254.737095318j),
        (-1.16644367406+28.4709770497j),
        (0.685571832259-10.0922164294j),
        (-0.504127833607+5.240586586j),
        (0.376217427242-3.07367121897j),
        (-0.323123561657+2.1187342583j),
        (0.256920499162-1.44821668795j),
        (-0.238383315852+1.1331307276j),
        (0.193524671284-0.829332009977j),
        (-0.189110706279+0.69953977579j)])

# create plot
f, ax = plt.subplots(1,2,figsize=[13,3])

# set maximum number of iterations
i_max = 20

# initialise our function
wave = 0.0

# for each iteration...
for i in range(0,len(Cn)):
    # new component   
    wave_amp = Cn[i]
    wave_freq = freqs[i]
    component = wave_amp*np.cos(2.*np.pi*t*wave_freq)+1.0j*wave_amp*np.sin(2.*np.pi*t*wave_freq) 
    
    # add that component to our existing wave
    wave = wave+component
    
    # plot the new component on the left, the sum on the right
    ax[0].plot(t,component.real)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude (real)')

    ax[1].plot(t,wave.real)
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Amplitude (real)')
    ax[1].set_ylim(-320,320)
    
    # show the plot, then get ready for the next plot
    plt.draw()
    time.sleep(2.0)
    clear_output(wait=True)
    display(f)
    ax[1].cla()
    
plt.close()

plt.stem(freqs,Cn.real)
plt.xlabel('Frequency')
plt.ylabel('Amplitude (real)')

# create plot
f, ax = plt.subplots(1,2,figsize=[13,3])

# triangle function
ax[0].plot(t,wave.real)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude (real)')
ax[0].set_ylim(-320,320)
ax[0].set_title('Time domain')

# harmonics
ax[1].stem(freqs,Cn.real)
ax[1].set_xlabel('Frequency')
ax[1].set_ylabel('Amplitude (real)')
ax[1].set_title('Frequency domain')

import os, sys
import numpy
import matplotlib
import IPython

print 'OS:          ', os.name, sys.platform
print 'Python:      ', sys.version.split()[0]
print 'IPython:     ', IPython.__version__
print 'Numpy:       ', numpy.__version__
print 'matplotlib:  ', matplotlib.__version__



