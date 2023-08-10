get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
import numpy as np
import time 
from IPython.core.display import clear_output
from IPython.core.display import display

x=np.arange(-np.pi,np.pi,0.01) 

# create plot
f, ax = plt.subplots(1,2,figsize=[13,3])

# set maximum number of iterations
i_max = 20

# initialise values
sqwave = 0.0
b = 1.0

# for each iteration...
for i in range(1,i_max):
    # new component   
    component = (1/b)*np.sin(b*x)
    # add that component to our existing wave
    sqwave = sqwave+component
    
    # plot the new component on the left, the sum on the right
    ax[0].plot(x,component)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude (real)')
    
    ax[1].plot(x,sqwave)
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Amplitude (real)')
    
    # iterate
    b=b+2

    # show the plot, then get ready for the next plot
    plt.draw()
    time.sleep(2.0)
    clear_output(wait=True)
    display(f)
    ax[1].cla()
    
plt.close()

import os, sys
import numpy
import matplotlib
import IPython

print 'OS:          ', os.name, sys.platform
print 'Python:      ', sys.version.split()[0]
print 'IPython:     ', IPython.__version__
print 'Numpy:       ', numpy.__version__
print 'matplotlib:  ', matplotlib.__version__



