#helpful settings
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
get_ipython().magic('pylab inline')

#Loading all of the helpful modules
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import scipy.io
import scipy.signal

#loading ECoG data
filename = 'emodat.mat'
filename = os.path.join('./', filename)
data = sp.io.loadmat(filename)
srate = data['srate'] # sampling rate
srate = srate[0][0];
data = data['data'] # time series
data = data[0, :]
print "There are " + str(len(data)) + "samples in this data"
print "The sampling rate is " + str(srate)

event_inds = [1034, 2133, 4681, 9960, 12574] #Time indicies when a stimulus is presented to the subject
this_event = event_inds[0]
timewin = [500, 1000] #500 samples before and 1000 samples after. Remember this: The units here are samples, not milliseconds. 
#We sampled at 1017.253 samples per second so we'll looking at about a second and a half of data.


my_ERP = data[this_event-timewin[0]:this_event + timewin[1]]

plot(my_ERP,label = 'ERP')
plot([500,500], [-150, 100],label = 'event onset')
legend()

print 'ECoG data ~500 ms before and ~1000 ms after an event'

#Let's look at ERP's of 5 events next to eachother

for event in range(len(event_inds)):
    this_ERP = data[event_inds[event]-timewin[0]:event_inds[event] + timewin[1]]
    plot(this_ERP,label = 'ERP ' +str(event+1))

plot([500,500], [-200, 200],label = 'event onset')    
legend()
print my_ERP.shape

chan_baselines = zeros(len(event_inds)); #We're going to get a baseline ERP data before we analyze any of the data.
ERP_matrix = zeros((len(event_inds),timewin[0]+timewin[1])); #We're making a matrix that will hold all the trials. We will get the mean from this.

for event in range(len(event_inds)):
    chan_baselines[event] = mean(data[event_inds[event]-100:event_inds[event] + 0]) 
    ERP_baseline_removed = data[event_inds[event]-timewin[0]:event_inds[event] + timewin[1]] - chan_baselines[event] #subtracting baseline from each trial.    
    ERP_matrix[event,:] = ERP_baseline_removed;
    
mean_ERP = mean(ERP_matrix,0) #taking the mean voltage across trials.
print chan_baselines #see why it's important to remove baseline?

#plotting
plot(mean_ERP, label = 'ERP');
plot([500,500], [-60, 60],label = 'event onset');
legend()





