get_ipython().magic("config InlineBackend.figure_format = 'retina'")
get_ipython().magic('pylab inline')
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import scipy.io
import scipy.signal
from scipy.signal import butter, filtfilt, hilbert

# Importing Buzsáki's data from a local source
# This data can be downloaded from www.github.com/voytekresearch/Tutorials
fpath = ('./')
filename = 'lfp_example.mat'
filename = os.path.join(fpath+filename)

data = sp.io.loadmat(filename)

lfpsrate = int(data['srate']); #sampling rate for LFP
spikesrate = int(data['spksrate']); #sampling rate for spikes (different than LFP sampling rate)


lfp = data['data']; # Local field potential recorded by 31 shanks
spikes = data['T']; # Times of spikes

lfpspiketimes = np.ceil(spikes / (spikesrate/lfpsrate)); #Converting spike timing to srate of LFP.
lfpspiketimes = lfpspiketimes.astype(int);

lfp = mean(lfp,0); #Average LFP of all shanks


plot(lfp[0:lfpsrate]) #Average local field
ylabel('Amplitude')
xlabel('time (ms)')
print 'One Second of Local field potential in CA2'

#helpful filtering functions
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs #nyquist frequency
    low = float(lowcut) / nyq
    high = float(highcut) / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(mydata, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, mydata)
    return y


filtdat = butter_bandpass_filter(lfp,4,8,lfpsrate); #Filtering the LFP data

figure(figsize = (15,6))
plot(lfp[0:lfpsrate*2],label = 'Raw LFP')


plot(filtdat[0:lfpsrate*2],label = 'Filtered LFP')

for spike in lfpspiketimes[0:85]:
    plot([spike, spike],[-2500,2500])
legend(loc='center left', bbox_to_anchor=(1, 0.5))

spike_triggered_average_mat = np.zeros((100,1250)) # What does the lfp look like when spikes happen?
spike_triggered_average_mat_filt = np.zeros((100,1250)) # What does the filtered data look like when spikes happen?

c=0;
for spike in lfpspiketimes[100:200,0]:
    spike_triggered_average_mat[c,:] = lfp[spike-625:spike+625:];
    spike_triggered_average_mat_filt[c,:] = filtdat[spike-625:spike+625:];
    c +=1;

spike_triggered_average = mean(spike_triggered_average_mat,0); #Average signal when spikes happen
spike_triggered_average_filt = mean(spike_triggered_average_mat_filt,0);#Average filtered signal when spikes happen

figure(figsize = (15,6))
plot(spike_triggered_average,label = 'Raw signal')
plot(spike_triggered_average_filt,label = 'Filtered signal')
plot([626, 626],[-250,250],label = 'Spike')
legend()
    

#calculating the phase of filtered data. This might take a minute to run
hilbert_filt = sp.signal.hilbert(filtdat); 
angle_dat = angle(hilbert_filt); 

phases_at_spikes = np.zeros((10000));
amp_at_spikes = np.zeros((10000));

c=0;
for spike in lfpspiketimes[0:10000]:
    spike = int(spike);
    phases_at_spikes[c]= angle_dat[spike];
    amp_at_spikes[c]= lfp[spike];
    c +=1;
sfc =np.histogram(phases_at_spikes,20);
sfc_amp =np.histogram(amp_at_spikes);

bins= (sfc[1]+pi); 
bins = bins[:len(bins)-1];

fig = figure(figsize=(8,8))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
bars = ax.bar(bins, sfc[0], width=bins[1]-bins[0], bottom=0.0)
plt.title('Spike Field Coherence');

fig = figure(figsize=(8,8))
hist(amp_at_spikes);
plt.xlabel('Amplitude');

import random as rd

phase_locking_value= abs(sum(exp(1j*phases_at_spikes))/len(phases_at_spikes)); #fancy math
print 'Phase Locking Value: '+ str(phase_locking_value)

bootstrap_vals = zeros((1000));

#bootstraping significance value
for x in xrange(1000):
    random_phases = [ angle_dat[i] for i in rd.sample(xrange(len(angle_dat)), len(phases_at_spikes)) ]
    bootstrap_vals[x] = np.abs(np.sum(np.exp(np.dot(1j,random_phases)))/len(random_phases));
  
fig = figure(figsize=(8,8))
hist(bootstrap_vals)
plot([phase_locking_value, phase_locking_value],[0, 200])
xlabel('SFC score');



