import numpy as np
from scipy.stats import norm

# Fix random seed
np.random.seed(123456789)

numvals = 1000
# First, build the relevant linspace to grab 1000 points
times = np.linspace(0, 1, numvals)
# Then define the general sine wave used throughout
sin = np.sin(2 * np.pi * times)
# Define function for white noise
def gen_whitenoise(lower, upper, size):
    retval = np.random.uniform(lower, upper, size=size)
    return retval

sample_data = [np.column_stack([sin] * 100),
               np.column_stack([sin] * 100),
               np.column_stack([sin] * 100),
               np.column_stack([sin] * 100),
               np.column_stack([sin] * 100)]

for i in range(0, len(sample_data)):
    x = 0
    if i == 0 or i == 1:
        x = 3
    elif i == 2:
        x = 3
    elif i == 3:
        x = 5
    elif i == 4:
        x = 10
    elif i == 5:
        x = 15
    
    sim = sample_data[i]
    for j in range(0, sim.shape[1]):
        if j < x:
            sim[:, j] = sim[:, j] + gen_whitenoise(-2, 2, numvals).T 
        else:
            sim[:, j] = sim[:, j] + gen_whitenoise(-.5, .5, numvals).T 

ideal = [range(0, 3),
         range(0, 3),
         range(0, 5),
         range(0, 10),
         range(0, 15)]

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
from plotly.graph_objs import *
from plotly import tools
init_notebook_mode()

def plot_sim(data, title):
    # Setup plotly data
    datasets =[]
    for i in range(0, data.shape[1]):
        datasets.append(Scatter(
            x = times,
            y = data[:,i],
            name = 'Sample ' + str(i)
        ))

    data = [datasets[:]]

    # Setup layout

    layout = dict(title = title,
                  xaxis = dict(title = 'Time'),
                  yaxis = dict(title = 'Unit'),
                  )

    # Make figure object

    fig = dict(data=datasets, layout=layout)

    iplot(fig)

i = 0
for data in sample_data:
    plot_sim(data, "Data " + str(i))
    i += 1

def partial_disc(eeg_data, s_p, t, t_p, delta):
    T = eeg_data.shape[2] # Number of trials
    S = eeg_data.shape[3] # Number of subjects
    total_true = 0
    for s in range(S):
        if not (s == s_p):
            for t_pp in range(T):
                intra = delta(eeg_data[:, :, t, s], eeg_data[:, :, t_p, s])
                inter = delta(eeg_data[:, :, t, s], eeg_data[:, :, t_pp, s_p])
                print intra, inter
                total_true += int(intra < inter)
    print total_true
    return float(total_true) / ((T-1) * S)

def disc(eeg_data, delta):
    #T = eeg_data.shape[2] # Number of trials
    S = eeg_data.shape[2] # Number of subjects
    tot = 0
    for s in range(S):
        for t in range(T):
            for t_p in range(T):
                if not (t_p == t):
                    tot += partial_disc(eeg_data, s, t, t_p, delta)
    return float(tot) / ((S - 1) * T * S)

## Currently basic test delta function
def delta(arr1, arr2):
    if np.array_equal(arr1, arr2):
        return 0
    return 1

from scipy.stats import kurtosis

def kurt_baddetec(inEEG, threshold):
    electrodes = inEEG.shape[1]
    
    # Start by reshaping data (if necessary)
    if len(inEEG.shape) == 3:
        inEEG = np.reshape(inEEG, (inEEG.shape[0] * inEEG.shape[2], inEEG.shape[1]))
    elif len(inEEG.shape) != 1 and len(inEEG.shape) != 2:
        # fail case
        return -1
    
    # Then, initialize a probability vector of electrode length
    kurtvec = np.zeros(electrodes)
    
    # iterate through electrodes and get kurtoses
    for i in range(0, electrodes):
        # add kurtosis to the vector
        kurtvec[i] = kurtosis(inEEG[:, i])
        #print kurtvec[i]
        
    
    # normalize kurtvec
    # first calc mean
    avg = np.mean(kurtvec)
    # then std dev
    stddev = np.std(kurtvec)
    # then figure out which electrodes are bad
    badelec = []
    #print probvec
    for i in range(0, len(kurtvec)):
        #print i, avg, stddev, (avg - kurtvec[i]) / stddev
        if abs((avg - kurtvec[i]) / stddev) >= threshold:
            badelec.append(i)
            
    return badelec

def kurt_threshold(inEEG):
    thresh_range = np.linspace(1, 4, 31)
    thresh_dict = dict()
    for thresh in thresh_range:
        bad_elec = kurt_baddetec(inEEG, thresh)
        good_elec = list(set(range(0, inEEG.shape[1])) - set(bad_elec))
        discrim = disc(inEEG[:, good_elec], delta)
        thresh_dict[discrim] = thresh
        return thresh_dict[max(thresh_dict.keys())]
    

def qual_plot(data, title):
    # Setup plotly data
    datasets =[]
    for i in range(0, data.shape[1]):
        datasets.append(Scatter(
            x = times,
            y = data[:,i],
            name = 'Sample ' + str(i)
        ))

    data = [datasets[:]]

    # Setup layout

    layout = dict(title = title,
                  xaxis = dict(title = 'Time'),
                  yaxis = dict(title = 'Unit'),
                  )

    # Make figure object

    fig = dict(data=datasets, layout=layout)

    iplot(fig)

final_out = []
for data in sample_data:
    final_out.append(kurt_threshold(data))



