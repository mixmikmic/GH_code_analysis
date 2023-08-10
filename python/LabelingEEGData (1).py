import h5py
import numpy as np
import pandas as pd
import plotly
plotly.offline.init_notebook_mode()
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import (download_plotlyjs,
                            init_notebook_mode,
                            plot,
                            iplot)
def plot_heat(M, title):
    data = [
        go.Heatmap(
            z = M
        )
    ]
    layout = go.Layout(
        title = title
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)

data = h5py.File('test1.mat', 'r')
eeg_data1 = np.array(data['result']['data'])
event = data['result']['event']
urlabels = data['result']['urevent']
base = data['result']['chanlocs']

numTime = len(eeg_data1[:,0])
numElectrodes = len(eeg_data1[0,:])

print "Number of Time Steps in Total:" , numTime
print "Number of Electrodes Included:" , numElectrodes

latency = [event[e[0]][:][0][0] for e in event['latency']]
for i in range(len(latency)):
    latency[i] = int(latency[i])
print "The Time Stamp Triggers are:", latency

labels = []
temp = []
for i in range(len(latency) + 1):
    if i == 0:
        temp = np.full(latency[i], i+1, dtype = int)
    elif i <> len(latency):
        temp = np.full(latency[i]-latency[i-1], i+1, dtype = int)
    else:
        temp = np.full(len(eeg_data1[:,0]) - latency[i-1], i+1, dtype = int)
    labels = np.concatenate((labels,temp))

eeg_labeled = []
eeg_labeled = np.zeros((numTime, numElectrodes+1))
eeg_labeled[:,:-1] = eeg_data1
eeg_labeled[:,0] = labels

for i in range(len(latency) + 1):
    print "For Trial#", i+1
    if i == 0:
        eeg_slice = eeg_labeled[0:latency[i],:]
        print eeg_slice.shape
    elif i <> len(latency):
        eeg_slice = eeg_labeled[latency[i-1]:latency[i],:]
        print eeg_slice.shape
    else:
        eeg_slice = eeg_labeled[latency[i-1]:numTime,:]
        print eeg_slice.shape
    plot_heat(eeg_slice.T[:, ::100], 'Subject 1, microVoltage of Channels over time')



