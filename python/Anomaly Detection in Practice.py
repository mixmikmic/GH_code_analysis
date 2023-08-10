# load libraries and set plot parameters
import numpy as np
from scipy.signal import savgol_filter

from anomaly_utils import segment, plot_segments, window_segments, cluster, reconstruct

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 75

plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 10

plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = "sans serif"
plt.rcParams['font.serif'] = "cm"
plt.rcParams['text.latex.preamble'] = r"\usepackage{subdepth}, \usepackage{type1cm}"

data = open('watertank_data.csv').readlines()
points = [float(p.strip()) for p in data]
all_data = np.array(points)
half = int(len(points)/2)
train = all_data[1200:half]
train -= np.amin(train) + (np.amax(train) - np.amin(train))/2
smoothed_train = savgol_filter(train, 51, 2)

segment_length = 90
step_size = 9
num_clusters = 40

# segment the data
segments = segment(smoothed_train, segment_length, step_size)

# apply window function to all segments
windowed_segments = window_segments(segments, segment_length)

# cluster the segments
clusterer = cluster(windowed_segments, num_clusters)

test = all_data[1200:half]
test -= np.amin(train) + (np.amax(train) - np.amin(train))/2
smoothed_test = savgol_filter(test, 51, 2)
plt.plot(smoothed_test)
plt.show()
# force the last 120 readings to simulate water levels below normal
ramp = np.linspace(smoothed_test[-120], -12, 120)
smoothed_test[-120:] = ramp
plt.plot(smoothed_test)

# reconstruct the training time-series
reconstructed_data = reconstruct(smoothed_test, segment_length, clusterer)
error = reconstructed_data - smoothed_test

# plot the results
plt.plot(smoothed_test, label="Original training data")
plt.plot(reconstructed_data, label="Reconstructed training data")
plt.plot(error, label="Reconstruction error")
plt.legend()
plt.show()

