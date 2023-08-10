# load libraries and set plot parameters
import numpy as np
from scipy.signal import savgol_filter

from anomaly_utils import segment, plot_segments, window_segments, cluster, reconstruct, init_pyplot

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

init_pyplot()

data = open('watertank_data.csv').readlines()
points = [float(p.strip()) for p in data]
all_data = np.array(points)
# lets take half of the input data for training, and half for testing
# and create numpy arrays
half = int(len(points)/2)
train_data = all_data[:half]

# scale the data top be centered on zero
from sklearn import preprocessing
train = preprocessing.scale(np.copy(train_data))

plt.plot(train)
plt.xlabel("Reading number")
plt.ylabel("Water level")

smoothed_train = savgol_smoothed = savgol_filter(train, 51, 2)
plt.plot(smoothed_train)
plt.xlabel("Reading number")
plt.ylabel("Water level")

segment_length = 90
step_size = 9
num_clusters = 40

# segment the data
segments = segment(smoothed_train, segment_length, step_size)

# apply window function to all segments
windowed_segments = window_segments(segments, segment_length) 

# cluster the segments
clusterer = cluster(windowed_segments, num_clusters)

# reconstruct the training time-series
reconstructed_data = reconstruct(smoothed_train, segment_length, clusterer)
error = reconstructed_data - smoothed_train

# plot the results
plt.plot(smoothed_train, label="Original training data")
plt.plot(reconstructed_data, label="Reconstructed training data")
plt.plot(error, label="Reconstruction error")
plt.legend()
plt.show()

max_error = error.max()
error_98th_percentile = np.percentile(error, 98)
print('The maxiumum reconstruction error is: {:0.2f}'.format(max_error))
print('The 98th percentile of reconstruction error is: {:0.2f}'.format(error_98th_percentile))

