# load libraries and set plot parameters
import numpy as np

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sklearn.cluster import KMeans

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
# lets take half of the input data for training, and half for testing
# and create numpy arrays
half = int(len(points)/2)
train_data = all_data[:half]
test = all_data[half:]

#plt.figure(figsize=(16, 10))
plt.plot(train_data)
plt.xlabel("Reading number")
plt.ylabel("Water level")

from sklearn import preprocessing
train = preprocessing.scale(np.copy(train_data))
plt.plot(train)
plt.xlabel("Reading number")
plt.ylabel("Water level")

data2d = open('2d.csv').readlines()
xdata = []
ydata = []
for line in data2d:
    x,y = line.strip().split(',')
    xdata.append(x)
    ydata.append(y)
    
plt.scatter(x=xdata,y=ydata)

# calculate and plot the cluster centroids
zipped = zip(xdata, ydata)
coords = [[a,b] for a,b in zipped]
num_clusters = 2
clusterer = KMeans(n_clusters=num_clusters)
clusterer.fit(coords)
centers_x = [l[0] for l in clusterer.cluster_centers_]
centers_y = [l[1] for l in clusterer.cluster_centers_]

plt.scatter(centers_x, centers_y, marker='o', c='crimson', s=80)
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
plt3d = fig.add_subplot(111, projection='3d')
xdata = []
ydata = []
zdata = []
coords = []
lines = open('3d.csv').readlines()
for line in lines:
    x,y,z = line.strip().split(',')
    coords.append([x,y,z])
    xdata.append(float(x))
    ydata.append(float(y))
    zdata.append(float(z))
    
plt3d.scatter(xdata, ydata, zdata, c='r', marker='o')

num_clusters = 2
clusterer = KMeans(n_clusters=num_clusters)
clusterer.fit(coords)
centers_x = [l[0] for l in clusterer.cluster_centers_]
centers_y = [l[1] for l in clusterer.cluster_centers_]
centers_z = [l[2] for l in clusterer.cluster_centers_]

plt3d.scatter(centers_x, centers_y, centers_z, marker='o', c='blue', s=80)

plt.show()


# based on https://github.com/mrahtz/sanger-machine-learning-workshop/blob/master/learn_utils.py
def segment(data, segment_len, step_size):
    segments = []

    for start in range(0, len(data), step_size):
        end = start + segment_len
        segment = np.copy(data[start:end])
        # if the final segment is shorter than our segment length, discard it
        if len(segment) != segment_len:
            continue
        segments.append(segment)

    return segments

segments = segment(train, 90, 9)

# based on https://github.com/mrahtz/sanger-machine-learning-workshop/blob/master/learn_utils.py
def plot_segments(segments, step):
    plt.figure(figsize=(12,12))
    num_cols = 4
    num_rows = 4
    graph_num = 1
    seg_num = 1

    for _ in range(num_rows):
        for _ in range(num_cols):
            subplt = plt.subplot(num_rows, num_cols, graph_num)
            plt.plot(segments[seg_num])
            graph_num += 1
            seg_num += step

    plt.tight_layout()
    plt.show()

plot_segments(segments, 3)

rad = np.linspace(0, np.pi, 90)
window = np.sin(rad)**2
plt.plot(window)

def window_segments(segments, segment_len):
    rad = np.linspace(0, np.pi, segment_len)
    window = np.sin(rad)**2
    windowed_segments = [np.copy(segment) * window for segment in segments]
           
    return windowed_segments

windowed_segments = window_segments(segments, 90)
plot_segments(windowed_segments, 3)

num_clusters = 40
clusterer = KMeans(n_clusters=num_clusters)
clusterer.fit(windowed_segments)

centroids = clusterer.cluster_centers_
plot_segments(centroids,2)

# based on https://github.com/mrahtz/sanger-machine-learning-workshop/blob/master/learn_utils.py
def reconstruct(data, segment_len, clusterer):
    slide_len = int(segment_len / 2)
    segments = segment(data, segment_len, slide_len)
    windowed_segments = window_segments(segments, segment_len)
    reconstructed = np.zeros(len(data))
    for segment_num, seg in enumerate(windowed_segments):
        # calling seg.reshape(1,-1) is done to avoid a DeprecationWarning from sklearn
        nearest_centroid_idx = clusterer.predict(seg.reshape(1,-1))[0]
        nearest_centroid = np.copy(clusterer.cluster_centers_[nearest_centroid_idx])
        pos = segment_num * slide_len
        reconstructed[pos:pos+segment_len] += nearest_centroid

    return reconstructed

reconstructed_data = reconstruct(train, 90, clusterer)

error = reconstructed_data - train
plt.plot(train, label="Original training data")
plt.plot(reconstructed_data, label="Reconstructed training data")
plt.plot(error, label="Reconstruction error")
plt.legend()
plt.show()

max_error = np.absolute(error).max()
error_98th_percentile = np.percentile(error, 98)
print('The maxiumum reconstruction error is: {:0.2f}'.format(max_error))
print('The 98th percentile of reconstruction error is: {:0.2f}'.format(error_98th_percentile))

