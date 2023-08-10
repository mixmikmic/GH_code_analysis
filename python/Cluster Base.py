import pandas as pd
import numpy as np
import seaborn as sns
import os
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

DATA_DIR = 'sample_csv'
file_names = os.listdir('./%s' % DATA_DIR)

FEAT_TRANSFORM = {
    'Points:0': 'x',
    'Points:1': 'y'
}
FEATURES = FEAT_TRANSFORM.values()

def load_edge(file_name):
    image_path = '%s/%s' % (DATA_DIR, file_name)
    return pd.read_csv(image_path)[list(FEAT_TRANSFORM.keys())].rename(columns=FEAT_TRANSFORM)

def frame_to_row(in_frame):
    cols = get_cols_from_frame(in_frame)
    unstacked = in_frame.unstack()
    return pd.Series(unstacked.ravel(), index=cols)

def get_cols_from_frame(in_frame):
    unstacked = in_frame.unstack()
    return list('y' + unstacked['y'].index.astype(str)) + list('x' + unstacked['x'].index.astype(str))

edges = []
for file_name in file_names:
    edge = axis_align_pandas(load_edge(file_name).sort_values(by='x'))
    edges.append(edge)
    edge.plot(kind="scatter", x='x', y='y')

downsampled_edges = []
for edge in edges:
    downsampled_edges.append(helpers.downsample(axis_align_pandas(edge), 1000))

total_frame_cols = get_cols_from_frame(downsampled_edges[0])
total_frame = pd.DataFrame(columns=total_frame_cols)

for edge in downsampled_edges:
    total_frame = total_frame.append(frame_to_row(edge), ignore_index=True)
    
# fr.append(pd.Series(unstacked.ravel(), index=cols), ignore_index=True)

total_frame

from sklearn.metrics.pairwise import euclidean_distances
pd.DataFrame(euclidean_distances(total_frame))

axis_align_pandas(edges[0])

from sklearn.cluster import KMeans

kmeans = KMeans(5)
kmeans.fit(total_frame)
kmeans.predict(total_frame)

