import pandas as pd
import scipy.sparse as ss
import numpy as np
import sklearn.manifold
import re

raw_data = pd.read_csv('subreddit-overlap')

raw_data.head()

subreddit_popularity = raw_data.groupby('t2_subreddit')['NumOverlaps'].sum()
subreddits = np.array(subreddit_popularity.sort_values(ascending=False).index)

index_map = dict(np.vstack([subreddits, np.arange(subreddits.shape[0])]).T)

count_matrix = ss.coo_matrix((raw_data.NumOverlaps, 
                              (raw_data.t2_subreddit.map(index_map),
                               raw_data.t1_subreddit.map(index_map))),
                             shape=(subreddits.shape[0], subreddits.shape[0]),
                             dtype=np.float64)

count_matrix

count_matrix.data = 1.0 / count_matrix.data
count_matrix.data

count_matrix = count_matrix.tolil()

normalizing_values = np.ones(10000)
for i, row in enumerate(count_matrix.data[:10000]):
    normalizing_values[i] = np.sort(row)[50]
normalizing_values

for i, row in enumerate(count_matrix.data[:10000]):
    for j in range(len(row)):
        count_matrix.data[i][j] /= normalizing_values[i]

count_matrix = count_matrix.tocsr()[:10000,:][:,:10000]

count_matrix.data = np.exp(-count_matrix.data**2)

count_matrix.data[count_matrix.data < 0.25] = 0.0
count_matrix.eliminate_zeros()
count_matrix

joint_prob_matrix = np.sqrt(count_matrix * count_matrix.T)
joint_prob_matrix /= joint_prob_matrix.sum()
joint_prob_ndarray = joint_prob_matrix.toarray()
joint_prob_ndarray[range(joint_prob_ndarray.shape[0]),range(joint_prob_ndarray.shape[0])] = 0.0

neighbors = []
for row in joint_prob_ndarray:
    neighbors.append((np.argsort(row)[-150:])[::-1])
neighbors = np.array(neighbors)

neighbors

P = sklearn.manifold.t_sne.squareform(joint_prob_ndarray)
embedder = sklearn.manifold.TSNE(perplexity=50.0, 
                                 init='pca', 
                                 n_iter=2000, 
                                 n_iter_without_progress=60)
random_state = sklearn.manifold.t_sne.check_random_state(embedder.random_state)
subreddit_map = embedder._tsne(P, 1, joint_prob_ndarray.shape[0], random_state,
                               neighbors=neighbors)

subreddit_map_df = pd.DataFrame(subreddit_map[:10000], columns=('x', 'y'))
subreddit_map_df['subreddit'] = subreddits[:10000]
subreddit_map_df.head()

import hdbscan

clusterer = hdbscan.HDBSCAN(min_samples=5, 
                            min_cluster_size=20).fit(subreddit_map[:10000])
cluster_ids = clusterer.labels_

subreddit_map_df['cluster_id'] = cluster_ids

from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import HoverTool, ColumnDataSource, value
from bokeh.models.mappers import LinearColorMapper, CategoricalColorMapper
from bokeh.palettes import viridis
from collections import OrderedDict

output_notebook()

color_mapper = LinearColorMapper(palette=viridis(256), low=0, high=cluster_ids.max())
color_dict = {'field': 'cluster_id', 'transform': color_mapper}

plot_data_clusters = ColumnDataSource(subreddit_map_df[subreddit_map_df.cluster_id >= 0])
plot_data_noise = ColumnDataSource(subreddit_map_df[subreddit_map_df.cluster_id < 0])

tsne_plot = figure(title=u'A Map of Subreddits',
                   plot_width = 700,
                   plot_height = 700,
                   tools= (u'pan, wheel_zoom, box_zoom,'
                           u'box_select, resize, reset'),
                   active_scroll=u'wheel_zoom')
tsne_plot.add_tools( HoverTool(tooltips = OrderedDict([('subreddit', '@subreddit'),
                                                       ('cluster', '@cluster_id')])))


# draw clusters
tsne_plot.circle(u'x', u'y', source=plot_data_clusters,
                 fill_color=color_dict, line_alpha=0.002, fill_alpha=0.1,
                 size=10, hover_line_color=u'black')
# draw noise
tsne_plot.circle(u'x', u'y', source=plot_data_noise,
                 fill_color=u'gray', line_alpha=0.002, fill_alpha=0.05,
                 size=10, hover_line_color=u'black')

# configure visual elements of the plot
tsne_plot.title.text_font_size = value(u'16pt')
tsne_plot.xaxis.visible = False
tsne_plot.yaxis.visible = False
tsne_plot.grid.grid_line_color = None
tsne_plot.outline_line_color = None

show(tsne_plot);

def is_nsfw(subreddit):
    return re.search(r'(nsfw|gonewild)', subreddit)

for cid in range(cluster_ids.max() + 1):
    subreddits = subreddit_map_df.subreddit[cluster_ids == cid]
    if np.any(subreddits.map(is_nsfw)):
        subreddits = ' ... Censored ...'
    else:
        subreddits = subreddits.values
        
    print '\nCluster {}:\n{}\n'.format(cid, subreddits) 



