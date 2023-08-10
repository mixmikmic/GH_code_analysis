import pandas as pd
import scipy.sparse as ss
import numpy as np
from sklearn.decomposition import TruncatedSVD
import sklearn.manifold
import tsne
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

conditional_prob_matrix = count_matrix.tocsr()
row_sums = np.array(conditional_prob_matrix.sum(axis=1))[:,0]
row_indices, col_indices = conditional_prob_matrix.nonzero()
conditional_prob_matrix.data /= row_sums[row_indices]

reduced_vectors = TruncatedSVD(n_components=500,
                               random_state=0).fit_transform(conditional_prob_matrix)

reduced_vectors /= np.sqrt((reduced_vectors**2).sum(axis=1))[:, np.newaxis]

seed_state = np.random.RandomState(0)
subreddit_map = tsne.bh_sne(reduced_vectors[:10000], perplexity=50.0, random_state=seed_state)

subreddit_map_df = pd.DataFrame(subreddit_map, columns=('x', 'y'))
subreddit_map_df['subreddit'] = subreddits[:10000]
subreddit_map_df.head()

import hdbscan

clusterer = hdbscan.HDBSCAN(min_samples=5, 
                            min_cluster_size=20).fit(subreddit_map)
cluster_ids = clusterer.labels_

subreddit_map_df['cluster'] = cluster_ids

from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import HoverTool, ColumnDataSource, value
from bokeh.models.mappers import LinearColorMapper
from bokeh.palettes import viridis
from collections import OrderedDict

output_notebook()

color_mapper = LinearColorMapper(palette=viridis(256), low=0, high=cluster_ids.max())
color_dict = {'field': 'cluster', 'transform': color_mapper}

plot_data_clusters = ColumnDataSource(subreddit_map_df[subreddit_map_df.cluster >= 0])
plot_data_noise = ColumnDataSource(subreddit_map_df[subreddit_map_df.cluster < 0])

tsne_plot = figure(title=u'A Map of Subreddits',
                   plot_width = 700,
                   plot_height = 700,
                   tools= (u'pan, wheel_zoom, box_zoom,'
                           u'box_select, resize, reset'),
                   active_scroll=u'wheel_zoom')
tsne_plot.add_tools( HoverTool(tooltips = OrderedDict([('subreddit', '@subreddit'),
                                                       ('cluster', '@cluster')])))


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



