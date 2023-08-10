import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import six

sns.set(style='ticks', context='talk', rc={'font.sans-serif':'Arial', 'pdf.fonttype': 42})


import bonvoyage

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Figures in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Set random seed
np.random.seed(sum(map(ord, 'bonvoyage')))


# Define folder to save figures
folder = 'pdf'
get_ipython().system('mkdir -p $folder')

data = pd.read_csv('data.csv', index_col=0)
data.head()



waypoints = pd.read_csv('waypoints.csv', index_col=0)
waypoints.head()

metadata = pd.read_csv('metadata.csv')
metadata.head()

import bonvoyage

get_ipython().run_line_magic('pinfo2', 'bonvoyage.waypointplot')

get_ipython().run_line_magic('pinfo2', 'bonvoyage.visualize._waypoint_scatter')

plot_kinds = 'scatter', 'hex'

folder


for kind in plot_kinds:
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    if kind == 'scatter':
        kwargs = {'rasterized': True}
    else:
        kwargs = {}
    
    bonvoyage.waypointplot(waypoints, kind=kind, **kwargs)
    fig.savefig('{}/waypoints_all_{}.pdf'.format(folder, kind), dpi=300)

noise_percentages = 0, 25, 50, 75

for noise_percentage in noise_percentages:
    
    rows = metadata['% Noise'] == noise_percentage
    feature_ids = metadata.loc[rows, 'Feature ID']
    
    waypoints_subset = waypoints.loc[feature_ids]
    for kind in plot_kinds:
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
        kwargs = {'rasterized': True} if kind == 'scatter' else {}
        
        bonvoyage.waypointplot(waypoints_subset, kind=kind, **kwargs)
        fig.savefig('{}/waypoints_noise{}_{}.pdf'.format(folder, noise_percentage, kind), dpi=300)

import anchor

estimator = anchor.BayesianModalities()

modalities = estimator.fit_predict(data)
modalities.head()



