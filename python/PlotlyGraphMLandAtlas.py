import os
os.chdir('/Users/Tony/Documents/Git Folder/seelviz/graphfiles/LukeGraphs/')

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly import tools
import plotly
plotly.offline.init_notebook_mode()

import numpy as np
import nibabel as nb
import networkx as nx
import pandas as pd

import plotly.plotly as py
from plotly.graph_objs import *

# Change the filename below to run different graphml files
graphMLfilename = 'Fear199localeq.5000.graphml'
G = nx.read_graphml(graphMLfilename)

# pandas DataFrame containing X, Y, Z coordinates of each node (note top row was changed to nodes,x,y,z)
csvfilepath = '/Users/Tony/Documents/Git Folder/seelviz/graphfiles/Fear199localeq.nodes.csv'
datapanda = pd.read_csv(
    filepath_or_buffer = csvfilepath,      # Filepath
    delimiter = ','                        # Delimiter
)

G.edges()

print G.nodes()

print datapanda

def get_brain_figure(g, atlas_data, plot_title=''):
    """
    Returns the plotly figure object for vizualizing a 3d brain network.

    g: igraph object of brain

    atlas_data: pandas DataFrame containing the x,y,z coordinates of
    each brain region


    Example
    -------
    import plotly
    plotly.offline.init_notebook_mode()

    fig = get_brain_figure(g, atlas_data)
    plotly.offline.iplot(fig)
    """

    # grab the node positions from the centroids file
    V = atlas_data.shape[0]
    node_positions_3d = pd.DataFrame(columns=['x', 'y', 'z'], index=range(V))
    for r in range(V):
        node_positions_3d.loc[r] = atlas_data.loc[r, ['x', 'y', 'z']].tolist()

    # grab edge endpoints
    edge_x = []
    edge_y = []
    edge_z = []

    for e in g.edges_iter():
        strippedSource = int(e[0].replace('s', ''))
        strippedTarget = int(e[1].replace('s', ''))
        source_pos = node_positions_3d.loc[strippedSource]
        target_pos = node_positions_3d.loc[strippedTarget]
    
        edge_x += [source_pos['x'], target_pos['x'], None]
        edge_y += [source_pos['y'], target_pos['y'], None]
        edge_z += [source_pos['z'], target_pos['z'], None]

    # node style
    node_trace = Scatter3d(x=node_positions_3d['x'],
                           y=node_positions_3d['y'],
                           z=node_positions_3d['z'],
                           mode='markers',
                           # name='regions',
                           marker=Marker(symbol='dot',
                                         size=6,
                                         color='red'),
                           # text=[str(r) for r in range(V)],
                           text=atlas_data['nodes'],
                           hoverinfo='text')

    # edge style
    edge_trace = Scatter3d(x=edge_x,
                           y=edge_y,
                           z=edge_z,
                           mode='lines',
                           line=Line(color='black', width=0.5),
                           hoverinfo='none')

    print edge_x
    
    # axis style
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False)

    # overall layout
    layout = Layout(title=plot_title,
                    width=800,
                    height=900,
                    showlegend=False,
                    scene=Scene(xaxis=XAxis(axis),
                                yaxis=YAxis(axis),
                                zaxis=ZAxis(axis)),
                    margin=Margin(t=50),
                    hovermode='closest')

    data = Data([node_trace, edge_trace])
    fig = Figure(data=data, layout=layout)

    return fig

output = get_brain_figure(G, datapanda, '')

iplot(output, validate=False)

edge_x = []
source_pos = node_positions_3d.loc[0]
target_pos = node_positions_3d.loc[1]
print source_pos
print source_pos.name
#print target_pos
edge_x += [source_pos['x'], target_pos['x'], None]
#print edge_x

# grab edge endpoints
edge_x = []
edge_y = []
edge_z = []

for e in G.edges_iter():
    strippedSource = int(e[0].replace('s', ''))
    strippedTarget = int(e[1].replace('s', ''))
    source_pos = node_positions_3d.loc[strippedSource]
    target_pos = node_positions_3d.loc[strippedTarget]
    
    edge_x += [source_pos['x'], target_pos['x'], None]
    edge_y += [source_pos['y'], target_pos['y'], None]
    edge_z += [source_pos['z'], target_pos['z'], None]

