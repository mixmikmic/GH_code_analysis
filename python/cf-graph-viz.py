import os
import pandas as pd
import holoviews as hv
import networkx as nx

from colorcet import fire
from datashader.bundling import directly_connect_edges, hammer_bundle
from datashader.layout import random_layout, circular_layout, forceatlas2_layout

from holoviews.operation.datashader import datashade, dynspread
from holoviews.operation import decimate

from dask.distributed import Client
client = Client()

hv.notebook_extension('bokeh','matplotlib')

decimate.max_samples=20000
dynspread.threshold=0.01
datashade.cmap=fire[40:]
sz = dict(width=150,height=150)

get_ipython().run_line_magic('opts', 'RGB [xaxis=None yaxis=None show_grid=False bgcolor="black"]')

# stackoverflow https://stackoverflow.com/questions/12329853/how-to-rearrange-pandas-column-sequence
# reorder columns
def set_column_sequence(dataframe, seq, front=True):
    '''Takes a dataframe and a subsequence of its columns,
       returns dataframe with seq as first columns if "front" is True,
       and seq as last columns if "front" is False.
    '''
    cols = seq[:] # copy so we don't mutate seq
    for x in dataframe.columns:
        if x not in cols:
            if front: #we want "seq" to be in the front
                #so append current column to the end of the list
                cols.append(x)
            else:
                #we want "seq" to be last, so insert this
                #column in the front of the new column list
                #"cols" we are building:
                cols.insert(0, x)
    return dataframe[cols]

r_graph_file = os.getenv('CF_GRAPH')
r_graph = nx.read_yaml(r_graph_file)
pd_nodes = pd.DataFrame([(node, node) for node in r_graph.nodes], columns=['id', 'node'])
pd_nodes.set_index('id', inplace=True)
pd_edges = pd.DataFrame(list(r_graph.edges), columns=['source', 'target'])

get_ipython().run_line_magic('time', 'fa2_layout = forceatlas2_layout(pd_nodes, pd_edges)')

r_nodes = hv.Points(set_column_sequence(fa2_layout, ['x', 'y']) , label='Nodes')
r_edges = hv.Curve(pd_edges, label='Edges')

get_ipython().run_cell_magic('opts', 'RGB [tools=["hover"] width=400 height=400] ', '\n%time r_direct = hv.Curve(directly_connect_edges(r_nodes.data, r_edges.data),label="Direct")\n\ndynspread(datashade(r_nodes,cmap=["cyan"])) + \\\ndatashade(r_direct)')

get_ipython().run_line_magic('time', 'r_bundled = hv.Curve(hammer_bundle(r_nodes.data, r_edges.data),label="Bundled")')

get_ipython().run_cell_magic('opts', 'RGB [tools=["hover"] width=400 height=400] ', '\ndynspread(datashade(r_nodes,cmap=["cyan"])) + \\\ndatashade(r_bundled)')

get_ipython().run_cell_magic('opts', 'Points (color="cyan", size=3) [tools=["hover"] width=900 height=650] ', 'datashade(r_bundled, width=900, height=650) * \\\ndecimate(hv.Points(r_nodes),max_samples=1000)')

