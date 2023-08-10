#Importing necessary packages

import geopandas as gpd
import pandas as pd
import matplotlib.pylab as plt
import fiona
import osmnx as ox
import networkx as nx
from geopy import Nominatim

get_ipython().run_line_magic('matplotlib', 'inline')

areas = ['Santa monica, Los Angeles County, California',
         'Boulder, Colorado, USA',
         'Manhattan, NY, USA']

def Download_graph(query, network_type='drive', timeout=10, return_n=False, draw=True):
    """
    Use osmnx.graph_from_place to download street network, 
    automatically select the result that return a polygon and visualize it.
    
    Parameters
    ----------
    query : string or dict or list
        the place(s) to geocode/download data for
    network_type : string
        what type of street network to get
    timeout : int
        the timeout interval for requests and to pass to API
    return_n : bool
        if True, return which result is selected
    draw : bool
        if True, plot the polygon
        
    Returns
    -------
    networkx multidigraph or networkx multidigraph and which_result as a tuple
    """
    
    n = 1
    while True:
        try:
            nw = ox.graph_from_place(query=query, network_type=network_type, timeout=timeout, which_result=n)
            break
        except ValueError:
            n += 1
    nw_p = ox.project_graph(nw)
    
    if draw:
        print(nx.info(nw_p))
        ox.plot_graph(nw_p, fig_height=10)
    
    if return_n:
        return (nw_p, n)
    else:
        return nw_p

nw = list(map(Download_graph, areas))

(gdf_nodes, gdf_edges) = ox.save_load.graph_to_gdfs(nw[0])

gdf_edges[gdf_edges.name == '2nd Street'].head()

gdf_edges.loc[21:21]

fig, ax = plt.subplots(figsize=(15,15))
gdf_edges.plot(ax=ax, alpha=0.5)
gdf_edges[gdf_edges.name == '2nd Street'].plot(ax=ax, color='black', linewidth=3)
gdf_edges.loc[21:21].plot(ax=ax, color='red', linewidth=4)
ax.set_title('2nd Street', fontsize=30)
ax.set_axis_off()

