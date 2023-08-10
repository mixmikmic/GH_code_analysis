get_ipython().magic('matplotlib inline')
import sys
sys.path.insert(0,'..')
from IPython.display import HTML,Image,SVG,YouTubeVideo
from helpers import header

HTML(header())

import networkx as nx
G=nx.DiGraph()
G.add_weighted_edges_from([('y','s',7),('y','v',6),('x','y',2),('x','u',3),('x','v',9),
                          ('s','x',5),('s','u',10),('u','x',2),('u','v',1),('v','y',4)])

pos=nx.spring_layout(G) 
edgewidth=[]
for (u,v,d) in G.edges(data=True):
    edgewidth.append(d['weight'])
    
nx.draw_networkx_nodes(G,pos=pos)
nx.draw_networkx_labels(G,pos=pos,font_size=20,font_family='sans-serif')
nx.draw_networkx_edges(G,pos,alpha=0.3,width=edgewidth, edge_color='m',arrows=True)

path=nx.all_pairs_dijkstra_path(G)
print(path['s']['v'])
print(path['v']['s'])



