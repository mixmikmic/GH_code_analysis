import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx

import mygene
mg = mygene.MyGeneInfo()


# latex rendering of text in graphs
import matplotlib as mpl
mpl.rc('text', usetex = False)
mpl.rc('font', family = 'serif')

get_ipython().magic('matplotlib inline')

import visJS2jupyter.visJS_module 
import visJS2jupyter.visualizations

reactome_df = pd.read_csv('../../interactomes/reactome/reactome.homo_sapiens.interactions.psi-mitab.txt',sep='\t')
reactome_df.head()

# create a networkx graph from the pandas dataframe, with all the other columns as edge attributes
attribute_cols = reactome_df.columns.tolist()[2:]
G_reactome = nx.from_pandas_dataframe(reactome_df,source='#ID(s) interactor A',target = 'ID(s) interactor B',
                                      edge_attr = attribute_cols)
len(G_reactome.nodes())

# check that edge attributes have been loaded
G_reactome.edges(data=True)[0]



# only keep nodes which have uniprot ids
uniprot_nodes = []
for n in G_reactome.nodes():
    if n.startswith('uniprot'):
        uniprot_nodes.append(n)

len(uniprot_nodes)

G_reactome = nx.subgraph(G_reactome,uniprot_nodes)
    

# take the largest connected component (to speed up visualization)
G_LCC = max(nx.connected_component_subgraphs(G_reactome), key=len)
len(G_LCC.nodes())

#mg_temp = mg.querymany(genes_temp,fields='symbol')
# parse the uniprot ids to HGNC gene symbols
uniprot_temp = [n[n.find(':')+1:] for n in G_LCC.nodes()]
mg_temp = mg.querymany(uniprot_temp,scopes='uniprot',species=9606)
uniprot_list = ['uniprotkb:'+x['query'] for x in mg_temp]
symbol_list = [x['symbol'] if 'symbol' in x.keys() else 'uniprotkb:'+x['query'] for x in mg_temp]
uniprot_to_symbol = dict(zip(uniprot_list,symbol_list))
uniprot_to_symbol = pd.Series(uniprot_to_symbol)
uniprot_to_symbol.head()

# relabel the nodes with their gene names
G_LCC = nx.relabel_nodes(G_LCC,dict(uniprot_to_symbol))
G_LCC.nodes()[0:10]



# map from interaction type to integer, and add the integer as an edge attribute

int_types = reactome_df['Interaction type(s)'].unique().tolist()
int_types_2_num = dict(zip(int_types,range(len(int_types))))
num_2_int_types = dict(zip(range(len(int_types)),int_types))

int_num_list = []
for e in G_LCC.edges(data=True):
    int_type_temp = e[2]['Interaction type(s)']
    int_num_list.append(int_types_2_num[int_type_temp])
    
# add int_num_list as attribute
int_num_dict = dict(zip(G_LCC.edges(),int_num_list))
nx.set_edge_attributes(G_LCC,'int_type_numeric',int_num_dict)


# set up the edge title for displaying info about interaction type
edge_title = {}
for e in G_LCC.edges():
    edge_title[e]=num_2_int_types[int_num_dict[e]]
    
# add node degree as a node attribute
deg = nx.degree(G_LCC)
nx.set_node_attributes(G_LCC,'degree',deg)



# set the layout with networkx spring_layout
pos = nx.spring_layout(G_LCC)

# plot the Reactome largest connected component with edges color-coded by interaction type
nodes = G_LCC.nodes()
numnodes = len(nodes)
edges = G_LCC.edges()
numedges = len(edges)
edges_with_data = G_LCC.edges(data=True)

# draw the graph here

edge_to_color = visJS2jupyter.visJS_module.return_edge_to_color(G_LCC,field_to_map = 'int_type_numeric',cmap=mpl.cm.Set1_r,alpha=.9)

nodes_dict = [{"id":n,"degree":G_LCC.degree(n),"color":'black',
              "node_size":deg[n],'border_width':0,
              "node_label":n,
              "edge_label":'',
              "title":n,
              "node_shape":'dot',
              "x":pos[n][0]*1000,
              "y":pos[n][1]*1000} for n in nodes
              ]
node_map = dict(zip(nodes,range(numnodes)))  # map to indices for source/target in edges

edges_dict = [{"source":node_map[edges[i][0]], "target":node_map[edges[i][1]], 
              "color":edge_to_color[edges[i]],"title":edge_title[edges[i]]} for i in range(numedges)]

visJS2jupyter.visJS_module.visjs_network(nodes_dict,edges_dict,
                            node_color_border='black',
                            node_size_field='node_size',
                            node_size_transform='Math.sqrt',
                            node_size_multiplier=1,
                            node_border_width=1,
                            node_font_size=40,
                            node_label_field='node_label',
                            edge_width=2,
                            edge_smooth_enabled=False, 
                            edge_smooth_type='continuous',
                            physics_enabled=False,
                            node_scaling_label_draw_threshold=100,
                            edge_title_field='title',
                            graph_title = 'Reactome largest connected component')



