# import some useful packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import networkx as nx
import pandas as pd
import random
import community
import json
import os

# latex rendering of text in graphs
import matplotlib as mpl
mpl.rc('text', usetex = False)
mpl.rc('font', family = 'serif')


get_ipython().magic('matplotlib inline')

# import visJS_2_jupyter 
from visJS2jupyter import visJS_module

def load_DB_data(fname):
    '''
    Function to load drug bank data (in format of this file 'drugbank.0.json.new')
    '''

    with open(fname, 'r') as f:
        read_data = f.read()
    f.closed

    si = read_data.find('\'\n{\n\t"source":')
    sf = read_data.find('\ncurl')

    DBdict = dict()

    # fill in DBdict
    while si > 0:
        
        db_temp = json.loads(read_data[si+2:sf-2])
        DBdict[db_temp['drugbank_id']]=db_temp

        # update read_data
        read_data = read_data[sf+10:]
        
        si = read_data.find('\'\n{\n\t"source":')
        sf = read_data.find('\ncurl')
    return DBdict

DBdict = load_DB_data('drugbank.0.json.new')
# make a Series object to map from genes to drugs

drug_list = []
gene_list = []
for drug in DBdict.keys():
    
    gene_targets_temp = [n['name'] for n in DBdict[drug]['node_list']]
    gene_list.extend(gene_targets_temp)
    num_gene_targets = len(gene_targets_temp)
    
    drug_temp = [drug]*num_gene_targets
    drug_list.extend(drug_temp)
    
gene_to_drug = pd.Series(drug_list,index=gene_list)
drug_to_gene = pd.Series(gene_list,index=drug_list)
    

# list of TCGA cancer types
d_list =  ['ACC','BLCA','BRCA','CESC','CHOL','COAD','COADREAD','DLBC','ESCA','GBM','GBMLGG','HNSC','KICH','KIPAN',
           'KIRC','KIRP','LAML','LGG','LIHC','LUAD','LUSC','OV','PAAD','PCPG','PRAD','READ','SARC','SKCM',
           'STAD','STES','TGCT','THCA','UCEC','UCS','UVM']
len(d_list)

# if you don't have access to TCGA mutation files, just load the pre-computed edge list
num_mutations=25
mutation_EL = pd.read_csv('mutation_EL.csv')
G_mutation = nx.from_pandas_dataframe(mutation_EL,source='gene1',target='gene2',edge_attr = 'weighted_mutation_average')
mutation_EL.head()





# --- UNCOMMENT THIS CELL IF YOU HAVE ACCESS TO TCGA MUTATION_MATRIX.TXT files ----

# mutation_EL = []
# num_mutations = 25
# for d1 in d_list:
#     print(d1)

#     mutation_df = pd.read_csv('/Users/brin/Documents/TCGA_data/expression_files/'+d1+'/mutation_matrix.txt',sep='\t')
#     mutation_df.index=mutation_df['gene_name']
#     index_no_v = [index[:-2] for index in list(mutation_df.index)]
#     mutation_df.index = index_no_v
#     mutation_df['row_average'] = mutation_df.mean(axis=1)
#     mutation_df = mutation_df.sort('row_average',ascending=False)
#     # reorder the columns
#     cols = mutation_df.columns
#     cols_reorder = [cols[0]]
#     cols_reorder.append(cols[-1])
#     cols_reorder.extend(cols[1:-1])
#     mutation_df = mutation_df[cols_reorder]


#     # sort by individual patient mutation list
#     patient_name = 'row_average' #cols[patient_num]
#     mutation_df = mutation_df.sort(patient_name,ascending=False)
#     mutation_df.head()

#     mutation_EL_temp = zip(list(mutation_df.head(num_mutations).index),[d1]*num_mutations,list(mutation_df.head(num_mutations)['row_average']))
#     mutation_EL.extend(mutation_EL_temp)
    
# # make a graph from mutation_EL
# G_mutation = nx.Graph()
# G_mutation.add_weighted_edges_from(mutation_EL)

# mutation_DF = pd.DataFrame(mutation_EL)
# mutation_DF.to_csv('mutation_EL.csv')

# prep graph for visJS_2_jupyter visualization
# add BC and CC as attributes
# create nodes_dict and edges_dict for input to visjs
pos = nx.spring_layout(G_mutation,k=.2)

nodes = G_mutation.nodes()

numnodes = len(nodes)
edges = G_mutation.edges()
edges_with_data = G_mutation.edges(data=True)
numedges = len(edges)

# add a node attributes to color-code by
cc = nx.clustering(G_mutation)
degree = G_mutation.degree()
bc = nx.betweenness_centrality(G_mutation)
nx.set_node_attributes(G_mutation,'clustering_coefficient',cc)
nx.set_node_attributes(G_mutation,'degree',degree)
nx.set_node_attributes(G_mutation,'betweenness_centrality',bc)


# set node_size to degree
node_size = [int(float(n)/np.max(degree.values())*25+1) for n in degree.values()]
node_to_nodeSize = dict(zip(degree.keys(),node_size))


# add nodes to highlight (druggable genes)
nodes_HL = [1 if node in gene_to_drug.keys() else 0 for node in G_mutation.nodes()]  
nodes_HL = dict(zip(G_mutation.nodes(),nodes_HL))

nodes_shape=[]
node_shape = ['triangle' if (node in d_list) else 'dot' for node in G_mutation.nodes()]
node_to_nodeShape=dict(zip(G_mutation.nodes(),node_shape))

# add a field for node labels
node_labels_temp = []
list_of_genes = list(np.setdiff1d(G_mutation.nodes(),d_list))
for node in G_mutation.nodes():
    label_temp = node
    
    if node in list_of_genes:
        label_temp+= ': <br/>'
        
        label_temp+='mutation is frequently mutated set in ' + str(nx.degree(G_mutation,node)) + ' out of '+str(len(d_list))+' diseases<br/><br/>'
        label_temp+='Drugs targeting this gene: <br/>'
        
        
        if node in gene_to_drug.keys():
            drugs_temp = gene_to_drug[node]
            if type(drugs_temp)==unicode:
                drugs_temp = [drugs_temp]
            for d in drugs_temp:
                label_temp+=d + '<br/>'
        else:
            label_temp+='None'
        
    
    node_labels_temp.append(label_temp)

node_labels = dict(zip(G_mutation.nodes(),node_labels_temp))

node_titles = [node for node in G_mutation.nodes()]

node_titles = dict(zip(G_mutation.nodes(),node_titles))
    
    

# draw the graph here

scaling_factor=1

node_to_color = visJS_module.return_node_to_color(G_mutation,field_to_map='degree',cmap=mpl.cm.jet,alpha = 1,
                                                  color_vals_transform=None,
                                                 color_max_frac = .9,color_min_frac = .2)

edge_to_color = visJS_module.return_edge_to_color(G_mutation,field_to_map = 'weighted_mutation_average',cmap=mpl.cm.Blues,alpha=.4)

nodes_dict = [{"id":n,"degree":G_mutation.degree(n),"color":node_to_color[n],
              "node_size":node_to_nodeSize[n],'border_width':nodes_HL[n],
              "node_label":node_labels[n],
              "edge_label":'',
              "title":node_labels[n],
              "node_shape":node_to_nodeShape[n],
              "x":pos[n][0]*1000*scaling_factor,
              "y":pos[n][1]*1000*scaling_factor} for n in nodes
              ]


node_map = dict(zip(nodes,range(numnodes)))  # map to indices for source/target in edges

edges_dict = [{"source":node_map[edges[i][0]], "target":node_map[edges[i][1]], 
              "color":edge_to_color[edges[i]],"title":edges_with_data[i][2]['weighted_mutation_average']} for i in range(numedges)]

visJS_module.visjs_network(nodes_dict,edges_dict,
                            node_color_highlight_border="black",
                            node_color_hover_border = 'orange',
                            node_color_border='black',
                            node_size_field='node_size',
                            node_size_transform='Math.sqrt',
                            node_size_multiplier=3*scaling_factor,
                            node_border_width=1*scaling_factor,
                            hover = False,
                            node_label_field='id',
                            edge_width=1*scaling_factor,
                            hover_connected_edges = False,
                            physics_enabled=False,
                            min_velocity=.5,
                            max_velocity=16,
                            draw_threshold=21,
                            min_label_size=12*scaling_factor,
                            max_label_size=25*scaling_factor,
                            max_visible=10*scaling_factor,
                            scaling_factor=scaling_factor,
                            edge_title_field='title',
                            graph_title = 'mutation graph- top '+str(num_mutations)+' mutations in each disease')

# change focal_gene, to see which drugs have been associated with it, in the DrugBank database
focal_gene = 'MGAM'

print('drugs associated with ' + focal_gene+': ')
if focal_gene in gene_to_drug.keys():
    drugs_temp = gene_to_drug[focal_gene]
    if type(drugs_temp)==unicode:
        drugs_temp = [drugs_temp]

    
    for d in drugs_temp:
        print(d)
else:
    print('None')



