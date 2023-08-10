#Import necessary python libraries

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')

import warnings
warnings.filterwarnings('ignore')

a = pd.read_csv("https://panamadata.blob.core.windows.net/icij/all_edges.csv")
e = pd.read_csv("https://panamadata.blob.core.windows.net/icij/Entities.csv")
i = pd.read_csv("https://panamadata.blob.core.windows.net/icij/Intermediaries.csv")
o = pd.read_csv("https://panamadata.blob.core.windows.net/icij/Officers.csv")
ad = pd.read_csv("https://panamadata.blob.core.windows.net/icij/Addresses.csv")

# All Edges
a.head(3)

# Entities
e.head(3)

#Intermediaries
i.head(3)

#Officers
o.head(3)

#Addresses
ad.head(3)

pp_edge = pd.DataFrame(a)
pp_entity = pd.DataFrame(e)
pp_intermediary = pd.DataFrame(i)
pp_officer = pd.DataFrame(o)
pp_address = pd.DataFrame(ad)

# Use pandas groupby() frunction to determine loactions with the highest incorporation rates
# Here we're counting how many entities there are per jurisdiction

shell_locations = pp_entity.groupby(['jurisdiction_description'])["name"].count().reset_index(name="Entity Count")
shell_locations

#Sample edge data
pp_edge[pp_edge['node_2'] == 10000001]

#Create a new DataFrame with records with only a "intermediary of" relationship type
pp_edge_2 = pp_edge[pp_edge['rel_type'] == "intermediary of"]

pp_edge_2.head(3)

# Merge 1
# Pandas function - merge()
# Type of join performed - inner join to only get records that are present in both the entities and edges datasets

test_1 = pd.merge(pp_entity,pp_edge_2, how='inner', left_on='node_id',right_on = 'node_2')
test_1.head(3)

# Merge 2
# From Merge 1 performed above and the intermediaries dataset, create a new dataset

test_2 = pd.merge(test_1,pp_intermediary, how='inner', left_on='node_1',right_on = 'node_id')
test_2.head(3)

# Import "collections" python library in order to create an ordered dictionary

import collections
new_dict = {'name': test_2['name_x'],
            'jurisdiction_description': test_2['jurisdiction_description'],
            'address': test_2['address_x'],
            'intermediary_name': test_2['name_y'],
            'country_code': test_2['country_codes_y'],
            'country': test_2['countries_y'],
            'status': test_2['status_y']
           }
name_dict = test_2['name_x']
jurisdiction_dict = test_2['jurisdiction_description']
address_dict = test_2['address_x']
intermediary_dict = test_2['name_y']
code_dict = test_2['country_codes_y']
country_dict = test_2['countries_y']
status_dict = test_2['status_y']
new_dict = collections.OrderedDict(new_dict)

# Convert our new dictionary into a Pandas DataFrame and display the summary details

df = pd.DataFrame(new_dict)
df.info()

#Limiting the data set to specific jurisdictions with the most data
#scope_list = ["Bahamas","British Virgin Islands","Niue","Panama","Samoa","Seychelles","Undetermined"]

#A sub scope list is used for faster processing. One can switch to the larger set in the above commented list
scope_list = ["Niue","Samoa"]

#loc - label-location based indexer for selection by label.
df_2 = df.loc[df['jurisdiction_description'].isin(scope_list)]        

df_2.info()

df_2.head(3)

#Create tuples of nodes and edges
edge1 = zip(df_2['jurisdiction_description'],df_2['intermediary_name'].unique())
edge2 = zip(df_2['intermediary_name'].unique(),df_2['name'])
edge3 = zip(df_2['name'],df_2['country_code'].unique())
mylist1 = list(edge1)
mylist2 = list(edge2)
mylist3 = list(edge3)

# Create a networkx graph using mylist1, mylist2, and mylist3 sets of edges
G = nx.Graph()
G.add_edges_from(mylist1)
G.add_edges_from(mylist2)
G.add_edges_from(mylist3)
print(nx.info(G))

# Center of Graph - center of current graph
nx.center(G)

# Closeness Centrality
nx.closeness_centrality(G)

# Betweenness Centrality
nx.betweenness_centrality(G)

nx.draw(G)
plt.show()

# This graph is included to show the challenge of showing our data with label
nx.draw_networkx(G)
plt.show

