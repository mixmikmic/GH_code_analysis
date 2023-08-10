#Before you can do this lecture, you'll need to install networkx
#run  'conda install networkx' at the command line 

#this idea comes from http://kieranhealy.org/blog/archives/2013/06/09/using-metadata-to-find-paul-revere/

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

df = pd.read_csv("../data/PaulRevereAppD.csv", index_col='Name')

df.head()

df.shape

people_x_groups = df

df.T

groups_x_people = df.T

print(people_x_groups.shape)
print(groups_x_people.shape)

people_adj =  people_x_groups.dot(groups_x_people)

people_adj.shape

people_adj.head()

G = nx.from_numpy_matrix(people_adj.values)
G = nx.relabel_nodes(G, dict(enumerate(people_adj.columns)))


plt.figure(figsize=(40,40))
pos=nx.spring_layout(G, iterations=500, scale=5, k=.3)
nx.draw_networkx_labels(G,pos,fontsize=8)
font = {'fontname'   : 'Helvetica',
            'color'      : 'k',
            'fontweight' : 'bold',
            'fontsize'   : 14}
plt.title("Boston Citizens 1776", font)


nx.draw_networkx_nodes(G,pos,node_color='b',alpha=0.4)
nx.draw_networkx_edges(G,pos,alpha=0.4,node_size=0,width=1,edge_color='k')
plt.axis('off')
plt.show()

centrality=nx.eigenvector_centrality(G)


person_list = []
for node in centrality:
    person_list.append((node,centrality[node]))
    
sorted_person_list = person_list.sort(key=lambda x: x[1])

sorted(person_list, key=lambda x: x[1], reverse=True)





