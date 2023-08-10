import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import ast
from sklearn.model_selection import train_test_split
import numpy as np

userDf = pd.read_csv('updated_users.csv')
print len(userDf['score'])
print max(userDf['score'])
print min(userDf['score'])

# elite users

eliteDf = userDf.nlargest(20, 'score')
eliteDf['friends'] = eliteDf['friends'].apply( lambda x : ast.literal_eval(x))

# creating graph and adding edges
eg = nx.Graph()
for i in eliteDf.index:
    for j in eliteDf.loc[i,'friends']:
        eg.add_edge(eliteDf.loc[i,'user_id'], j)

print len(eg.edges())
print len(eg.nodes())

# drawing graph

elite_nodes = []
for i in eliteDf.index:
    elite_nodes.append(eliteDf.loc[i, 'user_id'])
#print elite_nodes

extra_nodes = []
for node in eg.nodes():
    if node not in elite_nodes:
        extra_nodes.append(node)
print len(extra_nodes)

plt.figure(figsize=(20,20))
plt.axis('equal')
pos = nx.spring_layout(eg)
nx.draw_networkx(eg, pos, node_color = 'y', with_labels = False, nodelist=extra_nodes, node_size=110, alpha=0.8)
nx.draw_networkx(eg, pos, node_color = 'r', with_labels = False, nodelist=elite_nodes, node_size=350, aplha=0.8)
plt.savefig('spring_20_elite.png')
plt.show()

# save graph
fh = open("elite_20_adjlist",'wb')
nx.write_adjlist(eg, fh)

# calculating betweenness of each node
betweenness = nx.closeness_centrality(eg)
print 'done'
for i in elite_nodes:
    print betweenness[i]

# the betweenness values of elite users found by our classifier is generally high
print sorted(betweenness.values())

# normal users

normDf = userDf.nsmallest(20000, 'score')
normDf['friends'] = normDf['friends'].apply(lambda x : ast.literal_eval(x))

# creating graph and adding edges
ng = nx.Graph()
# add edges
for i in normDf.index:
    for j in normDf.loc[i,'friends']:
        ng.add_edge(normDf.loc[i,'user_id'], j)
# add nodes
for i in normDf.index:
    ng.add_node(normDf.loc[i,'user_id'])

print len(ng.edges())
print len(ng.nodes())

# drawing graph

plt.figure(figsize=(20,20))
plt.axis('equal')
nx.draw_spring(ng, node_color = 'g', with_labels = False, alpha=0.5)
plt.savefig('spring_15000_norm.png')
plt.show()
# the figure shows a very graph that is not well connected



