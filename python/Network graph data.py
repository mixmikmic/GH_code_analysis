import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

G=nx.Graph()

inputedges = pd.read_csv('edges.csv')
inputnodes = pd.read_csv('nodes.csv')

for i,r in inputnodes.iterrows():
    G.add_node(r['ID'])
    
for i,r in inputedges.iterrows():
    G.add_edge(r['Source'],r['Target'], weight=r['Weight'])

pos=nx.spring_layout(G, k=0.04, iterations=10, scale=100)
#pos=nx.shell_layout(G)
#pos=nx.spectral_layout(G)
#pos=nx.random_layout(G)

nx.set_node_attributes(G,'pos',pos)

pos[-2] = [100.0, 0.0]
pos[-1] = [0.0, 100.0]

nx.draw(G, pos=pos)
plt.savefig("network_graph.png")
plt.show()

positions = pd.DataFrame(pos).transpose()
positions.columns = ['X','Y']

positions.to_csv('nodepositions.csv', encoding='utf-16', index_label='ID')

