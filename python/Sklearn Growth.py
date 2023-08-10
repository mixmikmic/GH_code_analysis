import os
import csv
import datetime
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter

get_ipython().magic('matplotlib inline')

INPATH     = 'sklearncommits_mini.txt' # full set was slow so clipped last 2yrs
IMGPATH    = 'sklearngraph.png'
GRAPHML    = 'sklearngraph.graphml'
DATEFORMAT = ' %a %b %d %H:%M:%S %Y %z'

g = nx.Graph(name="Sklearn Commits")

ifile   = open(os.path.join('data', INPATH), 'r')
commits  = csv.reader(ifile)

for commit in commits:
    commit_hash = commit[0]     # Uniquely identifies a commit
    parent_hashes = commit[1]
    contributor = commit[2]
    try: 
        commit_timestamp = datetime.datetime.strptime(commit[3], DATEFORMAT).date()
    except:
        pass
    
    g.add_node(commit_hash, timestamp=commit_timestamp)  # add other elements?
    g.add_node(contributor)
    g.add_edge(contributor, commit_hash, label='contributor')

    for parent in parent_hashes:
        g.add_node(parent, timestamp=commit_timestamp)
        delta = g.node[parent]['timestamp']-g.node[commit_hash]['timestamp']
        g.add_edge(parent, commit_hash, label='parent', weight=delta.total_seconds())

center, degree = sorted(g.degree().items(), key=itemgetter(1), reverse=True)[0]
# A special type of subgraph
ego = nx.ego_graph(g, center)

pos = nx.spring_layout(g)
nx.draw(g, pos, node_color='#0080C9', edge_color='#cccccc', node_size=50)
nx.draw_networkx_nodes(g, pos, nodelist=[center], node_size=100, node_color="r")
plt.show()

for component in nx.connected_components(g):
    print(len(component))

degree_sequence=sorted(nx.degree(g).values(),reverse=True) # degree sequence

plt.loglog(degree_sequence,'b-',marker='.')
plt.title("Degree rank plot")
plt.ylabel("degree")
plt.xlabel("rank")

print("Order: %i" % g.number_of_nodes())
print("Size: %i" % g.number_of_edges())
print("Clustering: %0.5f" % nx.average_clustering(g))
print("Transitivity: %0.5f" % nx.transitivity(g))

hairball = nx.subgraph(g, [x for x in nx.connected_components(g)][0])
print("Average shortest path: %0.4f" % nx.average_shortest_path_length(hairball))

print(nx.density(g))

nx.draw(g)

# plt.savefig(os.path.join('images', IMGPATH))

nx.write_graphml(g,GRAPHML)



