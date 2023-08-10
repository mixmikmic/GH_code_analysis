
get_ipython().system('python -m pip install --upgrade pip')


get_ipython().system('pip install networkx==1.11')


from IPython.display import HTML

HTML('<img src="../../saves/png/degree_centrality_undirected_networks.png" />')


import networkx as nx

karate_club_graph = nx.karate_club_graph()
karate_club_graph = nx.convert_node_labels_to_integers(karate_club_graph, first_label=1)


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Draw the graph using the kamada kawai layout
plt.figure(figsize=(18, 8))
nx.draw_networkx(karate_club_graph, node_color='xkcd:off white',
                 width=2.5, node_size=600)


degree_centrality = nx.degree_centrality(karate_club_graph)
print(degree_centrality[34])
print(degree_centrality[33])


disconnected_dipraph = nx.DiGraph()
disconnected_dipraph.add_nodes_from([chr(i) for i in range(65, 80)])
disconnected_dipraph.add_edges_from([('A', 'B'), ('A', 'E'), ('A', 'N'), ('B', 'C'), ('B', 'E'),
                                     ('C', 'A'), ('C', 'D'), ('D', 'B'), ('D', 'E'),
                                     ('E', 'C'), ('E', 'D'), ('F', 'G'), ('G', 'A'), ('G', 'J'),
                                     ('H', 'G'), ('H', 'I'), ('I', 'F'), ('I', 'G'), ('I', 'H'), ('I', 'J'),
                                     ('J', 'F'), ('J', 'O'), ('K', 'L'), ('K', 'M'), ('L', 'M'),
                                     ('N', 'L'), ('N', 'O'), ('O', 'J'), ('O', 'K'), ('O', 'L')])


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Draw the graph using the kamada kawai layout
plt.figure(figsize=(18, 8))
nx.draw_networkx(disconnected_dipraph, node_color='xkcd:light green',
                 alpha=.75, width=2.5, node_size=600)


# The degree centrality for a node v is the fraction of nodes it
# is connected to
degree_centrality = nx.degree_centrality(disconnected_dipraph)

# The in-degree centrality for a node v is the fraction of nodes its
# incoming edges are connected to
in_degree_centrality = nx.in_degree_centrality(disconnected_dipraph)
print(in_degree_centrality['A'])
print(in_degree_centrality['L'])

# The out-degree centrality for a node v is the fraction of nodes its
# outgoing edges are connected to
out_degree_centrality = nx.out_degree_centrality(disconnected_dipraph)
print(out_degree_centrality['A'])
print(out_degree_centrality['L'])


from IPython.display import HTML

HTML('<img src="../../saves/png/disconnect_nodes.png" />')


# Closeness centrality of node L is the reciprocal of the
# average shortest path distance to L over all n-1 reachable nodes
print('{:.3} = ({}-1)/{} = {:.3}'.format(nx.closeness_centrality(karate_club_graph, u=32, normalized=False),
                                 len(karate_club_graph.nodes()),
                                 sum(nx.shortest_path_length(karate_club_graph, source=32).values()),
                                 (len(karate_club_graph.nodes())-1)/sum(nx.shortest_path_length(karate_club_graph,
                                                                                                source=32).values())))

get_ipython().run_line_magic('pinfo', 'nx.shortest_path_length')


# Closeness centrality of node L is the reciprocal of the
# average shortest path distance to L over all n-1 reachable nodes
nx.closeness_centrality(disconnected_dipraph, u='L', wf_improved=False, reverse=True)


# Wasserman and Faust propose an improved formula for graphs with
# more than one connected component. The result is "a ratio of the
# fraction of actors in the group who are reachable, to the average
# distance" from the reachable actor
nx.closeness_centrality(disconnected_dipraph, u='L', wf_improved=True, reverse=True)


import numpy as np

disconnected_dipraph = nx.DiGraph()
node_list = [chr(i) for i in range(65, 69)]
disconnected_dipraph.add_nodes_from(node_list)
disconnected_dipraph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])

# Draw the graph using the kamada kawai layout
plt.figure(figsize=(18, 2))
pos = {'A': np.array([-0.5, 0]),
       'B': np.array([0, 0]),
       'C': np.array([0.5, 0]),
       'D': np.array([1, 0])}
nx.draw_networkx(disconnected_dipraph, pos, node_color='xkcd:baby blue',
                 alpha=.75, width=2.5, node_size=1000)
for node in node_list:
    print(node, nx.closeness_centrality(disconnected_dipraph, u=node, wf_improved=False, reverse=True))


for node in node_list:
    print(node, nx.closeness_centrality(disconnected_dipraph, u=node, wf_improved=True, reverse=True))


betweenness_graph = nx.Graph()
node_list = [chr(i) for i in range(65, 72)]
betweenness_graph.add_nodes_from(node_list)
betweenness_graph.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D'),
                                  ('D', 'E'), ('E', 'F'), ('E', 'G'), ('F', 'G')])


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Draw the graph using the kamada kawai layout
plt.figure(figsize=(18, 8))
nx.draw_networkx(betweenness_graph, node_color='xkcd:light green',
                 alpha=.75, width=2.5, node_size=600)


betweenness_centrality = nx.betweenness_centrality(betweenness_graph, normalized=False, endpoints=False)
for node in node_list:
    print(node, betweenness_centrality[node])


nx.betweenness_centrality(betweenness_graph, normalized=True, endpoints=False)['D']


import operator

betweenness_centrality = nx.betweenness_centrality(karate_club_graph, normalized=False, endpoints=False)
sorted(betweenness_centrality.items(), key=operator.itemgetter(1), reverse=True)[0:5]


get_ipython().run_line_magic('pprint', '')

dir(operator)



