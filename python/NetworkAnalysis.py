import pandas as pd
import networkx as nx
import itertools

import os
os.chdir('set_the_working_directory')

### reading a 2-mode network
two_mode = nx.read_weighted_edgelist('the_data_file', delimiter=',', nodetype=str, encoding='utf-8')
two_mode.edges(data=True)

# left: provider ids, right: patient ids
left, right = nx.bipartite.sets(two_mode)
#print list(right) 
#print list(left) 

### projecting the 2-mode network to 1-mode network
one_mode = nx.projected_graph(two_mode, left)

print("Number of edges:  %.0f" %one_mode.size())
print("Number of nodes:  %.0f" %one_mode.number_of_nodes())

### the largest connected component
largest_component = max(nx.connected_component_subgraphs(one_mode), key=len)
print("Number of edges in the largest component:  %.0f" %largest_component.size())
print("Number of nodes in the largest component:  %.0f" %largest_component.number_of_nodes())

### calculating clustering coefficient and path nelgth in the actual network
g_cc = nx.average_clustering(largest_component) #actual clustering coefficient
g_pl = nx.average_shortest_path_length(largest_component) #actual average path length
print("=== The Actual Network ===")
print(u"    \u2022 Average clustering coefficient: %.2f" %g_cc)
print(u"    \u2022 Average shortest path length: %.2f" %g_pl)

### Erdos-Renyi random graphs
g_erdos = nx.erdos_renyi_graph(one_mode.number_of_nodes(), 0.05)
print("=== Erdos-Renyi Random Network ===")
print(u"    \u2022 Number of edges:  %.0f" %g_erdos.size())
print(u"    \u2022 Number of nodes:  %.0f" %g_erdos.number_of_nodes())

### calculating the network properties for the Erdos-Renyi graph
g_erdos_cc = nx.average_clustering(g_erdos) #random clustering coefficient
g_erdos_pl = nx.average_shortest_path_length(g_erdos) #random path length
print(u"    \u2022 Average clustering coefficient: %.2f" %g_erdos_cc)
print(u"    \u2022 Average shortest path length: %.2f" %g_erdos_pl)

print("\nSmall-world indicator based on ER random graph: %.2f" %((g_cc/g_erdos_cc)/(g_pl/g_erdos_pl)))

### Configuration graphs
print("=== Configuration Network ===")
degree_sequence = list(dict(nx.degree(one_mode)).values()) # degree sequence
#print(u"    \u2022 Degree sequence %s" % degree_sequence)

# creating the configuration graph with same degree distribution
g_conf = nx.configuration_model(degree_sequence)
# removing parallel edges
g_conf = nx.Graph(g_conf)
# removing self-loops
g_conf.remove_edges_from(g_conf.selfloop_edges())

print(u"    \u2022 Number of edges:  %.0f" %g_conf.size())
print(u"    \u2022 Number of nodes:  %.0f" %g_conf.number_of_nodes())

### calculating the network properties for the configuration graph
g_conf_cc = nx.average_clustering(g_conf) #configuration graph clustering coefficient
g_conf_pl = nx.average_shortest_path_length(g_conf) #configuration graph average path length
print(u"    \u2022 Average clustering coefficient: %.2f" %g_conf_cc)
print(u"    \u2022 Average shortest path length: %.2f" %g_conf_pl)

print("\nSmall-world indicator based on configuration graph: %.2f" %((g_cc/g_conf_cc)/(g_pl/g_conf_pl)))
################################

### Calculating number of triangles
def g_iterator(network):
    for node in network.nodes():  # change "nodes" to "nodes_iter" in networkx versions < 2.0
        neighbors = network.neighbors(node)
        for pair in itertools.combinations(neighbors, 2):
            yield(node, pair)

def count_triangles(network):
    count = 0
    for p in g_iterator(network):
        if (network.has_edge(p[1][0], p[1][1]) or network.has_edge(p[1][1], p[1][0])):
            count += 1
    return count/3

print("Number of triangles: %.0f" %count_triangles(one_mode))

### Calculating network density
print("Density: %.3f" %nx.density(one_mode))

### Structural holes, aggregate constraint
# Returns the sum of the weights of the edge from `u` to `v` and the edge from `v` to `u` in the network
def mutual_weight(network, u, v, weight=None):
    try:
        a_uv = network[u][v].get(weight, 1)
    except KeyError:
        a_uv = 0
    try:
        a_vu = network[v][u].get(weight, 1)
    except KeyError:
        a_vu = 0
    return (a_uv + a_vu)

# Returns normalized mutual weight of the edges from u to v with respect to the mutual weights of the neighbors of u in network
def normalized_mutual_weight(network, u, v, norm=sum):
    scale = norm(mutual_weight(network, u, w)
                 for w in set(nx.all_neighbors(network, u)))
    return 0 if scale == 0 else mutual_weight(network, u, v) / scale

# Returns the local constraint on the node u with respect to the node v in the network
def local_constraint(network, u, v):
    nmw = normalized_mutual_weight
    direct = nmw(network, u, v)
    indirect = sum(nmw(network, u, w) * nmw(network, w, v)
                   for w in set(nx.all_neighbors(network, u)))
    return (direct + indirect) ** 2

# Returns the constraint on all nodes in the network
def constraint(network):
    constraint = {}
    for v in network:
        # Constraint is not defined for isolated nodes
        if len(network[v]) == 0:
            constraint[v] = float('nan')
            continue
        constraint[v] = sum(local_constraint(network, v, n)
                            for n in set(nx.all_neighbors(network, v)))
    return constraint

cons = constraint(one_mode)
# converting the dictionary to dataframe
cons = pd.DataFrame(cons.items(), columns=['provider', 'constraint'])

# calculating the aggregate constraint for surgeons
df = cons[cons.provider.str.startswith(('sPRV'))]
surgeon_constraint_total = df['constraint'].sum()

# calculating the aggregate constraint for anesthesiologists
df = cons[cons.provider.str.startswith(('aPRV'))]
anesth_constraint_total = df['constraint'].sum()

# calculating the aggregate constraint for nurses
df = cons[cons.provider.str.startswith(('nPRV'))]
nurse_constraint_total = df['constraint'].sum()

