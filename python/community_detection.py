

G = small_graph()
write_graph(G, "net.html")

print(nx.adjacency_matrix(G))

print(nx.laplacian_matrix(G))

print(nx.normalized_laplacian_matrix(G))









G = block_graph(.3, 5)

weights = nx.fiedler_vector(G, normalized = True)

for n in G.nodes():
    G.node[n]["w"] = weights[n] / math.sqrt(G.degree(n)) * 100

write_graph(G, "net.html")





G = block_graph(.3, 5)
s = {n: 1 if n in range(0,1) else 0 for n in G.nodes()}

weights = nx.pagerank(G, alpha = .1, personalization = s)

for n in G.nodes():
    G.node[n]["w"] = weights[n] / G.degree(n) * 100000

write_graph(G, "net.html")

























def small_graph():
    G = nx.Graph()
    G.add_edge(0,1)
    G.add_edge(0,2)
    G.add_edge(0,3)
    G.add_edge(1,3)
    
    for n in G.nodes():
        G.node[n]["w"] = 1
    
    return G

import json
import math
from random import randint
import networkx as nx

def block_graph(p, m):
    #Create four blocks
    G1 = nx.binomial_graph(50, p)
    G2 = nx.binomial_graph(50, p)
    G3 = nx.binomial_graph(50, p)
    G4 = nx.binomial_graph(50, p)

    #Add them all to the same network
    G = nx.disjoint_union(G1, G2)
    G = nx.disjoint_union(G, G3)
    G = nx.disjoint_union(G, G4)
    
    #Add 'weight' attribute to every node
    for n in G.nodes():
        G.node[n]["w"] = 1

    #Add edges between blocks
    for i in range(0,m):
        G.add_edge(randint(0,49), randint(50,99))
        G.add_edge(randint(0,49), randint(100,149))
        G.add_edge(randint(0,49), randint(150,199))
        G.add_edge(randint(50,99), randint(100,149))
        G.add_edge(randint(50,99), randint(150, 199))
        G.add_edge(randint(100,149), randint(150,199))
    
    return G

def write_graph(G, file):
    #Convert to JSON for d3
    G_json = json.dumps({
        "nodes": [{"name": n, "w": G.node[n]["w"]} for n in G.nodes()],
        "links": [{"source": e[0], "target": e[1]} for e in G.edges()]
    }, indent = 4)
    
    #Write to file
    with open("template.html") as viz:
        new_viz = viz.read().replace("{{graph}}", G_json)

    with open(file, "w") as viz2:
        viz2.write(new_viz)

G = block_graph(.3, 5)

write_graph(G, "net.html")



