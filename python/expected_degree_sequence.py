import networkx as nx
get_ipython().magic('matplotlib inline')

n = 50 # n nodes

p = 0.1

w = [p*n for i in range(n)]      # w = p*n for all nodes

G = nx.expected_degree_graph(w)  # configuration model

dh = nx.degree_histogram(G)

low = min(d for n, d in G.degree())

nx.draw(G)

print"degree (#nodes) ****"
for i in range(low, len(dh)):
    bar=''.join(dh[i]*['*'])
    print i, dh[i], bar

