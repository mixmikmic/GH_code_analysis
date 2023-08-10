import numpy as np # load the library
a = np.linspace(-np.pi, np.pi, 100) # Create array 
b = np.cos(a) # Apply cosine to each element of a
c = np.ones(25) # An array of 25 ones
np.dot(c,c) # Compute inner product

from scipy.stats import norm
from scipy.integrate import quad
phi = norm()
value, error = quad(phi.pdf, -2,2) # Integrate using Gaussian quadrature
value

from sympy import Symbol
x, y = Symbol('x'), Symbol('y') # Treat 'x' and 'y' as algebraic symbols
print(x + x + x + y)

expression = (x+y)**2
expression.expand()

from sympy import solve
# Solving polynomials
solve(x**2 + x + 2)

from sympy import limit, sin, diff
limit(1 / x, x, 0)

limit(sin(x)/x , x, 0)

diff(sin(x),x)

import pandas as pd
import scipy as sp
data = sp.randn(5, 2) # Create 5x2 matrix of random numbers
dates = pd.date_range('28/12/2010', periods=5)
df = pd.DataFrame(data, columns=('price', 'weight'), index=dates)
print(df)

df.mean()

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

G = nx.random_geometric_graph(200,0.12) # Generate random graph
pos = nx.get_node_attributes(G, 'pos') # Get positions of nodes
# Find node nearest the center point (0.5, 0.5)
dists = [(x - 0.5)**2 + (y - 0.5)**2 for x,y in list(pos.values())]
ncenter = np.argmin(dists)

# Plot graph, coloring by path length from central node
p = nx.single_source_shortest_path_length(G, ncenter)
plt.figure()
nx.draw_networkx_edges(G, pos, alpha=0.4)
nx.draw_networkx_nodes(G, pos, nodelist=list(p.keys()), node_size=120, alpha=0.5,
                          node_color=list(p.values()), cmap=plt.cm.jet_r)

plt.show()



