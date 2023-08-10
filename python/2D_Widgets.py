from ipywidgets import interact

def f(x):
    return x

interact(f,x=(0,10))

interact(f,x=(0.0,10.0))

interact(f,x=True)

interact(f,x='Hello World')

get_ipython().magic('matplotlib inline')
from numpy import *
from matplotlib.pyplot import *

x=linspace(0,1)

def update(k=0):
    plot(x,x**k)
    show()

interact(update,k=(0,10))

t = linspace(0,1,1000)

def plot_sine(A=4,f=2,phi=0):
    plot(t,A*sin(2*pi*f*t + phi))
    axis([0,1,-6,6])
    show()

interact(plot_sine,A=(0,10), f=(1,10), phi=(0,2*pi))

import networkx as nx

# wrap a few graph generation functions so they have the same signature

def random_lobster(n, m, k, p):
    return nx.random_lobster(n, p, p / m)

def powerlaw_cluster(n, m, k, p):
    return nx.powerlaw_cluster_graph(n, m, p)

def erdos_renyi(n, m, k, p):
    return nx.erdos_renyi_graph(n, p)

def newman_watts_strogatz(n, m, k, p):
    return nx.newman_watts_strogatz_graph(n, k, p)

def plot_random_graph(n, m, k, p, generator):
    g = generator(n, m, k, p)
    nx.draw(g)
    show()

interact(plot_random_graph, n=(2,30), m=(1,10), k=(1,10), p=(0.0, 1.0, 0.001),
         generator={
             'lobster': random_lobster,
             'power law': powerlaw_cluster,
             'Newman-Watts-Strogatz': newman_watts_strogatz,
             u'Erdős-Rényi': erdos_renyi,
         });

