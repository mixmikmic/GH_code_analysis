import networkx
import math
import numpy

import pandas as pd

import matplotlib as mpl
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'svg'")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmap

class Dataset(object):
    """Represent a dataset of numbers drawn from an underlying distribution"""
    
    def __init__( self, n ):
        """Initialise to a dataset.
    
        :param n: the number of values in the dataset"""
        self._n = n
        self._gen = 0
        
    def __iter__( self ):
        """Use ourselves as the iterator.
        
        :returns: self"""
        return self

    def probabilityOf( self, v ):
        """The probability of a value appearing is taken directly from the
        probability mass function. This is a placeholder to be overridden
        in sub-classes.
        
        :param v: the value
        returns: the probability of this value occurring in the distribution"""
        raise Error("probabilityOf() not overridden in sub-class")

    def _candidate( self ):
        """Generate a candidate value. This is a placeholder to be overridden
        in sub-classes.
        
        :returns: a candidate random number"""
        raise Error("_candidate() not overridden in sub-class")

    def next( self ):
        """Return the next element from the dataset.
        
        :returns: the next random number drawn from the dataset"""
        if self._gen >= self._n:
            raise StopIteration
        else:
            v = self._candidate()
            if numpy.random.random() < self.probabilityOf(v):
                self._gen = self._gen + 1
                return v
            else:
                return self.next()
        
    def values( self ):
        """Return a list of all the elements remaining in the dataset. Typically
        used instead of iterating.
        
        :returns: the remaining values"""
        vs = []
        for v in self:
            vs.append(v)
        return vs

class InvalidDistribution(Exception):
    """Exception raised when we encounter an invalid probability distribution"""
    
    def __init__( self, msg ):
        self.message = msg
    
    def __str__( self ):
        return "Invalid distribution: " + self.message

class DiscreteDataset(Dataset):
    """Represent a dataset drawn against a given probability mass function"""
    
    # a value small enough that statistical errors of this magnitude
    # don't matter: one in a billion, by default
    _eps = 1.0e-9
    
    def __init__( self, n, dist ):
        """Initalise the dataset with the given discrete probabilities, to
        generate values in the range [0 ... len(dist) - 1].
        
        :param n: the size of the dataset
        :param dist: the probability mass function"""
        super(DiscreteDataset, self).__init__(n)
        
        # check given distribution is properly normalised
        if abs(1.0 - sum(dist)) < self._eps:
            self._dist = dist
        else:
            raise InvalidDistribution("Component probabilities do not sum to 1")
    
    def _candidate( self ):
        """Generate a candidate value in the range [0 ... len(dist) - 1].
        
        :returns: a candidate random numbr"""
        return int(numpy.random.random() * len(self._dist))
    
    def probabilityOf( self, v ):
        """The probability of a value appearing.
        
        :param v: the value
        :returns: the probbaility of this value"""
        return self._dist[v]

fig = plt.figure(figsize = (8, 5))
d = DiscreteDataset(1000, [ 0.25, 0.3, 0.15, 0.3 ])
plt.hist(d.values())
plt.xlabel("$k$")
plt.ylabel("$N_k$")
plt.title('Degree histogram for discrete dataset')
plt.show()

def degree_graph( dist ):
    """Generate a graph with node degrees taken from the given dataset.
    
    :param dist: the distribution for node degrees
    :returns: a network with this degree distribution"""
    
    # create the empty graph
    g = networkx.Graph()
    
    # populate the graph with nodes, capturing the number of edge stubs
    # desired on each
    i = 0
    ws = []
    dsum = 0
    for dd in dist:
        v = g.add_node(i)
        if dd > 0:
            ws.append((i, dd))
            i = i + 1
            dsum = dsum + dd

    # if the sum of degrees is odd, bump the degree of a random node
    # (the assumption being that this will make little difference)
    if dsum % 2 <> 0:
        i = int(numpy.random.random() * len(ws))
        (v, d) = ws[i]
        ws[i] = (v, d + 1)
            
    # wire-up edges between random nodes according to the remaining edge stubs
    while len(ws) > 1:
        si = int(numpy.random.random() * len(ws))
        ei = si
        while ei == si:
            # avoid self-loops
            # sd: should we do this or not?
            ei = int(numpy.random.random() * len(ws))
            
        # add edge between chosen nodes
        (s, ss) = ws[si]
        (e, es) = ws[ei]
        g.add_edge(s, e)
        
        # reduce the edge stubs on both nodes
        if ss > 1:
            ws[si] = (s, ss - 1)
        else:
            ws.pop(si)
            if si < ei:
                ei = ei - 1
        if es > 1:
            ws[ei] = (e, es - 1)
        else:
            ws.pop(ei)

    # if we have a node left, forget it (the assumption being that
    # a deviation 1 from the distribution won't matter, which is true
    # for networks of thousands of nodes or more)
        
    return g

g = degree_graph(DiscreteDataset(100, [ 0.25, 0.3, 0.15, 0.3 ]))

fig = plt.figure(figsize=(10, 8))
ax = fig.gca()
ax.set_xlim([-0.2, 1.2])      # axes bounded around 1
ax.set_ylim([-0.2, 1.2])
ax.grid(False)                # no grid
ax.get_xaxis().set_ticks([])  # no ticks on the axes
ax.get_yaxis().set_ticks([])

# run the spring layout algorithm over the network
pos = networkx.spring_layout(g, iterations = 100, k = 2/math.sqrt(g.order()))

# draw the network using the computed positions for the nodes
networkx.draw_networkx_edges(g, pos, width = 1, alpha = 0.4)
networkx.draw_networkx_nodes(g, pos, node_size=100, alpha = 1, linewidths = 0.5)
plt.show()

class PoissonDataset(Dataset):
    """Represent a dataset drawn against a Poisson distribution"""

    def __init__( self, k, n ):
        """Initalise the dataset with the given size and mean degree.
        
        :param k: the mean degree of the underlying Poisson distribution
        :param n: the size of the dataset"""
        super(PoissonDataset, self).__init__(n)
        self._meanDegree = k
        
    def _candidate( self ):
        """Generate a candidate value.
        
        :returns: a candidate degree"""
        return int(numpy.random.random() * self._meanDegree * 20)
    
    def probabilityOf( self, v ):
        """The probability of a value appearing, taken from the
        underlying binomial distribution.
        
        :param v: the value
        :returns: the probability of this value appearing"""
        return (pow(self._meanDegree, v) * math.exp(-self._meanDegree)) / math.factorial(v)

N = 1000
kmean = 5

g = degree_graph(PoissonDataset(kmean, N))

fig = plt.figure(figsize = (8, 5))
ks = g.degree().values()
plt.hist(ks, bins = max(ks))
plt.xlabel("$k$")
plt.ylabel("$N_k$")
plt.title('Histogram of Poisson node degrees, $N = {n}, \\langle k \\rangle = {k}$'.format(n = N, k = kmean))
plt.show()

# generate the degree sequence
nds = numpy.random.poisson(kmean, N)

# wire-up the network
g2 = networkx.configuration_model(nds)

# plot the same graph as above to check degree distribution
fig = plt.figure(figsize = (8, 8))
ks = g2.degree().values()
plt.hist(ks, bins = max(ks))
plt.xlabel("$k$")
plt.ylabel("$N_k$")
plt.title('Histogram of Poisson node degrees, $N = {n}, \\langle k \\rangle = {k}$'.format(n = N, k = kmean))
plt.show()

