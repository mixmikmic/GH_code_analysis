import networkx as nx

from networkx.exception import NetworkXError

class AntiGraph(nx.Graph):
    """
    Class for complement graphs.
    The main goal is to be able to work with big and dense graphs with
    a low memory foodprint.
    In this class you add the edges that *do not exist* in the dense graph,
    the report methods of the class return the neighbors, the edges and 
    the degree as if it was the dense graph. Thus it's possible to use
    an instance of this class with some of NetworkX functions. 
    """

    all_edge_dict = {'weight': 1}
    def single_edge_dict(self):
        return self.all_edge_dict
    edge_attr_dict_factory = single_edge_dict

    def __getitem__(self, n):
        """Return a dict of neighbors of node n in the dense graph.
        Parameters
        ----------
        n : node
           A node in the graph.
        Returns
        -------
        adj_dict : dictionary
           The adjacency dictionary for nodes connected to n.
        """
        return dict((node, self.all_edge_dict) for node in
                    set(self.adj) - set(self.adj[n]) - set([n]))

    def neighbors(self, n):
        """Return an iterator over all neighbors of node n in the 
           dense graph.
        """
        try:
            return iter(set(self.adj) - set(self.adj[n]) - set([n]))
        except KeyError:
            raise NetworkXError("The node %s is not in the graph."%(n,))

    def degree(self, nbunch=None, weight=None):
        """Return an iterator for (node, degree) or degree for single node.
        The node degree is the number of edges adjacent to the node.
        Parameters
        ----------
        nbunch : iterable container, optional (default=all nodes)
            A container of nodes.  The container will be iterated
            through once.
        weight : string or None, optional (default=None)
           The edge attribute that holds the numerical value used 
           as a weight.  If None, then each edge has weight 1.
           The degree is the sum of the edge weights adjacent to the node.
        Returns
        -------
        deg:
            Degree of the node, if a single node is passed as argument.
        OR
        nd_iter : an iterator
            The iterator returns two-tuples of (node, degree).
        See Also
        --------
        degree
        Examples
        --------
        >>> G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_path([0,1,2,3])
        >>> G.degree(0) # node 0 with degree 1
        1
        >>> list(G.degree([0,1]))
        [(0, 1), (1, 2)]
        """
        if nbunch in self:
            nbrs = {v: self.all_edge_dict for v in set(self.adj) -                     set(self.adj[nbunch]) - set([nbunch])}
            if weight is None:
                return len(nbrs) + (nbunch in nbrs)
            return sum((nbrs[nbr].get(weight, 1) for nbr in nbrs)) +                               (nbunch in nbrs and nbrs[nbunch].get(weight, 1))

        if nbunch is None:
            nodes_nbrs = ((n, {v: self.all_edge_dict for v in
                            set(self.adj) - set(self.adj[n]) - set([n])})
                            for n in self.nodes())
        else:
            nodes_nbrs = ((n, {v: self.all_edge_dict for v in
                            set(self.nodes()) - set(self.adj[n]) - set([n])})
                            for n in self.nbunch_iter(nbunch))

        if weight is None:
            def d_iter():
                for n,nbrs in nodes_nbrs:
                    yield (n,len(nbrs)+(n in nbrs)) # return tuple (n,degree)
        else:
            def d_iter():
                # AntiGraph is a ThinGraph so all edges have weight 1
                for n,nbrs in nodes_nbrs:
                    yield (n, sum((nbrs[nbr].get(weight, 1) for nbr in nbrs)) +
                                  (n in nbrs and nbrs[n].get(weight, 1)))
        return d_iter()

    def adjacency(self):
        """Return an iterator of (node, adjacency set) tuples for all nodes
           in the dense graph.
        This is the fastest way to look at every edge.
        For directed graphs, only outgoing adjacencies are included.
        Returns
        -------
        adj_iter : iterator
           An iterator of (node, adjacency set) for all nodes in
           the graph.
        """
        for n in self.adj:
            yield (n, set(self.adj) - set(self.adj[n]) - set([n]))

Gnp = nx.gnp_random_graph(20,0.8)

Anp = AntiGraph(nx.complement(Gnp))

Gd = nx.davis_southern_women_graph()

Ad = AntiGraph(nx.complement(Gd))

Gk = nx.karate_club_graph()

Ak = AntiGraph(nx.complement(Gk))

pairs = [(Gnp, Anp), (Gd, Ad), (Gk, Ak)]

for G, A in pairs:
    gc = [set(c) for c in nx.connected_components(G)]
    ac = [set(c) for c in nx.connected_components(A)]
    for comp in ac:
        assert comp in gc

for G, A in pairs:
    gc = [set(c) for c in nx.biconnected_components(G)]
    ac = [set(c) for c in nx.biconnected_components(A)]
    for comp in ac:
        assert comp in gc

for G, A in pairs:
    node = list(G.nodes())[0]
    nodes = list(G.nodes())[1:4]
    assert G.degree(node) == A.degree(node)
    assert sum(d for n, d in G.degree()) == sum(d for n, d in A.degree())
    assert sum(d for n, d in A.degree()) == sum(d for n, d in A.degree(weight='weight'))
    assert sum(d for n, d in G.degree(nodes)) == sum(d for n, d in A.degree(nodes))

