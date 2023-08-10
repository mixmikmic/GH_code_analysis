import DSGRN

network = DSGRN.Network("network.txt")

print(network)

print(network.graphviz())

import graphviz

graph = graphviz.Source(network.graphviz())

graph

parametergraph = DSGRN.ParameterGraph(network)

print("There are " + str(parametergraph.size()) + " nodes in the parameter graph.")

parameterindex = 34892  # An integer in [0,32592)

parameter = parametergraph.parameter(parameterindex)

parameter

print(parameter)

domaingraph = DSGRN.DomainGraph(parameter)

domaingraph

graphviz.Source(domaingraph.graphviz())

print(domaingraph.coordinates(5)) # ... I wonder what region in phase space domain 5 corresponds to.

morsedecomposition = DSGRN.MorseDecomposition(domaingraph.digraph())

graphviz.Source(morsedecomposition.graphviz())

[ morsedecomposition.morseset(i) for i in range(0,morsedecomposition.poset().size()) ]

morsegraph = DSGRN.MorseGraph()

morsegraph.assign(domaingraph, morsedecomposition)

morsegraph

print(morsegraph)

graphviz.Source(morsegraph.graphviz())

def labelstring(L,D):
  """
  Inputs: label L, dimension D
  Outputs:"label" output L of DomainGraph is converted into a string with "I", "D", and "?"
  """
  return ''.join([ "D" if L&(1<<d) else ("I" if L&(1<<(d+D)) else "?") for d in range(0,D) ])

{ v : labelstring(domaingraph.label(v),domaingraph.dimension()) for v in range(0,domaingraph.digraph().size())}

morseset_of_interest = morsedecomposition.morseset(1)

{ v : labelstring(domaingraph.label(v),domaingraph.dimension()) for v in morseset_of_interest}



